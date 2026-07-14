"""
上下文组装 — 将检索到的句子拼接为 LLM 可读的 Prompt 上下文

支持三种排序模式：
  - relevance: 按相关性分数降序 (默认)
  - chronological: 按时间升序 (故事线)
  - hybrid: 时间排序 + 相关性截断
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class ContextSynthesizer:
    """将精排后的记忆句子组装成 Prompt 上下文"""

    def __init__(
        self,
        max_tokens: int = 2000,
        order_by: str = "relevance",
        include_timestamps: bool = True,
        include_entities: bool = False,
        header_template: str = "[Relevant Past Memories]",
    ):
        """
        Args:
            max_tokens: 上下文最大 token 数 (粗略估算)
            order_by: "relevance" | "chronological" | "hybrid"
            include_timestamps: 是否显示时间戳
            include_entities: 是否显示关联实体
            header_template: 上下文头部文本
        """
        self.max_tokens = max_tokens
        self.order_by = order_by
        self.include_timestamps = include_timestamps
        self.include_entities = include_entities
        self.header_template = header_template

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def synthesize(
        self, query: str, memories: List[Dict[str, Any]]
    ) -> str:
        """
        组装上下文

        Args:
            query: 当前查询
            memories: 记忆列表，每条含 memory/id/score/...

        Returns:
            拼接后的上下文字符串
        """
        if not memories:
            return f"[Current Query]\n{query}"

        # 排序
        ordered = self._order(memories)

        # 拼接
        sections = [self.header_template + "\n"]
        token_est = 0

        for i, mem in enumerate(ordered):
            line = self._format_line(i + 1, mem)
            token_est += self._estimate_tokens(line)
            if token_est > self.max_tokens:
                break
            sections.append(line)

        sections.append(f"\n[Current Query]\n{query}")
        return "\n".join(sections)

    def synthesize_bare(self, memories: List[Dict[str, Any]]) -> List[str]:
        """
        仅返回格式化后的句子列表 (不做拼接截断)

        适用于需要自行控制上下文长度的场景
        """
        ordered = self._order(memories)
        return [self._format_line(i + 1, m) for i, m in enumerate(ordered)]

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _order(self, memories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """按指定策略排序"""
        if self.order_by == "chronological":
            return sorted(
                memories,
                key=lambda m: self._get_timestamp(m),
            )
        elif self.order_by == "hybrid":
            # 按时间分桶 (每月一桶)，桶内按相关性排序
            buckets: Dict[str, List[Dict[str, Any]]] = {}
            for m in memories:
                ts = self._get_timestamp(m)
                bucket_key = ts[:7] if len(ts) >= 7 else ts  # "2024-03"
                buckets.setdefault(bucket_key, []).append(m)
            # 桶间按时间升序
            result = []
            for key in sorted(buckets.keys()):
                # 桶内按 score 降序
                bucket_mems = sorted(
                    buckets[key],
                    key=lambda m: m.get("score", 0),
                    reverse=True,
                )
                result.extend(bucket_mems)
            return result
        else:
            # relevance: 按 score 降序 (默认)
            return sorted(
                memories,
                key=lambda m: m.get("score", 0),
                reverse=True,
            )

    def _format_line(self, index: int, mem: Dict[str, Any]) -> str:
        """格式化单条记忆"""
        memory_text = mem.get("memory", "")
        parts = [f"{index}."]

        if self.include_timestamps:
            ts = self._get_timestamp(mem)
            parts.append(f"[{ts}]")

        parts.append(memory_text)

        if self.include_entities:
            entities = mem.get("payload", {}).get("entities", [])
            if entities:
                entity_names = [e.get("name", "") for e in entities if isinstance(e, dict)]
                if entity_names:
                    parts.append(f" (entities: {', '.join(entity_names[:5])})")

        return " ".join(parts)

    @staticmethod
    def _get_timestamp(mem: Dict[str, Any]) -> str:
        """从 memory 中提取时间戳 (YYYY-MM-DD 格式)"""
        payload = mem.get("payload", {})
        ts = payload.get("created_at", "") or mem.get("created_at", "")
        if ts and len(ts) >= 10:
            return ts[:10]
        return "unknown"

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """粗略 token 估算：英文 ~4 chars/token，中文 ~1.5 chars/token"""
        import re
        ascii_chars = len(re.findall(r'[a-zA-Z0-9\s]', text))
        non_ascii = len(text) - ascii_chars
        return int(ascii_chars / 4 + non_ascii / 1.5)
