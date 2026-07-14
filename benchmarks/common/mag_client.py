"""
MAG Client — 与 Mem0Client 相同接口，使 benchmark 可直接切换
"""
import asyncio, logging, os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from mem0.configs.base import MemoryConfig
from mag.core import MAGMemory
from mag.config import MAGConfig

logger = logging.getLogger(__name__)


class MAGClient:
    """Async wrapper around MAGMemory，兼容 Mem0Client 接口"""

    def __init__(self, project_name: str = "mag_bench", **kw):
        # 从统一配置加载
        self._cfg = MAGConfig.from_env_file()

        config = MemoryConfig(**self._cfg.to_mem0_config(project_name))

        self.memory = MAGMemory(
            config,
            mag_enabled=self._cfg.mag_enabled,
            segmentation_strategy=self._cfg.segmentation_strategy,
            entity_strategy=self._cfg.entity_strategy,
            attention_strategy=self._cfg.attention_strategy,
            graph_config=self._cfg.to_graph_config(project_name),
            linear_rag_config=self._cfg.to_linear_rag_config(),
        )
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._loop = None
        self._last_timestamp: int = None
        self._done_file = f"/tmp/mag_{project_name}_done"
        flags = self._cfg.ablation_flags
        print(f"[MAGClient] CREATED project={project_name} backend=mag "
              f"flags={flags}", flush=True)

    @property
    def mode(self) -> str:
        return "mag"

    def is_ingested(self, user_id: str = "") -> bool:
        """检查本地标记文件是否存在"""
        return os.path.exists(self._done_file)

    def mark_ingested(self):
        """摄入完成后写标记文件"""
        with open(self._done_file, "w") as f:
            f.write("done\n")

    async def add(self, messages, user_id, timestamp=None, **kw):
        loop = asyncio.get_running_loop()
        from datetime import datetime, timezone
        ts = None
        if timestamp:
            ts = datetime.fromtimestamp(timestamp, tz=timezone.utc)

        # 检测 session 边界：timestamp 变了 → flush 上个 session 的积累
        if timestamp is not None and self._last_timestamp is not None and timestamp != self._last_timestamp:
            await self.flush_relations()
            if hasattr(self.memory, 'flush_edge_sentences'):
                self.memory.flush_edge_sentences()

        result = await loop.run_in_executor(
            self._executor,
            lambda: self.memory.add(messages, user_id=user_id, default_timestamp=ts),
        )
        self._last_timestamp = timestamp
        return result

    async def search(self, query, user_id, top_k=200, score_debug=False, **kw):
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            self._executor,
            lambda: self.memory.search(query, filters={"user_id": user_id}, top_k=top_k),
        )
        if result is None:
            return []
        raw = result.get("results", [])
        formatted = []
        for r in raw:
            sid = r.get("id", "")[:8]
            ts = r.get("created_at", "")[:10]
            text = r.get("memory", "")

            # 提取 entities: entities字段 > payload.entities > metadata.entities
            entities = (r.get("entities", [])
                        or r.get("payload", {}).get("entities", [])
                        or r.get("metadata", {}).get("entities", [])
                        or [])
            ents_str = ""
            if entities:
                top_ents = sorted(entities, key=lambda e: e.get("attention_weight", 0) if isinstance(e, dict) else 0, reverse=True)[:5]
                ents_str = ", ".join(
                    f"{e['name']}({e['attention_weight']:.1f})"
                    for e in top_ents if isinstance(e, dict)
                )

            # 结构化 memory 字段: [ISO Timestamp] RawText
            if ts:
                memory_str = f"[{ts}] {text}"
            else:
                memory_str = text

            entry = {
                "memory": memory_str,
                "score": r.get("score", 0),
                "id": r.get("id", ""),
                "entities": entities,
                "source": r.get("source", ""),  # 追踪检索来源
            }
            if r.get("created_at"):
                entry["created_at"] = r["created_at"]
            formatted.append(entry)
        formatted.sort(key=lambda x: x.get("score", 0), reverse=True)
        return formatted

    async def delete_user(self, user_id: str) -> bool:
        try:
            self.memory.delete_all(user_id=user_id)
            return True
        except Exception:
            return False

    async def flush_relations(self):
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(self._executor, self.memory.flush_relations)
        if result.get("relations_added", 0) > 0:
            self.mark_ingested()
        return result

    async def flush_edge_sentences(self):
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._executor, self.memory.flush_edge_sentences)

    async def close(self):
        self._executor.shutdown(wait=False)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        await self.close()
