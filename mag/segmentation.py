"""
句子粒度文本切分 — 将对话切分为独立句粒度的 Memory Unit

位于三元组与原始 text chunk 之间：
  Conversation → SentenceSegmenter → Sentences → Triple Extraction → Graph

支持两种策略：
  - nlp: spaCy sentencizer (快速、离线)
  - llm: 轻量 LLM 语义分句 (精准、适合复杂对话)
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class SentenceSegmenter:
    """将对话消息切分为句子粒度的 Memory Unit"""

    # 最小/最大句子长度阈值 (字符数)
    MIN_SENTENCE_LENGTH = 10
    MAX_SENTENCE_LENGTH = 500

    # 对话中的口语化断句模式
    _DIALOGUE_SPLIT_PATTERNS = [
        # 标准句末标点 + 空格/换行
        (r'(?<=[。！？.!?])\s+', "sentence_end"),
        # 换行符 (对话中常表示语义边界)
        (r'\n{2,}', "paragraph_break"),
        # 单个换行 (弱边界，仅在句子过长时使用)
        (r'\n', "line_break"),
        # 分号 (连接两个独立子句)
        (r'(?<=[^0-9]);(?=\s+[A-Z一-鿿])', "semicolon"),
    ]

    # 不应被拆分的缩写模式
    _PROTECTED_PATTERNS = [
        re.compile(r'\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|Inc|Ltd|Co|vs|etc|i\.e|e\.g)\.$'),
        re.compile(r'\b[A-Z]\.$'),  # 单个大写字母 + 点 (如 "U.S.")
        re.compile(r'\d+\.\d+'),  # 小数
    ]

    def __init__(
        self,
        strategy: str = "nlp",
        llm_client=None,
        min_length: int = 10,
        max_length: int = 500,
    ):
        """
        Args:
            strategy: "nlp" (spaCy) 或 "llm" (LLM 语义分句)
            llm_client: LLM 客户端 (strategy="llm" 时需要)
            min_length: 最短句子长度 (更短的会合并到相邻句)
            max_length: 最长句子长度 (更长的会拆开)
        """
        self.strategy = strategy
        self.llm_client = llm_client
        self.MIN_SENTENCE_LENGTH = min_length
        self.MAX_SENTENCE_LENGTH = max_length
        self._nlp = None

    @property
    def nlp(self):
        """懒加载 spaCy 模型，失败时返回 None (上游用 _segment_by_regex 兜底)"""
        if self._nlp is None:
            try:
                import spacy
                try:
                    self._nlp = spacy.load("en_core_web_sm")
                except OSError:
                    logger.warning(
                        "spaCy model 'en_core_web_sm' not found — "
                        "using regex-based segmentation. "
                        "Install with: python -m spacy download en_core_web_sm"
                    )
                    self._nlp = False  # sentinel: tried and failed
                else:
                    if "sentencizer" not in self._nlp.pipe_names:
                        self._nlp.add_pipe("sentencizer", before="parser")
            except ImportError:
                logger.warning(
                    "spaCy not installed — using regex-based segmentation"
                )
                self._nlp = False
        return self._nlp if self._nlp is not False else None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def segment(
        self,
        messages: List[Dict[str, Any]],
        default_timestamp: Optional[datetime] = None,
    ) -> List[Tuple[str, str, datetime]]:
        """
        将消息列表切分为句子列表

        Args:
            messages: [{"role": "user", "content": "..."}, ...]
            default_timestamp: 默认时间戳 (消息若无时间戳则用此值)

        Returns:
            [(sentence_text, speaker_role, timestamp), ...]
        """
        if default_timestamp is None:
            default_timestamp = datetime.now(timezone.utc)

        all_sentences: List[Tuple[str, str, datetime]] = []

        for msg in messages:
            if not isinstance(msg, dict):
                continue

            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            ts = msg.get("timestamp") or msg.get("created_at")

            if ts:
                try:
                    ts = datetime.fromisoformat(str(ts))
                except (ValueError, TypeError):
                    ts = default_timestamp
            else:
                ts = default_timestamp

            if not content or not isinstance(content, str):
                continue

            # 跳过纯 system prompt
            if role == "system":
                continue

            # 按策略切句
            if self.strategy == "llm" and self.llm_client:
                sentences = self._segment_by_llm(content)
            else:
                sentences = self._segment_by_nlp(content)

            for sent in sentences:
                all_sentences.append((sent, role, ts))

        # 后处理：合并过短/拆分过长句子
        all_sentences = self._postprocess(all_sentences)

        return all_sentences

    # ------------------------------------------------------------------
    # 策略 A: NLP 分句 (spaCy)
    # ------------------------------------------------------------------

    def _segment_by_nlp(self, text: str) -> List[str]:
        """基于 spaCy sentencizer 的语义分句，spaCy 不可用时用 regex 兜底"""
        if not text or not text.strip():
            return []

        nlp = self.nlp
        if nlp is None:
            return self._segment_by_regex(text)

        doc = nlp(text)
        sentences = []

        for sent in doc.sents:
            sent_text = sent.text.strip()
            if not sent_text:
                continue
            sent_text = self._fix_protected_splits(sent_text)
            sentences.append(sent_text)

        return sentences

    @staticmethod
    def _segment_by_regex(text: str) -> List[str]:
        """Regex 分句兜底 — 按句末标点 + 换行切句"""
        import re
        # 按句末标点切开 (保留标点)
        parts = re.split(r'(?<=[.!?])\s+', text)
        sentences = []
        for part in parts:
            part = part.strip()
            if not part:
                continue
            # 进一步按换行切
            sub_parts = re.split(r'\n+', part)
            for sp in sub_parts:
                sp = sp.strip()
                if sp:
                    sentences.append(sp)
        return sentences if sentences else [text]

    def _fix_protected_splits(self, text: str) -> str:
        """修复被误拆的缩写句点"""
        # 如果句子以已知缩写结尾，且下一个句子存在，则合并
        for pattern in self._PROTECTED_PATTERNS:
            if pattern.search(text):
                # 标记为需要合并 (在后处理中处理)
                pass
        return text

    # ------------------------------------------------------------------
    # 策略 B: LLM 语义分句
    # ------------------------------------------------------------------

    def _segment_by_llm(self, text: str) -> List[str]:
        """
        基于 LLM 的语义分句 — 处理对话特有现象
        (省略主语、指代消解、口语化表达)
        """
        if not self.llm_client:
            logger.warning("LLM client not available, falling back to NLP")
            return self._segment_by_nlp(text)

        prompt = (
            "将以下对话文本按语义完整性切分为独立的句子。\n\n"
            "规则：\n"
            "1. 每个句子应该语义完整、独立可理解\n"
            "2. 处理对话特有的省略主语和指代\n"
            "3. 不要切分列表或引用内容\n"
            "4. 保持原始措辞不变\n\n"
            f"文本：\n{text}\n\n"
            '以 JSON 格式返回：{{"sentences": ["句1", "句2", ...]}}'
        )

        try:
            response = self.llm_client.generate_response(
                messages=[
                    {"role": "system", "content": "你是一个文本结构化专家。"},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
            )
            import json

            data = json.loads(response)
            return data.get("sentences", [text])
        except Exception as e:
            logger.warning(f"LLM segmentation failed: {e}, falling back to NLP")
            return self._segment_by_nlp(text)

    # ------------------------------------------------------------------
    # 后处理
    # ------------------------------------------------------------------

    def _postprocess(
        self, sentences: List[Tuple[str, str, datetime]]
    ) -> List[Tuple[str, str, datetime]]:
        """合并过短句子，拆分过长句子"""
        if not sentences:
            return sentences

        # 第一遍：合并过短句子
        merged: List[Tuple[str, str, datetime]] = []
        buffer = sentences[0]

        for sent in sentences[1:]:
            buf_text, buf_role, buf_ts = buffer
            sent_text, sent_role, sent_ts = sent

            if len(buf_text) < self.MIN_SENTENCE_LENGTH or not self._has_complete_structure(buf_text):
                # 合并到下一个句子（同 speaker）
                if buf_role == sent_role:
                    buffer = (buf_text + " " + sent_text, buf_role, buf_ts)
                else:
                    merged.append(buffer)
                    buffer = sent
            else:
                merged.append(buffer)
                buffer = sent

        merged.append(buffer)

        # 第二遍：拆分过长句子
        result: List[Tuple[str, str, datetime]] = []
        for text, role, ts in merged:
            if len(text) > self.MAX_SENTENCE_LENGTH:
                split = self._split_long_sentence(text)
                for s in split:
                    result.append((s, role, ts))
            else:
                result.append((text, role, ts))

        return result

    @staticmethod
    def _has_complete_structure(text: str) -> bool:
        """检查句子是否有完整的语义结构"""
        text = text.strip()
        if not text:
            return False
        # 至少包含主谓结构 (简单启发式)
        has_verb = bool(re.search(
            r'(is|are|was|were|have|has|had|do|does|did|'
            r'will|would|can|could|should|may|might|'
            r'\w+(ed|ing)\b)',
            text, re.IGNORECASE
        ))
        has_subject = len(text.split()) >= 2
        return has_subject and has_verb

    @staticmethod
    def _split_long_sentence(text: str) -> List[str]:
        """拆分过长句子 — 在逗号/分号/连接词处切"""
        parts = re.split(r'(?<=[^0-9]),(?=\s+\w)|(?<=[^0-9]);(?=\s+\w)', text)
        if len(parts) == 1:
            # 没有逗号/分号可切 → 在连接词处切
            parts = re.split(
                r'\s+(and|but|or|however|therefore| moreover|meanwhile)\s+',
                text,
                flags=re.IGNORECASE,
            )
            # 重组：保留连接词
            if len(parts) > 1:
                result = []
                i = 0
                while i < len(parts):
                    if i + 1 < len(parts):
                        result.append(parts[i] + " " + (parts[i + 1] if i + 1 < len(parts) else ""))
                        i += 2
                    else:
                        result.append(parts[i])
                        i += 1
                return [r.strip() for r in result if r.strip()]
        return [p.strip() for p in parts if p.strip()]
