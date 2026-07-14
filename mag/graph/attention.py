"""
Entity Attention 权重计算 — 为句子中的每个实体分配核心程度权重

不区分 Subject/Entity，统一通过 Attention 权重表达实体在句子语义中的核心程度。

方案：
  A) Cross-Encoder: 对 (sentence, entity) pair 打分 (推荐)
  B) LLM 联合: 抽取 triple 时一同评估 centrality
  C) 句法位置: 基于依存树位置 (主语 > 宾语 > 介词短语)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from mag.schema import EntityWeight

logger = logging.getLogger(__name__)


class EntityAttentionScorer:
    """为句子中的每个实体计算 Attention 权重"""

    def __init__(
        self,
        strategy: str = "syntactic",
        llm_client=None,
        reranker_model=None,
    ):
        """
        Args:
            strategy: "syntactic" | "cross_encoder" | "llm"
            llm_client: LLM 客户端 (strategy="llm" 时)
            reranker_model: Cross-Encoder 模型 (strategy="cross_encoder" 时)
        """
        self.strategy = strategy
        self.llm_client = llm_client
        self._reranker = reranker_model
        self._nlp = None
        self._cross_encoder = None

    @property
    def nlp(self):
        if self._nlp is None:
            import spacy
            try:
                self._nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning(
                    "spaCy model not found — using uniform weights. "
                    "Install: python -m spacy download en_core_web_sm"
                )
                self._nlp = False
        return self._nlp if self._nlp is not False else None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(
        self,
        sentence: str,
        entities: List[str],
        entity_types: Optional[List[str]] = None,
    ) -> List[EntityWeight]:
        """
        为句子中的实体列表分配 attention_weight

        Args:
            sentence: 原始句子
            entities: 实体名称列表
            entity_types: 实体类型列表 (可选)

        Returns:
            EntityWeight 列表，按 attention_weight 降序排列
        """
        if not entities:
            return []

        if entity_types is None:
            entity_types = [""] * len(entities)

        if self.strategy == "cross_encoder":
            weights = self._score_by_cross_encoder(sentence, entities)
        elif self.strategy == "llm" and self.llm_client:
            weights = self._score_by_llm(sentence, entities)
        else:
            weights = self._score_by_syntactic(sentence, entities)

        result = [
            EntityWeight(
                name=name,
                attention_weight=round(w, 4),
                entity_type=etype,
            )
            for name, w, etype in zip(entities, weights, entity_types)
        ]

        # 按权重降序排列
        result.sort(key=lambda e: e.attention_weight, reverse=True)
        return result

    def score_batch(
        self,
        sentences: List[str],
        entities_list: List[List[str]],
    ) -> List[List[EntityWeight]]:
        """批量计算 — 对多句中的实体分配权重"""
        return [
            self.score(sent, ents) for sent, ents in zip(sentences, entities_list)
        ]

    # ------------------------------------------------------------------
    # 方案 A: Cross-Encoder scoring
    # ------------------------------------------------------------------

    def _score_by_cross_encoder(
        self, sentence: str, entities: List[str]
    ) -> List[float]:
        """
        用 Cross-Encoder 对每个 (sentence, entity) pair 打分
        分数 = 实体在句子语义中的核心程度
        """
        if self._cross_encoder is None:
            try:
                from sentence_transformers import CrossEncoder
                self._cross_encoder = CrossEncoder(
                    "cross-encoder/ms-marco-MiniLM-L-6-v2"
                )
            except ImportError:
                logger.warning(
                    "sentence-transformers not installed. "
                    "Falling back to syntactic scoring."
                )
                return self._score_by_syntactic(sentence, entities)

        pairs = [[sentence, entity] for entity in entities]
        scores = self._cross_encoder.predict(pairs)

        # softmax 归一化到 [0, 1]
        import math
        exp_scores = [math.exp(s) for s in scores]
        total = sum(exp_scores)
        if total > 0:
            weights = [s / total for s in exp_scores]
        else:
            weights = [1.0 / len(entities)] * len(entities)

        return weights

    # ------------------------------------------------------------------
    # 方案 B: LLM 联合打分
    # ------------------------------------------------------------------

    _LLM_ATTENTION_PROMPT = (
        "For each entity in the sentence, rate its semantic centrality "
        "to the sentence's core meaning on a scale of 0 to 1.\n\n"
        "Guidelines:\n"
        "- 0.8-1.0: The entity IS what the sentence is about (core subject/topic)\n"
        "- 0.5-0.7: Important supporting entity\n"
        "- 0.2-0.4: Peripheral entity mentioned in passing\n"
        "- 0.0-0.1: Barely relevant entity\n\n"
        'Sentence: "{sentence}"\n\n'
        'Entities: {entities}\n\n'
        'Return JSON: {{"entities": ['
        '{{"name": "Google", "centrality": 0.9}}, ...]}}'
    )

    def _score_by_llm(self, sentence: str, entities: List[str]) -> List[float]:
        import json

        prompt = self._LLM_ATTENTION_PROMPT.format(
            sentence=sentence, entities=json.dumps(entities)
        )
        try:
            response = self.llm_client.generate_response(
                messages=[
                    {"role": "system", "content": "Rate entity centrality."},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
            )
            data = json.loads(response)
            items = data.get("entities", [])
        except Exception as e:
            logger.warning(f"LLM attention scoring failed: {e}")
            return self._score_by_syntactic(sentence, entities)

        # 映射 name -> centrality
        score_map = {}
        for item in items:
            name = item.get("name", "")
            centrality = float(item.get("centrality", 0.5))
            score_map[name] = centrality

        return [score_map.get(name, 0.5) for name in entities]

    # ------------------------------------------------------------------
    # 方案 C: 句法位置启发式 (快速，离线)
    # ------------------------------------------------------------------

    def _score_by_syntactic(
        self, sentence: str, entities: List[str]
    ) -> List[float]:
        """
        基于依存树位置打分，spaCy 不可用时用位置启发式兜底。
        """
        if not entities:
            return []

        nlp = self.nlp
        if nlp is None:
            return self._score_by_position_heuristic(sentence, entities)

        doc = nlp(sentence)
        weights = [0.3] * len(entities)  # 默认低权重

        for i, entity_name in enumerate(entities):
            entity_lower = entity_name.lower()
            entity_tokens = set(entity_lower.split())

            for token in doc:
                token_lower = token.text.lower()
                if token_lower not in entity_tokens:
                    continue

                # 句法位置加权
                if token.dep_ in ("nsubj", "nsubjpass"):
                    weights[i] = max(weights[i], 0.9)
                elif token.dep_ == "dobj":
                    weights[i] = max(weights[i], 0.7)
                elif token.dep_ == "pobj":
                    weights[i] = max(weights[i], 0.4)
                elif token.dep_ in ("iobj", "attr"):
                    weights[i] = max(weights[i], 0.6)
                elif token.pos_ == "PROPN":
                    weights[i] = max(weights[i], 0.5)

        # 归一化
        total = sum(weights)
        if total > 0:
            weights = [w / total for w in weights]

        return weights

    @staticmethod
    def _score_by_position_heuristic(
        sentence: str, entities: List[str]
    ) -> List[float]:
        """
        位置启发式兜底 — 基于实体在句子中的位置估算权重。
        - 句子开头出现 → 高权重 (可能是主语)
        - 句子末尾出现 → 中权重 (可能是宾语)
        - 中间位置 → 默认权重
        """
        import re
        if not entities:
            return []
        sent_lower = sentence.lower()
        sent_len = len(sent_lower)
        weights = []
        for entity in entities:
            ent_lower = entity.lower()
            # 找实体在句子中的首次出现位置
            match = re.search(re.escape(ent_lower), sent_lower)
            if not match:
                weights.append(0.3)
                continue
            pos_ratio = match.start() / max(sent_len, 1)
            if pos_ratio < 0.3:
                weights.append(0.8)  # 前 30%: 可能是主语
            elif pos_ratio > 0.7:
                weights.append(0.6)  # 后 30%: 可能是宾语
            else:
                weights.append(0.4)  # 中间: 默认
        # 归一化
        total = sum(weights)
        if total > 0:
            weights = [w / total for w in weights]
        return weights
