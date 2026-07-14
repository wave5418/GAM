"""
MAG Memory Schema — 句子粒度的记忆数据结构

层级关系：
  Conversation → Sentences (最小记忆单元) → Triples (索引层)
                                  ↑
                          Entity Attention Weights
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class EntityWeight:
    """带 Attention 权重的实体 — 权重越高越核心"""

    name: str
    attention_weight: float  # [0, 1], 1.0 = 句子语义的核心实体
    entity_type: str = ""    # PERSON / ORG / GPE / DATE / PRODUCT / ...

    def __post_init__(self):
        if not (0.0 <= self.attention_weight <= 1.0):
            raise ValueError(
                f"attention_weight must be in [0, 1], got {self.attention_weight}"
            )

    @property
    def is_core(self) -> bool:
        """核心实体：attention_weight >= 0.5"""
        return self.attention_weight >= 0.5

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "attention_weight": self.attention_weight,
            "entity_type": self.entity_type,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EntityWeight":
        return cls(
            name=d["name"],
            attention_weight=d["attention_weight"],
            entity_type=d.get("entity_type", ""),
        )


@dataclass
class Triple:
    """SPO 三元组 — 从句子中抽取的知识片段"""

    head: str  # 主语实体
    relation: str  # 谓语/关系
    tail: str  # 宾语实体
    confidence: float = 1.0  # 抽取置信度 [0, 1]
    source_sentence_id: str = ""  # ★ 反向引用到原始句子

    def to_dict(self) -> Dict[str, Any]:
        return {
            "head": self.head,
            "relation": self.relation,
            "tail": self.tail,
            "confidence": self.confidence,
            "source_sentence_id": self.source_sentence_id,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Triple":
        return cls(
            head=d["head"],
            relation=d["relation"],
            tail=d["tail"],
            confidence=d.get("confidence", 1.0),
            source_sentence_id=d.get("source_sentence_id", ""),
        )


@dataclass
class SentenceMemory:
    """记忆最小单元 — 一个完整句子"""

    id: str  # UUID
    text: str  # 原始句子文本
    timestamp: datetime  # 创建时间
    entities: List[EntityWeight] = field(default_factory=list)
    speaker: str = ""  # user / assistant / system
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> Dict[str, Any]:
        """转为 Qdrant payload 格式"""
        return {
            "data": self.text,
            "entities": [e.to_dict() for e in self.entities],
            "speaker": self.speaker,
            "created_at": self.timestamp.isoformat(),
            "updated_at": self.timestamp.isoformat(),
            **self.metadata,
        }

    @classmethod
    def from_payload(cls, point_id: str, payload: Dict[str, Any]) -> "SentenceMemory":
        entities_raw = payload.get("entities", [])
        entities = [EntityWeight.from_dict(e) for e in entities_raw]
        ts_str = payload.get("created_at", "")
        try:
            ts = datetime.fromisoformat(ts_str)
        except (ValueError, TypeError):
            ts = datetime.now(timezone.utc)
        return cls(
            id=point_id,
            text=payload.get("data", ""),
            timestamp=ts,
            entities=entities,
            speaker=payload.get("speaker", ""),
            metadata={
                k: v
                for k, v in payload.items()
                if k not in ("data", "entities", "speaker", "created_at", "updated_at")
            },
        )

    @property
    def core_entities(self) -> List[EntityWeight]:
        """仅返回核心实体 (weight >= 0.5)"""
        return [e for e in self.entities if e.is_core]

    @property
    def age_days(self) -> float:
        """记忆存续天数"""
        delta = datetime.now(timezone.utc) - self.timestamp
        return max(0.0, delta.total_seconds() / 86400.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "timestamp": self.timestamp.isoformat(),
            "entities": [e.to_dict() for e in self.entities],
            "speaker": self.speaker,
            "metadata": self.metadata,
        }
