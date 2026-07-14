"""
评分与时间衰减 — 记忆检索结果的后处理评分

核心功能：
  - 指数时间衰减 (half-life decay)
  - 新鲜度 boost (近期记忆加权)
  - 多信号融合的最终评分计算
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def apply_time_decay(
    memories: List[Dict[str, Any]],
    decay_strategy: str = "exponential",
    half_life_days: float = 30.0,
    max_age_days: float = 365.0,
    freshness_boost_days: float = 1.0,
    freshness_boost: float = 1.2,
) -> List[Dict[str, Any]]:
    """
    对记忆列表应用时间衰减。

    Args:
        memories: 记忆列表，每条含 "score" 和 "created_at" 字段
        decay_strategy: "exponential" | "linear" | "none"
        half_life_days: 指数衰减的半衰期 (天)
        max_age_days: 线性衰减的最大天数
        freshness_boost_days: 新鲜度 boost 的窗口 (天)
        freshness_boost: 新鲜度 boost 的乘数

    Returns:
        按最终分降序排列的记忆列表
    """
    now = datetime.now(timezone.utc)

    for mem in memories:
        # 解析时间戳
        payload = mem.get("payload", {})
        ts_str = payload.get("created_at", "") or mem.get("created_at", "")
        try:
            ts = datetime.fromisoformat(ts_str)
        except (ValueError, TypeError):
            ts = now

        age_days = max(0.0, (now - ts).total_seconds() / 86400.0)

        # 计算衰减因子
        if decay_strategy == "exponential":
            decay = 0.5 ** (age_days / half_life_days)
        elif decay_strategy == "linear":
            decay = max(0.1, 1.0 - age_days / max_age_days)
        else:
            decay = 1.0

        # 新鲜度 boost
        recent = freshness_boost if age_days < freshness_boost_days else 1.0

        # 保存原始分
        mem["raw_score"] = mem.get("score", 0.0)
        mem["time_decay"] = round(decay, 4)
        mem["age_days"] = round(age_days, 1)
        mem["score"] = mem["raw_score"] * decay * recent

    # 按最终分降序
    return sorted(memories, key=lambda x: x["score"], reverse=True)


def compute_entity_match_boost(
    query_entities: List[str],
    memory_entities: List[Dict[str, Any]],
    base_boost: float = 0.1,
) -> float:
    """
    计算实体匹配增强 — 匹配越多、attention 越高，boost 越大。

    Args:
        query_entities: 查询中的实体名列表
        memory_entities: 记忆中的 EntityWeight 列表
        base_boost: 每个匹配实体的基础 boost

    Returns:
        实体匹配 boost [0, 1]
    """
    if not query_entities or not memory_entities:
        return 0.0

    query_set = {e.lower().strip() for e in query_entities}
    total_boost = 0.0

    for e in memory_entities:
        if isinstance(e, dict):
            name = e.get("name", "").lower().strip()
            weight = e.get("attention_weight", 0.5)
        else:
            name = str(e).lower().strip()
            weight = 0.5

        if name in query_set:
            total_boost += base_boost * weight

    return min(1.0, total_boost)


def fuse_multi_route_scores(
    vector_score: float = 0.0,
    bm25_score: float = 0.0,
    bfs_score: float = 0.0,
    entity_boost: float = 0.0,
    time_decay: float = 1.0,
) -> float:
    """
    多路信号融合 — 加性评分归一化。

    max_possible 自适应:
      仅 vector: 1.0
      vector + bm25: 2.0
      vector + bm25 + bfs: 2.5
      vector + bm25 + bfs + entity: 3.0
    """
    active_signals = 1.0  # vector always counts

    if bm25_score > 0:
        active_signals += 1.0
    if bfs_score > 0:
        active_signals += 0.5
    if entity_boost > 0:
        active_signals += 0.5

    raw_combined = vector_score + bm25_score + bfs_score + entity_boost
    combined = min(raw_combined / active_signals, 1.0)

    return combined * time_decay
