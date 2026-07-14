"""
知识图谱模块 — 实体抽取 → 判别式关系连边 → 图索引 → BFS 检索

架构:
  EntityExtractor        实体抽取 (spaCy/LLM)
  DiscriminativeRelationDetector  LLM 判别式: 给定实体对，判断是否有关联
  EntityAttentionScorer  Attention 权重 (核心 vs 边缘实体)
  GraphStore             图存储 (NetworkX/Neo4j)
  BFSRetriever           带权重的 BFS 图拓扑检索
"""

from mag.graph.extraction import DiscriminativeRelationDetector, EntityExtractor
from mag.graph.attention import EntityAttentionScorer
from mag.graph.store import GraphStore
from mag.graph.bfs_search import BFSRetriever

__all__ = [
    "EntityExtractor",
    "DiscriminativeRelationDetector",
    "EntityAttentionScorer",
    "GraphStore",
    "BFSRetriever",
]
