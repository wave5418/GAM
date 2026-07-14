"""
图拓扑 BFS 检索 — 带 Attention 权重的广度优先搜索

从查询实体出发，在知识图谱上进行加权 BFS，收集间接相关的 sentence_id。

权重衰减:
  - 初始权重 = query entity 的 attention_weight
  - 每跳衰减: decay = 0.5^hop
  - 路径分数 = entity_attention_weight * decay
  - 多条路径命中同一 sentence → 取最高分

这是多路混合检索的第三路，解决向量检索无法处理的“多跳推理”问题。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

from mag.schema import EntityWeight

logger = logging.getLogger(__name__)


class BFSRetriever:
    """带 attention 权重的 BFS 图拓扑检索器"""

    def __init__(
        self,
        graph_store,
        max_hops: int = 2,
        decay_factor: float = 0.5,
    ):
        """
        Args:
            graph_store: GraphStore 实例
            max_hops: BFS 最大跳数
            decay_factor: 权重衰减因子 (每跳乘一次)
        """
        self.graph_store = graph_store
        self.max_hops = max_hops
        self.decay_factor = decay_factor

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self,
        query_entities: List[EntityWeight],
        max_hops: Optional[int] = None,
        max_results: int = 30,
        session_scope: Optional[str] = None,
    ) -> List[Tuple[str, float]]:
        """
        带权重的 BFS 检索 — 从 query entities 出发图遍历

        Args:
            query_entities: 查询中抽取的实体 (带 attention_weight)
            max_hops: 最大跳数 (None 则使用默认值)
            max_results: 最大返回的 sentence 数量

        Returns:
            [(sentence_id, bfs_score), ...] 按 bfs_score 降序
        """
        if not query_entities:
            return []

        if max_hops is None:
            max_hops = self.max_hops

        # sentence_id → 最高 bfs_score
        sentence_scores: Dict[str, float] = {}

        for q_entity in query_entities:
            entity_name = q_entity.name
            entity_weight = q_entity.attention_weight

            # BFS 从该实体出发
            results = self._bfs_from_entity(
                start_entity=entity_name,
                start_weight=entity_weight,
                max_hops=max_hops,
                session_scope=session_scope,
            )

            # 合并分数 (取最高)
            for sid, score in results:
                if sid in sentence_scores:
                    sentence_scores[sid] = max(sentence_scores[sid], score)
                else:
                    sentence_scores[sid] = score

        # 排序 + 截断
        sorted_results = sorted(
            sentence_scores.items(), key=lambda x: x[1], reverse=True
        )
        return sorted_results[:max_results]

    def get_indirect_memories(
        self,
        query_entities: List[EntityWeight],
        max_results: int = 30,
        session_scope: Optional[str] = None,
    ) -> List[str]:
        """
        便捷方法：仅返回 sentence_id 列表 (不含分数)

        用于 LinearRAG 实体扩展策略
        """
        scored = self.search(query_entities, max_results=max_results, session_scope=session_scope)
        return [sid for sid, _ in scored]

    def search_by_names(
        self,
        entity_names: List[str],
        uniform_weight: float = 0.5,
        max_hops: int = 2,
        max_results: int = 30,
        session_scope: Optional[str] = None,
    ) -> List[Tuple[str, float]]:
        """
        通过实体名列表搜索 (均匀权重快捷方式)
        """
        entities = [
            EntityWeight(name=name, attention_weight=uniform_weight)
            for name in entity_names
        ]
        return self.search(
            entities,
            max_hops=max_hops,
            max_results=max_results,
            session_scope=session_scope,
        )

    # ------------------------------------------------------------------
    # 路径感知 BFS (tolerance + 语义相似度)
    # ------------------------------------------------------------------

    def search_paths(
        self,
        query_entities: List[EntityWeight],
        query_embedding,
        get_semantic_sim,
        max_hops: int = 3,
        tolerance: int = 2,
        sim_threshold: float = 0.3,
        max_results: int = 20,
        session_scope: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        路径感知 BFS — tolerance=2 + 语义相似度过滤。

        从 query entities 出发，在图谱上遍历，每一步计算当前实体关联
        的句子与 query 的语义相似度。容忍最多 tolerance 步低相似度，
        当相似度回升时，整条路径作为一个候选。

        Args:
            query_entities: 查询中的实体（带 attention_weight）
            query_embedding: 查询的 embedding 向量
            get_semantic_sim: (sentence_id) → float 语义相似度
            max_hops: 最大跳数
            tolerance: 容忍的低相似度步数
            sim_threshold: 相似度阈值
            max_results: 最大路径数

        Returns:
            [{
                "path": [entity1, entity2, ...],      # 实体路径
                "sentences": [sid1, sid2, ...],        # 路径上的句子
                "path_score": float,                    # 路径综合分数
                "step_sims": [0.9, 0.3, 0.8],          # 每步语义相似度
                "graph_weight": float,                  # 图结构权重
            }, ...]
        """
        if not query_entities:
            return []

        all_paths: List[Dict[str, Any]] = []

        for q_entity in query_entities:
            # 尝试精确匹配 或 分词模糊匹配
            entity_name = q_entity.name
            if entity_name not in self.graph_store.graph.nodes:
                # 拆词分别尝试: "Caroline research" → ["Caroline", "research"]
                words = entity_name.split()
                found = [w for w in words if w in self.graph_store.graph.nodes]
                if found:
                    for w in found:
                        paths = self._bfs_paths_from_entity(
                            start_entity=w,
                            start_weight=q_entity.attention_weight * 0.8,
                            query_embedding=query_embedding,
                        get_semantic_sim=get_semantic_sim,
                        max_hops=max_hops,
                        tolerance=tolerance,
                        sim_threshold=sim_threshold,
                        session_scope=session_scope,
                    )
                        all_paths.extend(paths)
                    continue
            paths = self._bfs_paths_from_entity(
                start_entity=entity_name,
                start_weight=q_entity.attention_weight,
                query_embedding=query_embedding,
                get_semantic_sim=get_semantic_sim,
                max_hops=max_hops,
                tolerance=tolerance,
                sim_threshold=sim_threshold,
                session_scope=session_scope,
            )
            all_paths.extend(paths)

        # 按路径分数排序
        all_paths.sort(key=lambda p: p["path_score"], reverse=True)
        return all_paths[:max_results]

    def _bfs_paths_from_entity(
        self,
        start_entity: str,
        start_weight: float,
        query_embedding,
        get_semantic_sim,
        max_hops: int = 3,
        tolerance: int = 2,
        sim_threshold: float = 0.3,
        session_scope: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        从单个实体出发的路径感知 BFS。

        BFS 状态: (entity, path_entities, path_sentences,
                    path_weight, hop, tolerance_used, step_sims)
        """
        g = self.graph_store.graph
        paths: List[Dict[str, Any]] = []

        # Hop 0: 起始实体的直连句子
        direct_sids = self.graph_store.get_sentences_for_entity(
            start_entity,
            session_scope=session_scope,
        )
        direct_sims = [get_semantic_sim(sid) for sid in direct_sids]

        for sid, sim in zip(direct_sids, direct_sims):
            score = start_weight * sim
            if sim >= sim_threshold:
                # 直达路径（1-hop）
                paths.append({
                    "path": [start_entity],
                    "sentences": [sid],
                    "path_score": score,
                    "step_sims": [sim],
                    "graph_weight": start_weight,
                    "triples": [],  # hop-0 无跨越边
                })

        # BFS: (entity, path_entities, path_sids, weight, hop, tol_used, step_sims)
        visited: Set[str] = {start_entity}
        frontier: List[Tuple[str, List[str], List[str], float, int, int, List[float]]] = [
            (start_entity, [start_entity], [], start_weight, 0, 0, [])
        ]

        for hop in range(1, max_hops + 1):
            next_frontier = []

            for cur, path_ents, path_sids, weight, _, tol_used, step_sims in frontier:
                neighbors = self.graph_store.get_neighbors_scoped(
                    cur,
                    max_hops=1,
                    session_scope=session_scope,
                )

                for neighbor in neighbors:
                    if neighbor in visited:
                        continue
                    visited.add(neighbor)

                    hop_weight = weight * (self.decay_factor ** hop)

                    # 从边(cur→neighbor)取三元组文本 + source_sentence_ids
                    edge_sids = []
                    edge_triples = []  # [(cur, relation, neighbor), ...]
                    triple_texts = []  # "head relation tail" 用于语义相似度
                    for k, edge_data in g[cur].get(neighbor, {}).items():
                        if not self.graph_store._edge_matches_scope(edge_data, session_scope):
                            continue
                        sids = edge_data.get("source_sentence_ids", [])
                        scopes = edge_data.get("source_sentence_scopes", {})
                        scoped_sids = [
                            sid for sid in sids
                            if not session_scope or scopes.get(sid, edge_data.get("session_scope", "")) == session_scope
                        ]
                        edge_sids.extend(scoped_sids)
                        rel = edge_data.get("type", "related")
                        if scoped_sids:
                            edge_triples.append((cur, rel, neighbor))
                            triple_texts.append(f"{cur} {rel} {neighbor}")

                    # 用三元组文本算相似度（比句子向量更直接反映关系语义）
                    neighbor_sims = [(sid, get_semantic_sim(txt))
                                     for sid, txt in zip(edge_sids, triple_texts)] if edge_sids else []

                    if not neighbor_sims:
                        continue

                    best_sim = max(s for _, s in neighbor_sims)
                    best_sid = max(neighbor_sims, key=lambda x: x[1])[0]

                    new_tol = tol_used
                    if best_sim < sim_threshold:
                        new_tol += 1
                    else:
                        new_tol = 0  # 相似度恢复，重置 tolerance

                    if new_tol > tolerance:
                        continue  # 超过容忍度，剪枝

                    new_path_ents = path_ents + [neighbor]
                    new_path_sids = path_sids + [best_sid]
                    new_step_sims = step_sims + [best_sim]

                    # 实体 boost
                    entity_info = self.graph_store.get_entity(neighbor)
                    entity_boost = 1.0
                    if entity_info:
                        aw_sum = entity_info["attention_weight_sum"]
                        entity_boost = min(1.0, aw_sum / max(1.0, aw_sum + 1.0))

                    # 路径分数 = 路径中最高语义相似度句子的分数
                    path_score = max(new_step_sims) if new_step_sims else 0.5

                    paths.append({
                        "path": new_path_ents,
                        "sentences": new_path_sids,
                        "path_score": path_score,
                        "triples": edge_triples,  # 遍历边时提取的三元组
                    })

                    next_frontier.append(
                        (neighbor, new_path_ents, new_path_sids,
                         hop_weight, hop, new_tol, new_step_sims)
                    )

            frontier = next_frontier
            if not frontier:
                break

        return paths

    # ------------------------------------------------------------------
    # BFS 核心算法 (旧的单句检索，保留兼容)
    # ------------------------------------------------------------------

    def _bfs_from_entity(
        self,
        start_entity: str,
        start_weight: float,
        max_hops: int,
        session_scope: Optional[str] = None,
    ) -> List[Tuple[str, float]]:
        """
        从单个实体出发的加权 BFS

        BFS 状态: (current_entity, accumulated_weight, hop_count)

        路径分数计算:
          每跳: accumulated_weight *= decay_factor
          entity 权重 = start_weight * node.attention_weight_sum 归一化
          sentence_score = accumulated_weight

        Returns:
            [(sentence_id, bfs_score), ...] — 不包含起始实体的直连 sentence
        """
        g = self.graph_store.graph
        results: Dict[str, float] = {}

        # 收集起始实体的直连 sentence (hop 0)
        direct_sentences = set(
            self.graph_store.get_sentences_for_entity(
                start_entity,
                session_scope=session_scope,
            )
        )
        for sid in direct_sentences:
            results[sid] = start_weight  # hop 0: 完整权重

        if max_hops < 1:
            return list(results.items())

        # BFS 队列: (entity_name, path_weight, hop)
        visited: Set[str] = {start_entity}
        frontier: List[Tuple[str, float, int]] = [(start_entity, start_weight, 0)]

        for hop in range(1, max_hops + 1):
            next_frontier: List[Tuple[str, float, int]] = []

            for current_entity, path_weight, _ in frontier:
                # 获取邻居
                neighbors = self.graph_store.get_neighbors(
                    current_entity,
                    max_hops=1,
                    session_scope=session_scope,
                )

                for neighbor in neighbors:
                    if neighbor in visited:
                        continue
                    visited.add(neighbor)

                    # 衰减权重
                    hop_weight = path_weight * (self.decay_factor ** hop)

                    # 获取该邻居的 attention_weight_sum (归一化因子)
                    entity_info = self.graph_store.get_entity(neighbor)
                    entity_boost = 1.0
                    if entity_info:
                        aw_sum = entity_info["attention_weight_sum"]
                        # sigmoid 归一化: 避免高 weight 实体完全淹没低 weight
                        entity_boost = min(1.0, aw_sum / max(1.0, aw_sum + 1.0))

                    # 收集邻居实体的关联 sentence
                    neighbor_sentences = self.graph_store.get_sentences_for_entity(
                        neighbor,
                        session_scope=session_scope,
                    )
                    for sid in neighbor_sentences:
                        # 排除 hop 0 已收集的直连 sentence
                        if sid in direct_sentences:
                            continue
                        score = hop_weight * entity_boost
                        if sid in results:
                            results[sid] = max(results[sid], score)
                        else:
                            results[sid] = score

                    # 继续扩散
                    next_frontier.append((neighbor, hop_weight, hop))

            frontier = next_frontier
            if not frontier:
                break

        return list(results.items())

    # ------------------------------------------------------------------
    # 实体扩展辅助方法 (供 LinearRAG 使用)
    # ------------------------------------------------------------------

    def expand_entities(
        self,
        seed_entities: List[str],
        max_hops: int = 1,
        min_neighbor_weight: float = 0.3,
        session_scope: Optional[str] = None,
    ) -> List[str]:
        """
        扩展实体集 — 找到种子的邻居实体 (用于 LinearRAG 补充检索)

        Args:
            seed_entities: 种子实体名列表
            max_hops: 扩展跳数
            min_neighbor_weight: 邻居实体最小 attention_weight_sum

        Returns:
            扩展后的实体名列表 (不含种子)
        """
        all_neighbors: Set[str] = set()
        for seed in seed_entities:
            neighbors = self.graph_store.get_neighbors_scoped(
                seed,
                max_hops=max_hops,
                session_scope=session_scope,
            )
            for neighbor in neighbors:
                info = self.graph_store.get_entity(neighbor)
                if info and info["attention_weight_sum"] >= min_neighbor_weight:
                    all_neighbors.add(neighbor)

        # 排除就是种子的
        all_neighbors.difference_update(set(seed_entities))
        return list(all_neighbors)
