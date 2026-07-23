"""
知识图谱存储 — Entity Node + Relation Edge + 反向引用

核心设计：
  Entity Node: {name, attention_weight_sum, linked_sentence_ids}
  Relation Edge: {type, source_sentence_ids[]}  ← 反向引用

支持后端：
  - NetworkX: 内存图，适合开发/中小规模
  - Neo4j: 生产级，需独立服务
"""

from __future__ import annotations

import logging
import threading
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# 关系类型常量
REL_TYPE = "type"
REL_ORIGINS = "source_sentence_ids"
REL_SCOPES = "source_sentence_scopes"
NODE_SCOPES = "linked_sentence_scopes"


def _scope_matches(stored_scope: Optional[str], requested_scope: Optional[str]) -> bool:
    """Return whether a graph element belongs to the requested session scope."""
    if not requested_scope:
        return True
    return stored_scope == requested_scope


class GraphStore:
    """知识图谱存储 — Entity↔Entity + Entity↔Sentence 双向索引"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config: {"backend": "networkx" | "neo4j", ...}
        """
        config = config or {}
        self.backend_name = config.get("backend", "networkx")
        self._graph = None
        self._entity_index: Dict[str, Set[str]] = defaultdict(set)
        # entity_name → set of sentence_ids (快速反向索引)
        self._sentence_entities: Dict[str, List[str]] = defaultdict(list)
        # sentence_id → list of entity_names
        self._sentence_triples: Dict[str, List[Tuple[str, str, str]]] = defaultdict(list)
        # sentence_id → [(head, relation, tail), ...]  用于 rerank 时携带三元组上下文
        self._lock = threading.RLock()

    @property
    def graph(self):
        """懒加载图后端"""
        if self._graph is None:
            if self.backend_name == "neo4j":
                self._init_neo4j()
            else:
                self._init_networkx()
        return self._graph

    def _init_networkx(self):
        """初始化 NetworkX 有向图"""
        try:
            import networkx as nx
        except ImportError:
            logger.error("networkx not installed. Install with: pip install networkx")
            raise

        self._graph = nx.MultiDiGraph()
        logger.info("GraphStore initialized with NetworkX backend")

    def _init_neo4j(self):
        """初始化 Neo4j 后端 (placeholder — 按需实现)"""
        try:
            from neo4j import GraphDatabase
        except ImportError:
            logger.error("neo4j not installed. Install with: pip install neo4j")
            raise
        # 留空：连接信息由 config 提供
        self._neo4j_driver = None
        logger.warning("Neo4j backend not fully wired — using NetworkX fallback")
        self._init_networkx()

    # ==================================================================
    # Entity 操作
    # ==================================================================

    def upsert_entity(
        self,
        name: str,
        attention_weight: float = 0.5,
        sentence_id: str = "",
        entity_type: str = "",
        session_scope: Optional[str] = None,
    ) -> str:
        """
        创建或更新实体节点。

        - 新实体：创建节点，初始化所有属性
        - 已有实体：累加 attention_weight_sum，追加 sentence_id

        Args:
            name: 实体名
            attention_weight: 本次关联的 attention 权重
            sentence_id: 来源句子 ID
            entity_type: 实体类型 (PERSON/ORG/GPE/...)

        Returns:
            实体名 (用作唯一标识)
        """
        with self._lock:
            g = self.graph
            node_id = name.strip().lower()  # 归一化: 大小写统一

            if node_id in g.nodes:
                # 累加权重
                old_sum = g.nodes[node_id].get("attention_weight_sum", 0.0)
                g.nodes[node_id]["attention_weight_sum"] = old_sum + attention_weight

                # 追加 sentence_id
                linked = list(g.nodes[node_id].get("linked_sentence_ids", []))
                if sentence_id and sentence_id not in linked:
                    linked.append(sentence_id)
                g.nodes[node_id]["linked_sentence_ids"] = linked

                # 记录 sentence_id 所属 session，供图遍历阶段过滤。
                linked_scopes = dict(g.nodes[node_id].get(NODE_SCOPES, {}))
                if sentence_id:
                    linked_scopes[sentence_id] = session_scope or ""
                g.nodes[node_id][NODE_SCOPES] = linked_scopes

                # 更新类型 (如有)
                if entity_type and not g.nodes[node_id].get("entity_type"):
                    g.nodes[node_id]["entity_type"] = entity_type
            else:
                # 新建节点
                g.add_node(
                    node_id,
                    attention_weight_sum=attention_weight,
                    linked_sentence_ids=[sentence_id] if sentence_id else [],
                    **{NODE_SCOPES: {sentence_id: session_scope or ""} if sentence_id else {}},
                    entity_type=entity_type,
                    created_at=None,  # 可扩展
                )

            # 维护反向索引
            if sentence_id:
                self._entity_index[node_id].add(sentence_id)
                if node_id not in self._sentence_entities[sentence_id]:
                    self._sentence_entities[sentence_id].append(node_id)

            return node_id

    def get_core_entities(
        self, min_weight: float = 0.5, top_k: int = 100
    ) -> List[Dict[str, Any]]:
        """
        获取核心实体集 — attention_weight_sum 最高的前 K 个

        Returns:
            [{"name": "...", "attention_weight_sum": 1.5, "linked_count": 3}, ...]
        """
        with self._lock:
            g = self.graph
            entities = []
            for node_id, node in list(g.nodes(data=True)):
                weight = node.get("attention_weight_sum", 0.0)
                if weight < min_weight:
                    continue
                entities.append({
                    "name": node_id,
                    "attention_weight_sum": weight,
                    "linked_count": len(node.get("linked_sentence_ids", [])),
                    "entity_type": node.get("entity_type", ""),
                })

            entities.sort(key=lambda e: e["attention_weight_sum"], reverse=True)
            return entities[:top_k]

    def get_entity(self, name: str) -> Optional[Dict[str, Any]]:
        """获取单个实体信息"""
        with self._lock:
            g = self.graph
            if name not in g.nodes:
                return None
            node = g.nodes[name]
            return {
                "name": name,
                "attention_weight_sum": node.get("attention_weight_sum", 0.0),
                "linked_sentence_ids": list(node.get("linked_sentence_ids", [])),
                "entity_type": node.get("entity_type", ""),
            }

    def get_neighbors(
        self,
        entity_name: str,
        max_hops: int = 1,
        session_scope: Optional[str] = None,
    ) -> List[str]:
        """获取实体的邻居实体名列表 (BFS max_hops 跳)"""
        return self.get_neighbors_scoped(
            entity_name,
            max_hops=max_hops,
            session_scope=session_scope,
        )

    def get_neighbors_scoped(
        self,
        entity_name: str,
        max_hops: int = 1,
        session_scope: Optional[str] = None,
    ) -> List[str]:
        """获取实体邻居，并在提供 session_scope 时只沿同 scope 的边遍历。"""
        with self._lock:
            g = self.graph
            if entity_name not in g.nodes:
                return []

            if max_hops == 1:
                neighbors = set()
                for _, neighbor, edge_data in list(g.out_edges(entity_name, data=True)):
                    if self._edge_matches_scope(edge_data, session_scope):
                        neighbors.add(neighbor)
                for predecessor, _, edge_data in list(g.in_edges(entity_name, data=True)):
                    if self._edge_matches_scope(edge_data, session_scope):
                        neighbors.add(predecessor)
                return list(neighbors)

            # 多跳 BFS
            visited = {entity_name}
            frontier = {entity_name}
            for _ in range(max_hops):
                next_frontier = set()
                for node in frontier:
                    for _, neighbor, edge_data in list(g.out_edges(node, data=True)):
                        if not self._edge_matches_scope(edge_data, session_scope):
                            continue
                        if neighbor not in visited:
                            visited.add(neighbor)
                            next_frontier.add(neighbor)
                    for predecessor, _, edge_data in list(g.in_edges(node, data=True)):
                        if not self._edge_matches_scope(edge_data, session_scope):
                            continue
                        if predecessor not in visited:
                            visited.add(predecessor)
                            next_frontier.add(predecessor)
                frontier = next_frontier
                if not frontier:
                    break

            visited.discard(entity_name)
            return list(visited)

    # ==================================================================
    # Relation 操作 (含反向引用)
    # ==================================================================

    def add_relation(
        self,
        head: str,
        relation: str,
        tail: str,
        source_sentence_id: str = "",
        confidence: float = 1.0,
        session_scope: Optional[str] = None,
        source_fact_id: str = "",
        source_fact: str = "",
    ):
        """
        添加或更新关系边。

        核心能力：边携带 source_sentence_ids (反向引用)，
        支持从 Entity 追溯到原始 Sentence。

        Args:
            head: 主语实体
            relation: 关系/谓词
            tail: 宾语实体
            source_sentence_id: 产生此关系的句子 ID
            confidence: 抽取置信度
        """
        with self._lock:
            g = self.graph
            head = head.strip().lower()
            tail = tail.strip().lower()
            relation = relation.strip()

            if not head or not tail:
                return

            # 确保两端节点存在 (如果还没通过 upsert_entity 创建)
            if head not in g.nodes:
                g.add_node(head, attention_weight_sum=0.0, linked_sentence_ids=[])
            if tail not in g.nodes:
                g.add_node(tail, attention_weight_sum=0.0, linked_sentence_ids=[])

            # 每条 source_sentence_id 独立建边，不合并
            g.add_edge(
                head,
                tail,
                **{
                    REL_TYPE: relation,
                    REL_ORIGINS: [source_sentence_id] if source_sentence_id else [],
                    REL_SCOPES: {source_sentence_id: session_scope or ""} if source_sentence_id else {},
                    "session_scope": session_scope or "",
                    "confidence": confidence,
                    "source_fact_id": source_fact_id,
                    "source_fact": source_fact,
                },
            )

            # 维护 sentence → triples 索引
            if source_sentence_id:
                self._sentence_triples[source_sentence_id].append((head, relation, tail))

    # ==================================================================
    # Sentence → Triples 查询 (用于 rerank 时携带三元组上下文)
    # ==================================================================

    def get_triples_for_sentence(self, sentence_id: str) -> List[Tuple[str, str, str]]:
        """返回某句子关联的所有三元组 [(head, relation, tail), ...]"""
        with self._lock:
            return list(self._sentence_triples.get(sentence_id, []))

    # ==================================================================
    # 反向引用查询
    # ==================================================================

    def get_sentences_for_entity(
        self,
        entity_name: str,
        session_scope: Optional[str] = None,
    ) -> List[str]:
        """
        查询某实体关联的所有 sentence_id。

        数据来源：Entity Node 上的 linked_sentence_ids (来自 S6 upsert_entity)
        """
        with self._lock:
            g = self.graph
            sentence_ids: Set[str] = set()
            name_lower = entity_name.lower()
            for node, node_data in list(g.nodes(data=True)):
                if node.lower() == name_lower:
                    direct = list(node_data.get("linked_sentence_ids", []))
                    linked_scopes = dict(node_data.get(NODE_SCOPES, {}))
                    for sid in direct:
                        if _scope_matches(linked_scopes.get(sid, ""), session_scope):
                            sentence_ids.add(sid)
                    break
            return list(sentence_ids)

    def get_sentences_for_relation(
        self, head: str, relation: str, tail: str
    ) -> List[str]:
        """查询特定关系边的 source_sentence_ids"""
        with self._lock:
            g = self.graph
            head = head.strip()
            tail = tail.strip()

            if head not in g or tail not in g:
                return []

            for key, edge_data in list(g[head].get(tail, {}).items()):
                if edge_data.get(REL_TYPE) == relation:
                    return list(edge_data.get(REL_ORIGINS, []))

            return []

    def get_all_relations_for_entity(
        self, entity_name: str, session_scope: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """获取实体的所有关系 (含反向引用)"""
        with self._lock:
            g = self.graph
            relations = []

            if entity_name not in g.nodes:
                return relations

            # 出边
            for _, tail, edge_data in list(g.out_edges(entity_name, data=True)):
                if not self._edge_matches_scope(edge_data, session_scope):
                    continue
                relations.append({
                    "head": entity_name,
                    "relation": edge_data.get(REL_TYPE, ""),
                    "tail": tail,
                    "confidence": edge_data.get("confidence", 1.0),
                    "source_sentence_ids": list(edge_data.get(REL_ORIGINS, [])),
                    "source_fact_id": edge_data.get("source_fact_id", ""),
                    "source_fact": edge_data.get("source_fact", ""),
                    "direction": "out",
                })

            # 入边
            for head, _, edge_data in list(g.in_edges(entity_name, data=True)):
                if not self._edge_matches_scope(edge_data, session_scope):
                    continue
                relations.append({
                    "head": head,
                    "relation": edge_data.get(REL_TYPE, ""),
                    "tail": entity_name,
                    "confidence": edge_data.get("confidence", 1.0),
                    "source_sentence_ids": list(edge_data.get(REL_ORIGINS, [])),
                    "source_fact_id": edge_data.get("source_fact_id", ""),
                    "source_fact": edge_data.get("source_fact", ""),
                    "direction": "in",
                })

            return relations

    @staticmethod
    def _edge_matches_scope(edge_data: Dict[str, Any], session_scope: Optional[str]) -> bool:
        """Check whether an edge has at least one origin sentence in scope."""
        if not session_scope:
            return True
        edge_scope = edge_data.get("session_scope")
        if edge_scope == session_scope:
            return True
        scopes = edge_data.get(REL_SCOPES, {})
        if isinstance(scopes, dict):
            return any(scope == session_scope for scope in scopes.values())
        return False

    # ==================================================================
    # 维护操作
    # ==================================================================

    def delete_relations_for_sentence(self, sentence_id: str):
        """
        从所有边的 source_sentence_ids 中移除指定 sentence_id。
        如果某条边的 origins 变为空，删除该边。
        """
        with self._lock:
            g = self.graph
            edges_to_delete = []

            for head, tail, key, edge_data in list(g.edges(keys=True, data=True)):
                origins = list(edge_data.get(REL_ORIGINS, []))
                if sentence_id in origins:
                    origins.remove(sentence_id)
                    if not origins:
                        edges_to_delete.append((head, tail, key))
                    else:
                        g[head][tail][key][REL_ORIGINS] = origins

            for head, tail, key in edges_to_delete:
                g.remove_edge(head, tail, key)

            # 清理 _entity_index
            for entity_name in list(self._entity_index.keys()):
                self._entity_index[entity_name].discard(sentence_id)
                if not self._entity_index[entity_name]:
                    del self._entity_index[entity_name]

            # 清理 _sentence_entities
            if sentence_id in self._sentence_entities:
                del self._sentence_entities[sentence_id]

    def stats(self) -> Dict[str, int]:
        """图统计信息"""
        with self._lock:
            g = self.graph
            return {
                "num_entities": g.number_of_nodes(),
                "num_relations": len(list(g.edges(keys=True))),
                "num_sentence_indexed": len(self._sentence_entities),
            }

    def clear(self):
        """清空图"""
        with self._lock:
            self._graph = None
            self._entity_index.clear()
            self._sentence_entities.clear()
            self._sentence_triples.clear()
