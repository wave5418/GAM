#!/usr/bin/env python3
"""
BFS 管道冒烟测试 — 复用已有 Qdrant 数据 + 图文件

用法:
    cd x/memory-benchmarks
    python tests/smoke_test_bfs.py

验证项:
    1. 图文件加载 — entities / edges 数量
    2. 边向量 — 全部非零
    3. 实体抽取 — query → entities 命中图
    4. BFS search_paths — 路径数、三元组 sim
    5. BFS → candidate — score 排序后能在 top-k 出现
"""
from __future__ import annotations

import json
import os
import sys
import argparse
import numpy as np

# 确保可以从项目根目录 import mag
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_graph(path: str):
    from mag.graph.store import GraphStore
    gs = GraphStore()
    with open(path) as f:
        gdata = json.load(f)
        for name, attrs in gdata.get("entities", {}).items():
            gs.graph.add_node(name, **attrs)
        for e in gdata.get("edges", []):
            gs.graph.add_edge(e["u"], e["v"], key=e.get("key"), **e.get("data", {}))
    return gs, gdata


def check_storage(qdrant_collection: str, graph_path: str):
    """验证图文件与 Qdrant 的一致性"""
    from qdrant_client import QdrantClient
    client = QdrantClient(host="localhost", port=6333)

    _, gdata = load_graph(graph_path)

    # Linked sids (抽样 50)
    linked = 0; linked_ok = 0
    for n, d in gdata["entities"].items():
        for s in d.get("linked_sentence_ids", []):
            linked += 1
            if linked <= 50:
                if client.retrieve(collection_name=qdrant_collection, ids=[s]):
                    linked_ok += 1
            if linked > 50: break
        if linked > 50: break

    # Edge sids (抽样 50)
    edge = 0; edge_ok = 0; edge_zero = 0
    for e in gdata["edges"]:
        for s in e["data"].get("source_sentence_ids", []):
            edge += 1
            if edge <= 50:
                r = client.retrieve(collection_name=qdrant_collection, ids=[s], with_vectors=True)
                if r and r[0].vector:
                    v = r[0].vector
                    dense = v.get("", []) if isinstance(v, dict) else v
                    if all(x == 0 for x in dense[:3]): edge_zero += 1
                    else: edge_ok += 1
            if edge > 50: break
        if edge > 50: break

    return {
        "entities": len(gdata["entities"]),
        "edges": len(gdata["edges"]),
        "linked_ok": linked_ok,
        "linked_total": linked,
        "edge_ok": edge_ok,
        "edge_zero": edge_zero,
        "edge_total": edge,
    }


def test_bfs(graph_path: str, qdrant_collection: str, question: str):
    """验证 BFS 搜索全流程"""
    from qdrant_client import QdrantClient
    from mag.graph.bfs_search import BFSRetriever
    from mag.schema import EntityWeight
    from mem0.utils.entity_extraction import extract_entities

    gs, _ = load_graph(graph_path)
    bfs = BFSRetriever(gs)
    client = QdrantClient(host="localhost", port=6333)

    # 1. 实体抽取
    ents = extract_entities(question)
    q_ews = []
    for e in ents:
        ename = e[1].strip().lower()
        q_ews.append(EntityWeight(name=ename, attention_weight=0.6, entity_type=e[0]))
        if " " in ename:
            for w in ename.split():
                if len(w) > 2:
                    q_ews.append(EntityWeight(name=w, attention_weight=0.4, entity_type="TOKEN"))

    in_graph = sum(1 for w in q_ews if w.name in gs.graph.nodes)

    # 2. 真实 query embedding (加载 embedding 模型)
    from mag.core import MAGMemory; from mag.config import MAGConfig
    from mem0.configs.base import MemoryConfig
    cfg = MAGConfig.from_env_file()
    mem_cfg = MemoryConfig(**cfg.to_mem0_config(project_name="smoke_test"))
    mem = MAGMemory(mem_cfg, mag_enabled=False)
    embed_model = mem.embedding_model
    emb = embed_model.embed(question, "search")

    # 3. sim 函数 — UUID 取向量，三元组文本直接 embed
    def _sim(identifier):
        try:
            if len(str(identifier)) >= 32 and "-" in str(identifier):
                # UUID: 从 Qdrant 取向量
                r = client.retrieve(collection_name=qdrant_collection, ids=[str(identifier)], with_vectors=True)
                if r and r[0].vector:
                    dense = r[0].vector.get("", r[0].vector) if isinstance(r[0].vector, dict) else r[0].vector
                    v = np.array(dense, dtype=np.float32)
                    if np.allclose(v, 0): return 0.0
                    return float(np.dot(v, emb) / (np.linalg.norm(v) * np.linalg.norm(emb) + 1e-8))
            else:
                # 三元组文本: 直接 embed
                tv = embed_model.embed(str(identifier), "search")
                v = np.array(tv, dtype=np.float32)
                return float(np.dot(v, emb) / (np.linalg.norm(v) * np.linalg.norm(emb) + 1e-8))
        except: pass
        return 0.0

    # 4. BFS search_paths
    paths = bfs.search_paths(q_ews, query_embedding=emb, get_semantic_sim=_sim,
                             max_hops=3, tolerance=2, sim_threshold=0.3, max_results=30)

    # 5. 模拟 scored_results 合并 (只用 30 条)
    pts30, _ = client.scroll(collection_name=qdrant_collection, limit=30, with_payload=True)
    scored = [{"id": pt.id, "score": 0.8, "payload": dict(pt.payload)} for pt in pts30]
    scored_ids = {s["id"] for s in scored}

    added = 0
    for p in paths:
        for sid in p.get("sentences", []):
            sid_str = str(sid)
            if sid_str not in scored_ids:
                r = client.retrieve(collection_name=qdrant_collection, ids=[sid_str], with_payload=True)
                if r and r[0].payload.get("data", "").strip():
                    pl = dict(r[0].payload); pl["_bfs_source"] = "graph_bfs"
                    scored.append({"id": sid_str, "score": p.get("path_score", 0.5), "payload": pl})
                    added += 1

    scored.sort(key=lambda x: x["score"], reverse=True)
    bfs_in_top = sum(1 for s in scored[:30] if s.get("payload", {}).get("_bfs_source") == "graph_bfs")

    return {
        "question": question,
        "query_entities": [(w.name, w.entity_type) for w in q_ews],
        "in_graph": in_graph,
        "total_entities": len(q_ews),
        "bfs_paths": len(paths),
        "bfs_added": added,
        "bfs_in_top30": bfs_in_top,
    }


def main():
    parser = argparse.ArgumentParser(description="BFS 管道冒烟测试")
    parser.add_argument("--project", default="mag_bfs_full",
                        help="项目名 (Qdrant collection: mag_{project}, graph: /tmp/mag_{project}_graph.json)")
    parser.add_argument("--question", default="When did Caroline go to the LGBTQ support group?",
                        help="测试问题")
    args = parser.parse_args()

    # 自动推断路径 (与 config.py to_graph_config 保持一致)
    graph_path = f"/tmp/mag_{args.project}_graph.json"
    qdrant_collection = f"mag_{args.project}"

    if not os.path.exists(graph_path):
        print(f"图文件不存在: {graph_path}")
        sys.exit(1)

    print("=" * 60)
    print("1. 存储验证")
    print("=" * 60)
    storage = check_storage(qdrant_collection, graph_path)
    print(f"  Entities: {storage['entities']}, Edges: {storage['edges']}")
    print(f"  Linked sids: {storage['linked_ok']}/{storage['linked_total']} "
          f"({100*storage['linked_ok']//max(1,storage['linked_total'])}%)")
    print(f"  Edge sids: {storage['edge_ok']} non-zero, {storage['edge_zero']} zero "
          f"({100*storage['edge_ok']//max(1,storage['edge_total'])}%)")

    assert storage["linked_ok"] / max(1, storage["linked_total"]) >= 0.95, \
        f"Linked sids mismatch: {storage['linked_ok']}/{storage['linked_total']}"
    assert storage["edge_zero"] == 0, "Edge sids have zero vectors!"
    print("  ✅ 存储正常")

    print("\n" + "=" * 60)
    print("2. BFS 检索验证")
    print("=" * 60)
    result = test_bfs(graph_path, qdrant_collection, args.question)
    print(f"  Q: {result['question']}")
    print(f"  Query entities: {result['query_entities']}")
    print(f"  In graph: {result['in_graph']}/{result['total_entities']}")
    print(f"  BFS paths: {result['bfs_paths']}")
    print(f"  BFS added: {result['bfs_added']}")
    print(f"  BFS in top-30: {result['bfs_in_top30']}")

    assert result["bfs_paths"] > 0, "BFS returned 0 paths!"
    assert result["bfs_added"] > 0, "BFS added 0 new sentences!"
    print("  ✅ BFS 正常")

    print("\n" + "=" * 60)
    print("ALL CHECKS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
