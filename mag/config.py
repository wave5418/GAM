"""
MAG 统一配置 — 单一配置源，不再分散在 .env / MAGClient / core.py 三处。

使用方式:
    from mag.config import MAGConfig
    cfg = MAGConfig.from_env()          # 从环境变量加载
    cfg = MAGConfig.from_env_file()     # 从 mag/.env.mag 文件加载

    # 传给 MAGMemory:
    memory = MAGMemory(
        mem0_config=cfg.to_mem0_config(project_name="test"),
        mag_enabled=cfg.mag_enabled,
        ...
    )
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class MAGConfig:
    """MAG 全部可配置项，带默认值与说明。"""

    # ═══════════════════════════════════════════════════════════════
    # LLM — 关系检测 / LinearRAG query 改写
    # ═══════════════════════════════════════════════════════════════
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"
    llm_api_key: str = ""
    llm_base_url: str = "https://api.openai.com/v1"
    llm_temperature: float = 0.0

    # ═══════════════════════════════════════════════════════════════
    # Embedding
    # ═══════════════════════════════════════════════════════════════
    embed_provider: str = "fastembed"       # "fastembed" | "openai"
    embed_model: str = "BAAI/bge-small-en-v1.5"
    embed_dims: int = 384
    embed_api_key: str = ""
    embed_base_url: str = ""

    # ═══════════════════════════════════════════════════════════════
    # Qdrant 向量库 — 服务模式 (Docker)，设置 path 则切到嵌入式
    # ═══════════════════════════════════════════════════════════════
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_path: str = ""

    # ═══════════════════════════════════════════════════════════════
    # MAG 总开关
    # ═══════════════════════════════════════════════════════════════
    mag_enabled: bool = True

    # ═══════════════════════════════════════════════════════════════
    # 句子切分策略: "nlp" (spaCy) | "llm"
    # ═══════════════════════════════════════════════════════════════
    segmentation_strategy: str = "nlp"

    # ═══════════════════════════════════════════════════════════════
    # 图 — LLM direct triples (S5 session-batch)
    #   relation_batch_size: 每批 LLM 处理的句子数
    #     0 = 禁用连边（图只有独立实体节点，无边）
    #     N = 同 session 积累 N 句后批量 LLM 直抽 triples
    # ═══════════════════════════════════════════════════════════════
    relation_batch_size: int = 20

    # 句子积累阈值: 积累句子直到句子数 >= 此值才触发 LLM 连边
    # 0 = 即时模式（每个 add() 单独 LLM），>0 = 积累模式
    edge_entity_threshold: int = 0

    # ═══════════════════════════════════════════════════════════════
    # 检索 — 各路开关 (全部可独立 ablation)
    # ═══════════════════════════════════════════════════════════════

    # 路1: 向量 + BM25 (S4 基础检索，始终启用)

    # 路2: BFS 图拓扑扩展 — 从 query 实体 BFS 遍历图谱收集候选句
    use_bfs: bool = True

    # 路3: CrossEncoder Rerank — 对候选句用 ms-marco-MiniLM-L-6-v2 精排
    #   公式: score = 0.3 * raw + 0.7 * sigmoid(logit)
    use_rerank: bool = True

    # Entity Boost — 使用 LLM triples 写回的实体 payload 做检索加权
    use_entity_boost: bool = True

    # Entity Match Boost — 简单计数: 句子和 query 共现实体越多分越高
    use_entity_match: bool = False

    # ═══════════════════════════════════════════════════════════════
    # 存储 — 各路开关 (全部可独立 ablation)
    # ═══════════════════════════════════════════════════════════════

    # mem0 entity_store 写入 (MAG 句子也写一份到 mem0 实体索引)
    use_entity_store: bool = True

    # SQLite 历史记录
    use_history: bool = True

    # 内存去重 (同一 hash 的句子只存一次)
    use_dedup: bool = True

    # Evidence Unit 构建 — LLM 合并相邻句并消解明确指代，只存合成后的 unit
    build_evidence_units: bool = True

    # Graphiti-style fact edges — 将抽取出的 atomic facts 作为一等检索对象写入索引
    index_graph_facts: bool = True

    # Graphiti-style entity summaries — 由 fact edges 增量维护可检索实体摘要节点
    index_entity_summaries: bool = True

    # 上下文窗口 — 检索时每个候选句附带前后各一句
    use_context_window: bool = False

    # 建图上下文窗口 — fact/triple 抽取时使用最近 N 个已摄入 unit 做指代消解
    graph_context_window: int = 4

    # BM25 权重 — 降低到 < 1.0 可减少短句噪声
    bm25_weight: float = 1.0

    # ═══════════════════════════════════════════════════════════════
    # LinearRAG 在线补充 (已废弃，保留开关)
    # ═══════════════════════════════════════════════════════════════
    linear_rag_enabled: bool = False
    linear_rag_quality_threshold: float = 0.3
    linear_rag_max_supplement_rounds: int = 2

    # ═══════════════════════════════════════════════════════════════
    # 图持久化
    # ═══════════════════════════════════════════════════════════════
    graph_persist_dir: str = "/tmp"

    # ═══════════════════════════════════════════════════════════════
    # 其他
    # ═══════════════════════════════════════════════════════════════
    history_db_dir: str = "/tmp"

    # ──────────────────────────────────────────────────────────────
    # 工厂方法
    # ──────────────────────────────────────────────────────────────

    @classmethod
    def from_env(cls, env_prefix: str = "MAG_") -> "MAGConfig":
        """从环境变量加载（大写 + MAG_ 前缀）。

        例如 MAG_LLM_MODEL → llm_model, MAG_USE_BFS → use_bfs.
        """
        def _env(key: str, default=None):
            return os.getenv(env_prefix + key.upper(), default)

        def _bool(key: str, default: bool) -> bool:
            v = _env(key)
            return v.lower() in ("true", "1", "yes") if v else default

        def _int(key: str, default: int) -> int:
            v = _env(key)
            return int(v) if v else default

        def _float(key: str, default: float) -> float:
            v = _env(key)
            return float(v) if v else default

        return cls(
            llm_provider=_env("LLM_PROVIDER", "openai"),
            llm_model=_env("LLM_MODEL", "gpt-4o-mini"),
            llm_api_key=_env("LLM_API_KEY", ""),
            llm_base_url=_env("LLM_BASE_URL", "https://api.openai.com/v1"),
            llm_temperature=_float("LLM_TEMPERATURE", 0),
            embed_provider=_env("EMBED_PROVIDER", "fastembed"),
            embed_model=_env("EMBED_MODEL", "BAAI/bge-small-en-v1.5"),
            embed_dims=_int("EMBED_DIMS", 384),
            embed_api_key=_env("EMBED_API_KEY", ""),
            embed_base_url=_env("EMBED_BASE_URL", ""),
            qdrant_host=_env("QDRANT_HOST", "localhost"),
            qdrant_port=_int("QDRANT_PORT", 6333),
            qdrant_path=_env("QDRANT_PATH", ""),
            mag_enabled=_bool("ENABLED", True),
            segmentation_strategy=_env("SEGMENTATION", "nlp"),
            relation_batch_size=_int("RELATION_BATCH", 20),
            edge_entity_threshold=_int("EDGE_ENTITY_THRESHOLD", 0),
            use_bfs=_bool("USE_BFS", True),
            use_rerank=_bool("USE_RERANK", True),
            use_entity_boost=_bool("USE_ENTITY_BOOST", True),
            use_entity_match=_bool("USE_ENTITY_MATCH", False),
            use_entity_store=_bool("USE_ENTITY_STORE", True),
            use_history=_bool("USE_HISTORY", True),
            use_dedup=_bool("USE_DEDUP", True),
            build_evidence_units=_bool("BUILD_EVIDENCE_UNITS", True),
            index_graph_facts=_bool("INDEX_GRAPH_FACTS", True),
            index_entity_summaries=_bool("INDEX_ENTITY_SUMMARIES", True),
            use_context_window=_bool("USE_CONTEXT_WINDOW", False),
            graph_context_window=_int("GRAPH_CONTEXT_WINDOW", 4),
            bm25_weight=_float("BM25_WEIGHT", 1.0),
            linear_rag_enabled=_bool("LINEARRAG", False),
            linear_rag_quality_threshold=_float("QUALITY_THRESHOLD", 0.3),
            linear_rag_max_supplement_rounds=_int("MAX_SUPPLEMENT_ROUNDS", 2),
            graph_persist_dir=_env("GRAPH_PERSIST_DIR", "/tmp"),
            history_db_dir=_env("HISTORY_DB_DIR", "/tmp"),
        )

    @classmethod
    def from_env_file(cls, path: str = None) -> "MAGConfig":
        """加载 .env 文件后从环境变量读取。

        如果 path 为 None，默认找 mag/.env.mag。
        """
        if path is None:
            path = os.path.join(os.path.dirname(__file__), ".env.mag")
        if os.path.exists(path):
            from dotenv import load_dotenv
            load_dotenv(path, override=True)
        return cls.from_env()

    # ──────────────────────────────────────────────────────────────
    # 导出为下游模块需要的 dict 格式
    # ──────────────────────────────────────────────────────────────

    def to_mem0_config(self, project_name: str = "mag_default") -> Dict[str, Any]:
        """生成 mem0 MemoryConfig 所需的配置 dict。"""
        embed_cfg = {
            "model": self.embed_model,
            "embedding_dims": self.embed_dims,
        }
        if self.embed_provider == "openai":
            embed_cfg["api_key"] = self.embed_api_key
            embed_cfg["openai_base_url"] = self.embed_base_url

        vs_config: Dict[str, Any] = {
            "collection_name": f"mag_{project_name}",
            "embedding_model_dims": self.embed_dims,
        }
        if self.qdrant_path:
            vs_config["path"] = self.qdrant_path
        else:
            vs_config["host"] = self.qdrant_host
            vs_config["port"] = self.qdrant_port

        return {
            "vector_store": {
                "provider": "qdrant",
                "config": vs_config,
            },
            "llm": {
                "provider": self.llm_provider,
                "config": {
                    "model": self.llm_model,
                    "temperature": self.llm_temperature,
                    "api_key": self.llm_api_key,
                    "openai_base_url": self.llm_base_url,
                },
            },
            "embedder": {
                "provider": self.embed_provider,
                "config": embed_cfg,
            },
            "history_db_path": f"{self.history_db_dir}/mag_{project_name}.db",
            "version": "v1.1",
        }

    def to_linear_rag_config(self) -> Dict[str, Any]:
        """生成 core.py MAGMemory 所需的 linear_rag_config dict。"""
        return {
            "enabled": self.linear_rag_enabled,
            "relation_batch_size": self.relation_batch_size,
            "edge_entity_threshold": self.edge_entity_threshold,
            "use_bfs": self.use_bfs,
            "use_rerank": self.use_rerank,
            "use_entity_boost": self.use_entity_boost,
            "use_entity_match": self.use_entity_match,
            "use_entity_store": self.use_entity_store,
            "use_history": self.use_history,
            "use_dedup": self.use_dedup,
            "build_evidence_units": self.build_evidence_units,
            "index_graph_facts": self.index_graph_facts,
            "index_entity_summaries": self.index_entity_summaries,
            "use_context_window": self.use_context_window,
            "graph_context_window": self.graph_context_window,
            "bm25_weight": self.bm25_weight,
            "quality_threshold": self.linear_rag_quality_threshold,
            "max_supplement_rounds": self.linear_rag_max_supplement_rounds,
        }

    def to_graph_config(self, project_name: str = "default") -> Dict[str, Any]:
        """生成图存储配置。"""
        return {
            "persist_path": os.path.join(
                self.graph_persist_dir, f"mag_{project_name}_graph.json"
            ),
        }

    @property
    def ablation_flags(self) -> Dict[str, Any]:
        """返回所有 ablation 开关，方便打印/日志。"""
        return {
            "bfs": self.use_bfs,
            "rerank": self.use_rerank,
            "entity_boost": self.use_entity_boost,
            "entity_match": self.use_entity_match,
            "relation_batch": self.relation_batch_size,
            "entity_store": self.use_entity_store,
            "history": self.use_history,
            "dedup": self.use_dedup,
            "evidence_units": self.build_evidence_units,
            "graph_facts": self.index_graph_facts,
            "entity_summaries": self.index_entity_summaries,
            "context_window": self.use_context_window,
            "graph_context_window": self.graph_context_window,
            "bm25_weight": self.bm25_weight,
        }
