import asyncio
import gc
import hashlib
import json
import logging
import os
import re
import tempfile
import threading
import uuid
import warnings
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import ValidationError

from mem0.configs.base import MemoryConfig, MemoryItem
from mem0.configs.enums import MemoryType
from mem0.configs.prompts import (
    ADDITIVE_EXTRACTION_PROMPT,
    AGENT_CONTEXT_SUFFIX,
    PROCEDURAL_MEMORY_SYSTEM_PROMPT,
    generate_additive_extraction_prompt,
)
from mem0.exceptions import ValidationError as Mem0ValidationError
from mem0.memory.base import MemoryBase
from mem0.memory.setup import mem0_dir, setup_config
from mem0.memory.storage import SQLiteManager
from mem0.memory.telemetry import MEM0_TELEMETRY, capture_event
from mem0.memory.utils import (
    extract_json,
    parse_messages,
    parse_vision_messages,
    process_telemetry_filters,
    remove_code_blocks,
)
from mem0.utils.entity_extraction import extract_entities, extract_entities_batch
from mem0.utils.factory import (
    EmbedderFactory,
    LlmFactory,
    RerankerFactory,
    VectorStoreFactory,
)
from mem0.utils.lemmatization import lemmatize_for_bm25
from mem0.utils.scoring import (
    ENTITY_BOOST_WEIGHT,
    get_bm25_params,
    normalize_bm25,
    score_and_rank,
)

# ── MAG 增强模块 ──
from mag.schema import EntityWeight
from mag.segmentation import SentenceSegmenter
from mag.graph import (
    BFSRetriever,
    DiscriminativeRelationDetector,
    EntityAttentionScorer,
    EntityExtractor,
    GraphStore,
)

_MAG_KEY = "MAG_origin"
_MAG_VAL = "sentence"

# Suppress SWIG deprecation warnings globally
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*SwigPy.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*swigvarlink.*")

# Initialize logger early for util functions
logger = logging.getLogger(__name__)

_MAG_EVIDENCE_STOPWORDS = frozenset({
    "about",
    "after",
    "again",
    "also",
    "and",
    "any",
    "are",
    "before",
    "both",
    "but",
    "can",
    "could",
    "date",
    "did",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "her",
    "him",
    "his",
    "how",
    "into",
    "its",
    "last",
    "many",
    "month",
    "more",
    "most",
    "much",
    "next",
    "not",
    "own",
    "said",
    "she",
    "that",
    "the",
    "their",
    "them",
    "then",
    "there",
    "they",
    "this",
    "time",
    "was",
    "week",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "will",
    "with",
    "would",
    "year",
})

_MAG_TEMPORAL_TERMS = frozenset({
    "after",
    "before",
    "date",
    "day",
    "friday",
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
    "last",
    "monday",
    "month",
    "next",
    "saturday",
    "sunday",
    "thursday",
    "time",
    "today",
    "tomorrow",
    "tuesday",
    "wednesday",
    "week",
    "when",
    "year",
    "yesterday",
})

_MAG_DATE_RE = re.compile(
    r"\b(?:20\d{2}|jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?|"
    r"mon(?:day)?|tue(?:sday)?|wed(?:nesday)?|thu(?:rsday)?|fri(?:day)?|"
    r"sat(?:urday)?|sun(?:day)?|yesterday|tomorrow|last|next)\b",
    re.IGNORECASE,
)


# Fields that hold runtime auth/connection objects and must be preserved.
# These are non-serializable objects (e.g. AWSV4SignerAuth, RequestsHttpConnection)
# needed by clients like OpenSearch — not sensitive strings to redact.
_RUNTIME_FIELDS = frozenset({
    "http_auth",
    "auth",
    "connection_class",
    "ssl_context",
})

# Fields that are known to contain sensitive secrets and must be redacted.
_SENSITIVE_FIELDS_EXACT = frozenset({
    "api_key",
    "secret_key",
    "private_key",
    "access_key",
    "password",
    "credentials",
    "credential",
    "secret",
    "token",
    "access_token",
    "refresh_token",
    "auth_token",
    "session_token",
    "client_secret",
    "auth_client_secret",
    "azure_client_secret",
    "service_account_json",
    "aws_session_token",
})

# Suffixes that indicate a field likely holds a secret value.
_SENSITIVE_SUFFIXES = (
    "_password",
    "_secret",
    "_token",
    "_credential",
    "_credentials",
)

# Entity parameters that must be passed via filters, not top-level kwargs
ENTITY_PARAMS = frozenset({"user_id", "agent_id", "run_id"})


def _reject_top_level_entity_params(kwargs: Dict[str, Any], method_name: str) -> None:
    """Reject top-level entity parameters - must use filters instead."""
    invalid_keys = ENTITY_PARAMS & set(kwargs.keys())
    if invalid_keys:
        raise ValueError(
            f"Top-level entity parameters {invalid_keys} are not supported in {method_name}(). "
            f"Use filters={{'user_id': '...'}} instead."
        )


def _validate_and_trim_entity_id(value: Optional[str], name: str) -> Optional[str]:
    """
    Validates and normalizes an entity ID.
    - Trims leading/trailing whitespace
    - Rejects empty or whitespace-only strings
    - Rejects strings containing internal whitespace

    Args:
        value: The entity ID value to validate
        name: The parameter name (for error messages)

    Returns:
        The trimmed entity ID, or None if input is None

    Raises:
        ValueError: If entity ID is invalid
    """
    if value is None:
        return None
    trimmed = value.strip()
    if trimmed == "":
        raise ValueError(
            f"Invalid {name}: cannot be empty or whitespace-only. Provide a valid identifier."
        )
    if any(c.isspace() for c in trimmed):
        raise ValueError(
            f"Invalid {name}: cannot contain whitespace. Provide a valid identifier without spaces."
        )
    return trimmed


def _validate_search_params(threshold: Optional[float] = None, top_k: Optional[int] = None) -> None:
    """
    Validates search parameters.

    Args:
        threshold: Similarity threshold (must be between 0 and 1)
        top_k: Number of results to return (must be non-negative integer)

    Raises:
        ValueError: If threshold or top_k are invalid
    """
    if threshold is not None:
        if not isinstance(threshold, (int, float)):
            raise ValueError("threshold must be a valid number")
        if threshold < 0 or threshold > 1:
            raise ValueError(
                f"Invalid threshold: {threshold}. Must be between 0 and 1 (inclusive)."
            )
    if top_k is not None:
        if not isinstance(top_k, int) or isinstance(top_k, bool):
            raise ValueError("top_k must be a valid integer")
        if top_k < 0:
            raise ValueError(
                f"Invalid top_k: {top_k}. Must be a non-negative integer."
            )


def _is_sensitive_field(field_name: str) -> bool:
    """Check if a field should be redacted for telemetry safety.

    Uses a layered approach:
    1. Runtime fields (allowlist) — always preserved, highest priority.
    2. Exact deny list — known secret field names.
    3. Suffix deny list — catches patterns like db_password, auth_secret, etc.
    """
    name = field_name.lower().strip()
    if name in _RUNTIME_FIELDS:
        return False
    if name in _SENSITIVE_FIELDS_EXACT:
        return True
    return any(name.endswith(suffix) for suffix in _SENSITIVE_SUFFIXES)


def _safe_deepcopy_config(config):
    """Safely deepcopy config, falling back to dict-based cloning for non-serializable objects."""
    try:
        return deepcopy(config)
    except Exception as e:
        logger.debug(f"Deepcopy failed, using dict-based cloning: {e}")

        config_class = type(config)

        if hasattr(config, "model_dump"):
            try:
                clone_dict = config.model_dump()
            except Exception:
                clone_dict = dict(config.__dict__)
        else:
            clone_dict = dict(config.__dict__)

        # Restore runtime fields, redact sensitive ones
        for field_name in list(clone_dict.keys()):
            if field_name in _RUNTIME_FIELDS and hasattr(config, field_name):
                clone_dict[field_name] = getattr(config, field_name)
            elif _is_sensitive_field(field_name):
                clone_dict[field_name] = None

        try:
            return config_class(**clone_dict)
        except Exception:
            logger.debug("Config reconstruction failed, returning shallow dict clone")
            return type("Config", (), clone_dict)()


def _normalize_iso_timestamp_to_utc(timestamp: Optional[str]) -> Optional[str]:
    """Normalize timezone-aware ISO timestamps to UTC without rewriting naive values."""
    if not timestamp:
        return timestamp
    try:
        parsed = datetime.fromisoformat(timestamp)
    except ValueError:
        return timestamp
    if parsed.tzinfo is None:
        return timestamp
    return parsed.astimezone(timezone.utc).isoformat()


def _mag_text_tokens(text: Any) -> Set[str]:
    """Return content-bearing lowercase tokens for lightweight evidence checks."""
    tokens = re.findall(r"[a-z0-9]+", str(text).lower())
    return {
        token
        for token in tokens
        if len(token) > 2 and token not in _MAG_EVIDENCE_STOPWORDS
    }


def _mag_jaccard(left: Set[str], right: Set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / max(1, len(left | right))


def _mag_add_source(source: str, route: str) -> str:
    """Append a route label once while preserving route order."""
    parts = [part for part in str(source or "").split("+") if part]
    if route not in parts:
        parts.append(route)
    return "+".join(parts)


def _mag_candidate_evidence_features(query: str, candidate: Dict[str, Any]) -> Dict[str, float]:
    """Compute small, inspectable retrieval-time evidence features."""
    memory = candidate.get("memory", "")
    q_tokens = _mag_text_tokens(query)
    m_tokens = _mag_text_tokens(memory)
    coverage = len(q_tokens & m_tokens) / max(1, len(q_tokens))

    query_terms = set(re.findall(r"[a-z0-9]+", str(query).lower()))
    query_is_temporal = bool(_MAG_TEMPORAL_TERMS & query_terms)
    memory_has_date = bool(_MAG_DATE_RE.search(memory) or _MAG_DATE_RE.search(str(candidate.get("created_at", ""))))
    date_boost = 1.0 if query_is_temporal and memory_has_date else 0.0

    # Long path contexts can be useful, but very broad snippets often bury the
    # atomic answer and have caused distractor wins in LOCOMO.
    text_len = len(str(memory))
    length_penalty = 0.0
    if text_len > 1800:
        length_penalty = min(0.08, (text_len - 1800) / 10000)

    evidence_score = min(1.0, (0.75 * coverage) + (0.25 * date_boost))
    return {
        "query_coverage": round(coverage, 4),
        "temporal_cue": date_boost,
        "length_penalty": round(length_penalty, 4),
        "evidence_score": round(evidence_score, 4),
    }


def _mag_diverse_topk(candidates: List[Dict[str, Any]], limit: int, pool_size: int) -> List[Dict[str, Any]]:
    """Select high-scoring candidates while reducing near-duplicate evidence."""
    if limit <= 0 or len(candidates) <= limit:
        return candidates[:limit]

    pool = candidates[:pool_size]
    selected: List[Dict[str, Any]] = []
    selected_tokens: List[Set[str]] = []
    max_score = max((c.get("score", 0.0) for c in pool), default=1.0) or 1.0

    while pool and len(selected) < limit:
        best_idx = 0
        best_score = float("-inf")
        for idx, candidate in enumerate(pool):
            tokens = _mag_text_tokens(candidate.get("memory", ""))
            similarity = max((_mag_jaccard(tokens, seen) for seen in selected_tokens), default=0.0)
            normalized_score = candidate.get("score", 0.0) / max_score
            mmr_score = (0.82 * normalized_score) - (0.18 * similarity)
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx
        chosen = pool.pop(best_idx)
        selected.append(chosen)
        selected_tokens.append(_mag_text_tokens(chosen.get("memory", "")))

    if len(selected) < limit:
        selected.extend(candidates[len(selected):limit])
    return selected


def _build_filters_and_metadata(
    *,  # Enforce keyword-only arguments
    user_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    run_id: Optional[str] = None,
    actor_id: Optional[str] = None,  # For query-time filtering
    input_metadata: Optional[Dict[str, Any]] = None,
    input_filters: Optional[Dict[str, Any]] = None,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Constructs metadata for storage and filters for querying based on session and actor identifiers.

    This helper supports multiple session identifiers (`user_id`, `agent_id`, and/or `run_id`)
    for flexible session scoping and optionally narrows queries to a specific `actor_id`. It returns two dicts:

    1. `base_metadata_template`: Used as a template for metadata when storing new memories.
       It includes all provided session identifier(s) and any `input_metadata`.
    2. `effective_query_filters`: Used for querying existing memories. It includes all
       provided session identifier(s), any `input_filters`, and a resolved actor
       identifier for targeted filtering if specified by any actor-related inputs.

    Actor filtering precedence: explicit `actor_id` arg → `filters["actor_id"]`
    This resolved actor ID is used for querying but is not added to `base_metadata_template`,
    as the actor for storage is typically derived from message content at a later stage.

    Args:
        user_id (Optional[str]): User identifier, for session scoping.
        agent_id (Optional[str]): Agent identifier, for session scoping.
        run_id (Optional[str]): Run identifier, for session scoping.
        actor_id (Optional[str]): Explicit actor identifier, used as a potential source for
            actor-specific filtering. See actor resolution precedence in the main description.
        input_metadata (Optional[Dict[str, Any]]): Base dictionary to be augmented with
            session identifiers for the storage metadata template. Defaults to an empty dict.
        input_filters (Optional[Dict[str, Any]]): Base dictionary to be augmented with
            session and actor identifiers for query filters. Defaults to an empty dict.

    Returns:
        tuple[Dict[str, Any], Dict[str, Any]]: A tuple containing:
            - base_metadata_template (Dict[str, Any]): Metadata template for storing memories,
              scoped to the provided session(s).
            - effective_query_filters (Dict[str, Any]): Filters for querying memories,
              scoped to the provided session(s) and potentially a resolved actor.
    """

    base_metadata_template = deepcopy(input_metadata) if input_metadata else {}
    effective_query_filters = deepcopy(input_filters) if input_filters else {}

    # ---------- validate and add all provided session ids ----------
    session_ids_provided = []

    # Validate and trim entity IDs
    user_id = _validate_and_trim_entity_id(user_id, "user_id")
    agent_id = _validate_and_trim_entity_id(agent_id, "agent_id")
    run_id = _validate_and_trim_entity_id(run_id, "run_id")

    if user_id:
        base_metadata_template["user_id"] = user_id
        effective_query_filters["user_id"] = user_id
        session_ids_provided.append("user_id")

    if agent_id:
        base_metadata_template["agent_id"] = agent_id
        effective_query_filters["agent_id"] = agent_id
        session_ids_provided.append("agent_id")

    if run_id:
        base_metadata_template["run_id"] = run_id
        effective_query_filters["run_id"] = run_id
        session_ids_provided.append("run_id")

    if not session_ids_provided:
        raise Mem0ValidationError(
            message="At least one of 'user_id', 'agent_id', or 'run_id' must be provided.",
            error_code="VALIDATION_001",
            details={"provided_ids": {"user_id": user_id, "agent_id": agent_id, "run_id": run_id}},
            suggestion="Please provide at least one identifier to scope the memory operation."
        )

    # ---------- optional actor filter ----------
    resolved_actor_id = actor_id or effective_query_filters.get("actor_id")
    if resolved_actor_id:
        effective_query_filters["actor_id"] = resolved_actor_id

    return base_metadata_template, effective_query_filters


def _build_session_scope(filters):
    """Build deterministic session scope string from entity IDs."""
    parts = []
    for key in sorted(["user_id", "agent_id", "run_id"]):
        val = filters.get(key)
        if val:
            parts.append(f"{key}={val}")
    return "&".join(parts)


setup_config()
logger = logging.getLogger(__name__)


class MAGMemory(MemoryBase):
    """MAG Memory — 基于 mem0 Memory 的直接修改版本。

    在原始 mem0 管道中直接植入:
      1. 句子粒度切分
      2. 知识图谱索引 (先抽实体→LLM判别式连边→source_sentence_id反向引用)
      3. Entity Attention 权重
      4. LinearRAG 在线补充
    """

    def __init__(self, config: MemoryConfig = MemoryConfig(),
                 *, mag_enabled: bool = True,
                 segmentation_strategy: str = "nlp",
                 entity_strategy: str = "spacy",
                 attention_strategy: str = "syntactic",
                 graph_config: Optional[Dict[str, Any]] = None,
                 linear_rag_config: Optional[Dict[str, Any]] = None):
        self.config = config

        self.embedding_model = EmbedderFactory.create(
            self.config.embedder.provider,
            self.config.embedder.config,
            self.config.vector_store.config,
        )
        self.vector_store = VectorStoreFactory.create(
            self.config.vector_store.provider, self.config.vector_store.config
        )
        self.llm = LlmFactory.create(self.config.llm.provider, self.config.llm.config)
        self.db = SQLiteManager(self.config.history_db_path)
        self.collection_name = self.config.vector_store.config.collection_name
        self.api_version = self.config.version
        self.custom_instructions = self.config.custom_instructions

        # Initialize reranker if configured
        self.reranker = None
        if config.reranker:
            self.reranker = RerankerFactory.create(
                config.reranker.provider,
                config.reranker.config
            )

        # Entity store is initialized lazily on first use
        self._entity_store = None

        # ── MAG 增强组件 (全部可独立开关，ablation-friendly) ──
        self.mag_enabled = mag_enabled
        self.mag_use_attention = attention_strategy != "off"
        self.mag_relation_batch_size = linear_rag_config.get("relation_batch_size", 15) if linear_rag_config else 15
        self.mag_use_linear_rag = linear_rag_config is not None and linear_rag_config.get("enabled", True)
        self.mag_use_bfs = linear_rag_config is not None and linear_rag_config.get("use_bfs", True) and self.mag_relation_batch_size > 0
        self.mag_use_rerank = linear_rag_config is not None and linear_rag_config.get("use_rerank", True) if linear_rag_config else True
        self.mag_use_entity_store = linear_rag_config.get("use_entity_store", True) if linear_rag_config else True
        self.mag_use_history = linear_rag_config.get("use_history", True) if linear_rag_config else True
        self.mag_use_dedup = linear_rag_config.get("use_dedup", True) if linear_rag_config else True
        self.mag_use_entity_boost = linear_rag_config is not None and linear_rag_config.get("use_entity_boost", False) if linear_rag_config else False
        self.mag_use_entity_match = linear_rag_config.get("use_entity_match", False) if linear_rag_config else False
        self.mag_use_context_window = linear_rag_config.get("use_context_window", False) if linear_rag_config else False
        self.mag_bm25_weight = linear_rag_config.get("bm25_weight", 1.0) if linear_rag_config else 1.0
        self.mag_filter_short = linear_rag_config.get("filter_short", False) if linear_rag_config else False
        self.mag_merge_short = linear_rag_config.get("merge_short", False) if linear_rag_config else False
        self.mag_coref_mode = linear_rag_config.get("coref_mode", "off") if linear_rag_config else "off"
        self.mag_coref_resolve = self.mag_coref_mode != "off"  # back compat
        self.mag_llm_segment = linear_rag_config.get("llm_segment", False) if linear_rag_config else False
        self._llm_segment_carry: List[Tuple[str, str]] = []  # 跨 batch 的上下文
        self._seen_hashes: Set[str] = set()  # 内存去重，不依赖外部查询

        # 实体积累连边: 积累句子直到实体数达到阈值，再 batch LLM 判别
        self.mag_edge_entity_threshold = linear_rag_config.get("edge_entity_threshold", 0) if linear_rag_config else 0
        self._pending_edge_sentences: List[Tuple[str, str]] = []  # [(sid, text), ...]
        self._pending_edge_entity_count: int = 0
        self._pending_uid: str = ""  # 跟踪当前积累的 user_id，跨对话自动 flush
        self._mag_sentence_scopes: Dict[str, str] = {}
        self._graph_path = (graph_config or {}).get("persist_path", "/tmp/mag_graph.json")
        self._graph_save_lock = threading.Lock()

        if mag_enabled:
            llm = self.llm
            self.segmenter = SentenceSegmenter(strategy=segmentation_strategy, llm_client=llm)
            self.entity_extractor = EntityExtractor(
                llm_client=llm, use_llm=(entity_strategy == "llm"),
                use_mem0=(entity_strategy == "mem0"),
            )
            self.relation_detector = DiscriminativeRelationDetector(llm_client=llm)
            self.attention_scorer = EntityAttentionScorer(strategy=attention_strategy, llm_client=llm)
            self.graph_store = GraphStore(graph_config or {})
            self.bfs_retriever = BFSRetriever(self.graph_store)

            # 恢复持久化的图
            self._graph_load()

            # ── Reranker (sentence_transformer, 轻量本地 Cross-Encoder) ──
            self._reranker = None
            try:
                from sentence_transformers import CrossEncoder
                self._reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
                logger.info("Reranker loaded: cross-encoder/ms-marco-MiniLM-L-6-v2")
            except ImportError:
                logger.warning("sentence-transformers not installed, reranker disabled")
            except Exception as e:
                logger.warning("Reranker init failed: %s", str(e))

            # 延迟批量连边: 积累 (sentence_id, text)，LLM 读全文抽关系
            self._pending_relations: List[Tuple[str, str]] = []
            # [(sentence_id, sentence_text), ...]

            lr = linear_rag_config or {}
            self.mag_quality_threshold = lr.get("quality_threshold", 0.3)
            self.mag_max_supplement_rounds = lr.get("max_supplement_rounds", 2)

        if MEM0_TELEMETRY:
            # Create telemetry config manually to avoid deepcopy issues with thread locks
            telemetry_config_dict = {}
            if hasattr(self.config.vector_store.config, 'model_dump'):
                # For pydantic models
                telemetry_config_dict = self.config.vector_store.config.model_dump()
            else:
                # For other objects, manually copy common attributes
                for attr in ['host', 'port', 'path', 'api_key', 'index_name', 'dimension', 'metric']:
                    if hasattr(self.config.vector_store.config, attr):
                        telemetry_config_dict[attr] = getattr(self.config.vector_store.config, attr)

            # Override collection name for telemetry
            telemetry_config_dict['collection_name'] = "mem0migrations"

            # Set path for file-based vector stores
            telemetry_config = _safe_deepcopy_config(self.config.vector_store.config)
            if self.config.vector_store.provider in ["faiss", "qdrant"]:
                provider_path = f"migrations_{self.config.vector_store.provider}"
                telemetry_config_dict['path'] = os.path.join(mem0_dir, provider_path)
                os.makedirs(telemetry_config_dict['path'], exist_ok=True)

            # Create the config object using the same class as the original
            telemetry_config = self.config.vector_store.config.__class__(**telemetry_config_dict)
            self._telemetry_vector_store = VectorStoreFactory.create(
                self.config.vector_store.provider, telemetry_config
            )
        capture_event("mem0.init", self, {"sync_type": "sync"})

    @property
    def entity_store(self):
        """Lazily initialize entity store on first use."""
        if self._entity_store is None:
            entity_config = _safe_deepcopy_config(self.config.vector_store.config)
            entity_collection = f"{self.collection_name}_entities"
            # Set collection name on the cloned config
            if hasattr(entity_config, 'collection_name'):
                entity_config.collection_name = entity_collection
            elif isinstance(entity_config, dict):
                entity_config['collection_name'] = entity_collection
            # For Qdrant, share the existing client to avoid RocksDB lock contention
            # when using embedded mode (path=...). QdrantConfig.client takes precedence
            # over host/port/path.
            if self.config.vector_store.provider == "qdrant" and hasattr(self.vector_store, "client"):
                try:
                    entity_config.client = self.vector_store.client
                except (AttributeError, TypeError):
                    if isinstance(entity_config, dict):
                        entity_config["client"] = self.vector_store.client
            self._entity_store = VectorStoreFactory.create(
                self.config.vector_store.provider, entity_config
            )
        return self._entity_store

    def _upsert_entity(self, entity_text, entity_type, memory_id, filters):
        """Upsert an entity into the entity store, linking it to a memory."""
        try:
            entity_embedding = self.embedding_model.embed(entity_text, "add")
            search_filters = {k: v for k, v in filters.items() if k in ("user_id", "agent_id", "run_id") and v}

            existing = self.entity_store.search(
                query=entity_text,
                vectors=entity_embedding,
                top_k=1,
                filters=search_filters,
            )

            if existing and existing[0].score >= 0.95:
                # Update existing entity's linked_memory_ids
                match = existing[0]
                payload = match.payload or {}
                linked_ids = payload.get("linked_memory_ids", [])
                if memory_id not in linked_ids:
                    linked_ids.append(memory_id)
                    payload["linked_memory_ids"] = linked_ids
                    self.entity_store.update(
                        vector_id=match.id,
                        vector=None,
                        payload=payload,
                    )
            else:
                # Create new entity
                entity_id = str(uuid.uuid4())
                entity_payload = {
                    "data": entity_text,
                    "entity_type": entity_type,
                    "linked_memory_ids": [memory_id],
                    **{k: v for k, v in search_filters.items()},
                }
                self.entity_store.insert(
                    vectors=[entity_embedding],
                    ids=[entity_id],
                    payloads=[entity_payload],
                )
        except Exception as e:
            logger.warning(f"Entity upsert failed for '{entity_text}': {e}")

    def _remove_memory_from_entity_store(self, memory_id, filters):
        """Strip `memory_id` from every entity record scoped to `filters`.

        For each entity whose `linked_memory_ids` contains `memory_id`:
          - remove the id; if the list becomes empty, delete the entity record.
          - otherwise re-embed the entity text and update the payload
            (the vector store's update() requires a vector).

        No-op if the entity store has never been initialized in this process.
        Errors on individual entities are swallowed at debug level; outer
        failures are swallowed at warning level so the primary delete/update
        path is never broken by entity cleanup.
        """
        if self._entity_store is None:
            return
        search_filters = {k: v for k, v in filters.items() if k in ("user_id", "agent_id", "run_id") and v}
        try:
            listed = self.entity_store.list(filters=search_filters, top_k=10000)
            rows = listed[0] if isinstance(listed, (list, tuple)) and listed and isinstance(listed[0], list) else listed
            for row in rows or []:
                try:
                    payload = getattr(row, "payload", None) or {}
                    linked = payload.get("linked_memory_ids", [])
                    if not isinstance(linked, list) or memory_id not in linked:
                        continue
                    remaining = [mid for mid in linked if mid != memory_id]
                    if not remaining:
                        try:
                            self.entity_store.delete(vector_id=row.id)
                        except Exception as e:
                            logger.debug(f"Entity delete failed for id={row.id}: {e}")
                    else:
                        entity_text = payload.get("data")
                        if not isinstance(entity_text, str) or not entity_text:
                            logger.debug(f"Entity id={row.id} missing 'data'; skipping update during cleanup")
                            continue
                        try:
                            vec = self.embedding_model.embed(entity_text, "update")
                        except Exception as e:
                            logger.debug(f"Entity re-embed failed for '{entity_text}': {e}")
                            continue
                        new_payload = {**payload, "linked_memory_ids": remaining}
                        try:
                            self.entity_store.update(
                                vector_id=row.id,
                                vector=vec,
                                payload=new_payload,
                            )
                        except Exception as e:
                            logger.debug(f"Entity update failed for id={row.id}: {e}")
                except Exception as e:
                    logger.debug(f"Entity cleanup error: {e}")
        except Exception as e:
            logger.warning(f"Entity store cleanup failed for memory_id={memory_id}: {e}")

    def _link_entities_for_memory(self, memory_id, text, filters):
        """Extract entities from `text` and link them to `memory_id` in the
        entity store, scoped to `filters`. Simpler single-memory variant of
        Phase 7 in add(): per-entity search-then-update-or-insert via the
        existing `_upsert_entity` helper. Non-fatal on any failure.
        """
        try:
            entities = extract_entities(text)
            if not entities:
                return
            seen = set()
            for entity_type, entity_text in entities:
                key = entity_text.strip().lower()
                if not key or key in seen:
                    continue
                seen.add(key)
                try:
                    self._upsert_entity(entity_text, entity_type, memory_id, filters)
                except Exception as e:
                    logger.debug(f"Entity link failed for '{entity_text}': {e}")
        except Exception as e:
            logger.warning(f"Entity linking failed for memory_id={memory_id}: {e}")

    @classmethod
    def from_config(cls, config_dict: Dict[str, Any]):
        try:
            config = cls._process_config(config_dict)
            config = MemoryConfig(**config_dict)
        except ValidationError as e:
            logger.error(f"Configuration validation error: {e}")
            raise
        return cls(config)

    @staticmethod
    def _process_config(config_dict: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return config_dict
        except ValidationError as e:
            logger.error(f"Configuration validation error: {e}")
            raise

    def _should_use_agent_memory_extraction(self, messages, metadata):
        """Determine whether to use agent memory extraction based on the logic:
        - If agent_id is present and messages contain assistant role -> True
        - Otherwise -> False

        Args:
            messages: List of message dictionaries
            metadata: Metadata containing user_id, agent_id, etc.

        Returns:
            bool: True if should use agent memory extraction, False for user memory extraction
        """
        # Check if agent_id is present in metadata
        has_agent_id = metadata.get("agent_id") is not None

        # Check if there are assistant role messages
        has_assistant_messages = any(msg.get("role") == "assistant" for msg in messages)

        # Use agent memory extraction if agent_id is present and there are assistant messages
        return has_agent_id and has_assistant_messages

    def add(
        self,
        messages,
        *,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        infer: bool = True,
        memory_type: Optional[str] = None,
        prompt: Optional[str] = None,
        default_timestamp: Optional[datetime] = None,  # MAG: session reference date
    ):
        """
        Create a new memory.

        Adds new memories scoped to a single session id (e.g. `user_id`, `agent_id`, or `run_id`). One of those ids is required.

        Args:
            messages (str or List[Dict[str, str]]): The message content or list of messages
                (e.g., `[{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}]`)
                to be processed and stored.
            user_id (str, optional): ID of the user creating the memory. Defaults to None.
            agent_id (str, optional): ID of the agent creating the memory. Defaults to None.
            run_id (str, optional): ID of the run creating the memory. Defaults to None.
            metadata (dict, optional): Metadata to store with the memory. Defaults to None.
            infer (bool, optional): If True (default), an LLM is used to extract key facts from
                'messages' and decide whether to add, update, or delete related memories.
                If False, 'messages' are added as raw memories directly.
            memory_type (str, optional): Specifies the type of memory. Currently, only
                `MemoryType.PROCEDURAL.value` ("procedural_memory") is explicitly handled for
                creating procedural memories (typically requires 'agent_id'). Otherwise, memories
                are treated as general conversational/factual memories.
            prompt (str, optional): Prompt to use for the memory creation. Defaults to None.


        Returns:
            dict: A dictionary containing the result of the memory addition operation, typically
                  including a list of memory items affected (added, updated) under a "results" key.
                  Example for v1.1+: `{"results": [{"id": "...", "memory": "...", "event": "ADD"}]}`

        Raises:
            Mem0ValidationError: If input validation fails (invalid memory_type, messages format, etc.).
            VectorStoreError: If vector store operations fail.
            EmbeddingError: If embedding generation fails.
            LLMError: If LLM operations fail.
            DatabaseError: If database operations fail.
        """

        processed_metadata, effective_filters = _build_filters_and_metadata(
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
            input_metadata=metadata,
        )

        if memory_type is not None and memory_type != MemoryType.PROCEDURAL.value:
            raise Mem0ValidationError(
                message=f"Invalid 'memory_type'. Please pass {MemoryType.PROCEDURAL.value} to create procedural memories.",
                error_code="VALIDATION_002",
                details={"provided_type": memory_type, "valid_type": MemoryType.PROCEDURAL.value},
                suggestion=f"Use '{MemoryType.PROCEDURAL.value}' to create procedural memories."
            )

        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        elif isinstance(messages, dict):
            messages = [messages]

        elif not isinstance(messages, list):
            raise Mem0ValidationError(
                message="messages must be str, dict, or list[dict]",
                error_code="VALIDATION_003",
                details={"provided_type": type(messages).__name__, "valid_types": ["str", "dict", "list[dict]"]},
                suggestion="Convert your input to a string, dictionary, or list of dictionaries."
            )

        if agent_id is not None and memory_type == MemoryType.PROCEDURAL.value:
            results = self._create_procedural_memory(messages, metadata=processed_metadata, prompt=prompt)
            return results

        if self.config.llm.config.get("enable_vision"):
            messages = parse_vision_messages(messages, self.llm, self.config.llm.config.get("vision_details"))
        else:
            messages = parse_vision_messages(messages)

        # ── MAG: 只跑句子管道，不跑 mem0 V3 LLM 提取 ──
        if infer and getattr(self, 'mag_enabled', False):
            mag_results = self._mag_sentence_pipeline(messages, processed_metadata, effective_filters, default_timestamp=default_timestamp)
            return {"results": mag_results}

        vector_store_result = self._add_to_vector_store(messages, processed_metadata, effective_filters, infer)
        return {"results": vector_store_result}

    def _add_to_vector_store(self, messages, metadata, filters, infer):
        if not infer:
            returned_memories = []
            for message_dict in messages:
                if (
                    not isinstance(message_dict, dict)
                    or message_dict.get("role") is None
                    or message_dict.get("content") is None
                ):
                    logger.warning(f"Skipping invalid message format: {message_dict}")
                    continue

                if message_dict["role"] == "system":
                    continue

                per_msg_meta = deepcopy(metadata)
                per_msg_meta["role"] = message_dict["role"]

                actor_name = message_dict.get("name")
                if actor_name:
                    per_msg_meta["actor_id"] = actor_name

                msg_content = message_dict["content"]
                msg_embeddings = self.embedding_model.embed(msg_content, "add")
                mem_id = self._create_memory(msg_content, {msg_content: msg_embeddings}, per_msg_meta)

                returned_memories.append(
                    {
                        "id": mem_id,
                        "memory": msg_content,
                        "event": "ADD",
                        "actor_id": actor_name if actor_name else None,
                        "role": message_dict["role"],
                    }
                )
            return returned_memories

        # === V3 PHASED BATCH PIPELINE ===

        # Phase 0: Context gathering
        session_scope = _build_session_scope(filters)
        last_messages = self.db.get_last_messages(session_scope, limit=10)
        parsed_messages = parse_messages(messages)

        # Phase 1: Existing memory retrieval
        search_filters = {k: v for k, v in filters.items() if k in ("user_id", "agent_id", "run_id") and v}
        query_embedding = self.embedding_model.embed(parsed_messages, "search")
        existing_results = self.vector_store.search(
            query=parsed_messages,
            vectors=query_embedding,
            top_k=10,
            filters=search_filters,
        )

        # Map UUIDs to integers (anti-hallucination)
        existing_memories = []
        uuid_mapping = {}
        for idx, mem in enumerate(existing_results):
            uuid_mapping[str(idx)] = mem.id
            existing_memories.append({"id": str(idx), "text": mem.payload.get("data", "")})

        # Phase 2: LLM extraction (single call)
        is_agent_scoped = bool(filters.get("agent_id")) and not filters.get("user_id")
        system_prompt = ADDITIVE_EXTRACTION_PROMPT
        if is_agent_scoped:
            system_prompt += AGENT_CONTEXT_SUFFIX

        custom_instr = self.custom_instructions

        user_prompt = generate_additive_extraction_prompt(
            existing_memories=existing_memories,
            new_messages=parsed_messages,
            last_k_messages=last_messages,
            custom_instructions=custom_instr,
        )

        try:
            response = self.llm.generate_response(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
            )
        except Exception as e:
            logger.error(f"LLM extraction failed: {e}")
            return []

        # Parse response
        try:
            response = remove_code_blocks(response)
            if not response or not response.strip():
                extracted_memories = []
            else:
                try:
                    extracted_memories = json.loads(response, strict=False).get("memory", [])
                except json.JSONDecodeError:
                    extracted_json = extract_json(response)
                    extracted_memories = json.loads(extracted_json, strict=False).get("memory", [])
        except Exception as e:
            logger.error(f"Error parsing extraction response: {e}")
            extracted_memories = []

        if not extracted_memories:
            # Save messages even if nothing extracted
            self.db.save_messages(messages, session_scope)
            return []

        # Phase 3: Batch embed all extracted memory texts
        mem_texts = [m.get("text", "") for m in extracted_memories if m.get("text")]
        try:
            mem_embeddings_list = self.embedding_model.embed_batch(mem_texts, "add")
            embed_map = dict(zip(mem_texts, mem_embeddings_list))
        except Exception:
            # Fallback: embed individually
            embed_map = {}
            for text in mem_texts:
                try:
                    embed_map[text] = self.embedding_model.embed(text, "add")
                except Exception as e:
                    logger.warning(f"Failed to embed memory text: {e}")

        # Phase 4: Per-memory CPU processing + Phase 5: Hash dedup
        # Build set of existing hashes for dedup
        existing_hashes = set()
        for mem in existing_results:
            h = mem.payload.get("hash") if hasattr(mem, "payload") and mem.payload else None
            if h:
                existing_hashes.add(h)

        records = []  # (memory_id, text, embedding, payload)
        seen_hashes = set()  # dedup within the current batch
        for mem in extracted_memories:
            text = mem.get("text")
            if not text or text not in embed_map:
                continue

            mem_hash = hashlib.md5(text.encode()).hexdigest()
            if mem_hash in existing_hashes or mem_hash in seen_hashes:
                logger.debug(f"Skipping duplicate memory (hash match): {text[:50]}")
                continue
            seen_hashes.add(mem_hash)

            text_lemmatized = lemmatize_for_bm25(text)

            memory_id = str(uuid.uuid4())
            mem_metadata = deepcopy(metadata)
            mem_metadata["data"] = text
            mem_metadata["text_lemmatized"] = text_lemmatized
            mem_metadata["hash"] = mem_hash
            if "created_at" not in mem_metadata:
                mem_metadata["created_at"] = datetime.now(timezone.utc).isoformat()
            mem_metadata["updated_at"] = mem_metadata["created_at"]
            if mem.get("attributed_to"):
                mem_metadata["attributed_to"] = mem["attributed_to"]

            records.append((memory_id, text, embed_map[text], mem_metadata))

        if not records:
            self.db.save_messages(messages, session_scope)
            return []

        # Phase 6: Batch persist
        all_vectors = [r[2] for r in records]
        all_ids = [r[0] for r in records]
        all_payloads = [r[3] for r in records]

        try:
            self.vector_store.insert(
                vectors=all_vectors,
                ids=all_ids,
                payloads=all_payloads,
            )
        except Exception:
            # Fallback: insert one by one
            for mid, vec, pay in zip(all_ids, all_vectors, all_payloads):
                try:
                    self.vector_store.insert(vectors=[vec], ids=[mid], payloads=[pay])
                except Exception as e:
                    logger.error(f"Failed to insert memory {mid}: {e}")

        # Batch history
        history_records = [
            {
                "memory_id": r[0],
                "old_memory": None,
                "new_memory": r[1],
                "event": "ADD",
                "created_at": r[3].get("created_at"),
                "is_deleted": 0,
            }
            for r in records
        ]
        try:
            self.db.batch_add_history(history_records)
        except Exception:
            # Fallback: add one by one
            for hr in history_records:
                try:
                    self.db.add_history(hr["memory_id"], None, hr["new_memory"], "ADD", created_at=hr.get("created_at"))
                except Exception as e:
                    logger.error(f"Failed to add history for {hr['memory_id']}: {e}")

        # Phase 7: Batch entity linking
        try:
            all_texts = [r[1] for r in records]
            all_entities = extract_entities_batch(all_texts)

            # 7a: Global dedup — collect unique entities across all memories
            global_entities = {}  # normalized_key -> (entity_type, entity_text, set of memory_ids)
            for idx, (memory_id, text, embedding, payload) in enumerate(records):
                entities = all_entities[idx] if idx < len(all_entities) else []
                for entity_type, entity_text in entities:
                    key = entity_text.strip().lower()
                    if key in global_entities:
                        global_entities[key][2].add(memory_id)
                    else:
                        global_entities[key] = [entity_type, entity_text, {memory_id}]

            if global_entities:
                ordered_keys = list(global_entities.keys())
                entity_texts = [global_entities[k][1] for k in ordered_keys]

                # 7b: Single batch embed for all unique entities
                try:
                    entity_embeddings = self.embedding_model.embed_batch(entity_texts, "add")
                except Exception:
                    # Fallback: embed individually, use None for failures
                    entity_embeddings = []
                    for t in entity_texts:
                        try:
                            entity_embeddings.append(self.embedding_model.embed(t, "add"))
                        except Exception:
                            entity_embeddings.append(None)

                # Filter out entities with failed embeddings
                valid = [(i, k) for i, k in enumerate(ordered_keys) if entity_embeddings[i] is not None]
                if valid:
                    valid_indices, valid_keys = zip(*valid)
                    valid_vectors = [entity_embeddings[i] for i in valid_indices]

                    # 7c: Batch search for existing entities
                    valid_texts = [global_entities[k][1] for k in valid_keys]
                    existing_matches = self.entity_store.search_batch(
                        queries=valid_texts,
                        vectors_list=valid_vectors,
                        top_k=1,
                        filters=search_filters,
                    )

                    # 7d: Separate into inserts vs updates
                    to_insert_vectors, to_insert_ids, to_insert_payloads = [], [], []
                    for j, key in enumerate(valid_keys):
                        entity_type, entity_text, memory_ids = global_entities[key]
                        matches = existing_matches[j] if j < len(existing_matches) else []

                        if matches and matches[0].score >= 0.95:
                            # Update existing entity
                            match = matches[0]
                            payload = match.payload or {}
                            linked = set(payload.get("linked_memory_ids", []))
                            linked |= memory_ids
                            payload["linked_memory_ids"] = sorted(linked)
                            try:
                                self.entity_store.update(
                                    vector_id=match.id,
                                    vector=None,
                                    payload=payload,
                                )
                            except Exception as e:
                                logger.debug(f"Entity update failed for '{entity_text}': {e}")
                        else:
                            # New entity — collect for batch insert
                            to_insert_vectors.append(valid_vectors[j])
                            to_insert_ids.append(str(uuid.uuid4()))
                            to_insert_payloads.append({
                                "data": entity_text,
                                "entity_type": entity_type,
                                "linked_memory_ids": sorted(memory_ids),
                                **search_filters,
                            })

                    # 7e: Single batch insert for all new entities
                    if to_insert_vectors:
                        try:
                            self.entity_store.insert(
                                vectors=to_insert_vectors,
                                ids=to_insert_ids,
                                payloads=to_insert_payloads,
                            )
                        except Exception as e:
                            logger.warning(f"Batch entity insert failed: {e}")
        except Exception as e:
            logger.warning(f"Batch entity linking failed: {e}")

        # Phase 8: Save messages + return
        self.db.save_messages(messages, session_scope)

        returned_memories = [
            {"id": r[0], "memory": r[1], "event": "ADD"}
            for r in records
        ]

        keys, encoded_ids = process_telemetry_filters(filters)
        capture_event(
            "mem0.add",
            self,
            {"version": self.api_version, "keys": keys, "encoded_ids": encoded_ids, "sync_type": "sync"},
        )
        return returned_memories

    def get(self, memory_id):
        """
        Retrieve a memory by ID.

        Args:
            memory_id (str): ID of the memory to retrieve.

        Returns:
            dict: Retrieved memory.
        """
        capture_event("mem0.get", self, {"memory_id": memory_id, "sync_type": "sync"})
        memory = self.vector_store.get(vector_id=memory_id)
        if not memory:
            return None

        promoted_payload_keys = [
            "user_id",
            "agent_id",
            "run_id",
            "actor_id",
            "role",
        ]

        core_and_promoted_keys = {"data", "hash", "created_at", "updated_at", "id", "text_lemmatized", "attributed_to", *promoted_payload_keys}

        result_item = MemoryItem(
            id=memory.id,
            memory=memory.payload.get("data", ""),
            hash=memory.payload.get("hash"),
            created_at=memory.payload.get("created_at"),
            updated_at=memory.payload.get("updated_at"),
        ).model_dump()

        for key in promoted_payload_keys:
            if key in memory.payload:
                result_item[key] = memory.payload[key]

        additional_metadata = {k: v for k, v in memory.payload.items() if k not in core_and_promoted_keys}
        if additional_metadata:
            result_item["metadata"] = additional_metadata

        return result_item

    def get_all(
        self,
        *,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 20,
        **kwargs,
    ):
        """
        List all memories.

        Args:
            filters (dict): Filter dict containing entity IDs and optional metadata filters.
                Must contain at least one of: user_id, agent_id, run_id.
                Example: filters={"user_id": "u1", "agent_id": "a1"}
            top_k (int, optional): The maximum number of memories to return. Defaults to 20.

        Returns:
            dict: A dictionary containing a list of memories under the "results" key.
                  Example for v1.1+: `{"results": [{"id": "...", "memory": "...", ...}]}`

        Raises:
            ValueError: If filters doesn't contain at least one of user_id, agent_id, run_id,
                or if top_k is invalid.
        """
        # Reject top-level entity params - must use filters instead
        _reject_top_level_entity_params(kwargs, "get_all")

        # Validate top_k
        _validate_search_params(top_k=top_k)

        # Validate and trim entity IDs in filters
        effective_filters = dict(filters) if filters else {}
        if "user_id" in effective_filters:
            effective_filters["user_id"] = _validate_and_trim_entity_id(
                effective_filters["user_id"], "user_id"
            )
        if "agent_id" in effective_filters:
            effective_filters["agent_id"] = _validate_and_trim_entity_id(
                effective_filters["agent_id"], "agent_id"
            )
        if "run_id" in effective_filters:
            effective_filters["run_id"] = _validate_and_trim_entity_id(
                effective_filters["run_id"], "run_id"
            )

        # Validate filters contains at least one entity ID
        if not any(key in effective_filters for key in ("user_id", "agent_id", "run_id")):
            raise ValueError(
                "filters must contain at least one of: user_id, agent_id, run_id. "
                "Example: filters={'user_id': 'u1'}"
            )

        limit = top_k

        keys, encoded_ids = process_telemetry_filters(effective_filters)
        capture_event(
            "mem0.get_all", self, {"limit": limit, "keys": keys, "encoded_ids": encoded_ids, "sync_type": "sync"}
        )

        all_memories_result = self._get_all_from_vector_store(effective_filters, limit)

        return {"results": all_memories_result}

    def _get_all_from_vector_store(self, filters, limit):
        memories_result = self.vector_store.list(filters=filters, top_k=limit)

        # Handle different vector store return formats by inspecting first element
        if isinstance(memories_result, (tuple, list)) and len(memories_result) > 0:
            first_element = memories_result[0]

            # If first element is a container, unwrap one level
            if isinstance(first_element, (list, tuple)):
                actual_memories = first_element
            else:
                # First element is a memory object, structure is already flat
                actual_memories = memories_result
        else:
            actual_memories = memories_result

        promoted_payload_keys = [
            "user_id",
            "agent_id",
            "run_id",
            "actor_id",
            "role",
        ]
        core_and_promoted_keys = {"data", "hash", "created_at", "updated_at", "id", "text_lemmatized", "attributed_to", *promoted_payload_keys}

        formatted_memories = []
        for mem in actual_memories:
            memory_item_dict = MemoryItem(
                id=mem.id,
                memory=mem.payload.get("data", ""),
                hash=mem.payload.get("hash"),
                created_at=mem.payload.get("created_at"),
                updated_at=mem.payload.get("updated_at"),
            ).model_dump(exclude={"score"})

            for key in promoted_payload_keys:
                if key in mem.payload:
                    memory_item_dict[key] = mem.payload[key]

            additional_metadata = {k: v for k, v in mem.payload.items() if k not in core_and_promoted_keys}
            if additional_metadata:
                memory_item_dict["metadata"] = additional_metadata

            formatted_memories.append(memory_item_dict)

        return formatted_memories

    def search(
        self,
        query: str,
        *,
        top_k: int = 20,
        filters: Optional[Dict[str, Any]] = None,
        threshold: float = 0.1,
        rerank: bool = False,
        **kwargs,
    ):
        """
        Searches for memories based on a query.

        Args:
            query (str): Query to search for.
            top_k (int, optional): Maximum number of results to return. Defaults to 20.
            filters (dict): Filter dict containing entity IDs and optional metadata filters.
                Must contain at least one of: user_id, agent_id, run_id.
                Example: filters={"user_id": "u1", "agent_id": "a1"}

                Enhanced metadata filtering with operators:
                - {"key": "value"} - exact match
                - {"key": {"eq": "value"}} - equals
                - {"key": {"ne": "value"}} - not equals
                - {"key": {"in": ["val1", "val2"]}} - in list
                - {"key": {"nin": ["val1", "val2"]}} - not in list
                - {"key": {"gt": 10}} - greater than
                - {"key": {"gte": 10}} - greater than or equal
                - {"key": {"lt": 10}} - less than
                - {"key": {"lte": 10}} - less than or equal
                - {"key": {"contains": "text"}} - contains text
                - {"key": {"icontains": "text"}} - case-insensitive contains
                - {"key": "*"} - wildcard match (any value)
                - {"AND": [filter1, filter2]} - logical AND
                - {"OR": [filter1, filter2]} - logical OR
                - {"NOT": [filter1]} - logical NOT
            threshold (float, optional): Minimum score for a memory to be included. Defaults to 0.1.
            rerank (bool, optional): Whether to rerank results. Defaults to False.

        Returns:
            dict: A dictionary containing the search results under a "results" key.
                  Example for v1.1+: `{"results": [{"id": "...", "memory": "...", "score": 0.8, ...}]}`

        Raises:
            ValueError: If filters doesn't contain at least one of user_id, agent_id, run_id,
                or if threshold/top_k values are invalid.
        """
        # Reject top-level entity params - must use filters instead
        _reject_top_level_entity_params(kwargs, "search")

        # Validate search parameters (before applying defaults)
        _validate_search_params(threshold=threshold, top_k=top_k)

        # Validate and trim entity IDs in filters
        effective_filters = filters.copy() if filters else {}
        if "user_id" in effective_filters:
            effective_filters["user_id"] = _validate_and_trim_entity_id(
                effective_filters["user_id"], "user_id"
            )
        if "agent_id" in effective_filters:
            effective_filters["agent_id"] = _validate_and_trim_entity_id(
                effective_filters["agent_id"], "agent_id"
            )
        if "run_id" in effective_filters:
            effective_filters["run_id"] = _validate_and_trim_entity_id(
                effective_filters["run_id"], "run_id"
            )
        if not any(key in effective_filters for key in ("user_id", "agent_id", "run_id")):
            raise ValueError(
                "filters must contain at least one of: user_id, agent_id, run_id. "
                "Example: filters={'user_id': 'u1'}"
            )

        limit = top_k

        # Apply enhanced metadata filtering if advanced operators are detected
        if self._has_advanced_operators(effective_filters):
            processed_filters = self._process_metadata_filters(effective_filters)
            # Remove logical/operator keys that have been reprocessed
            for logical_key in ("AND", "OR", "NOT"):
                effective_filters.pop(logical_key, None)
            for fk in list(effective_filters.keys()):
                if fk not in ("AND", "OR", "NOT", "user_id", "agent_id", "run_id") and isinstance(effective_filters.get(fk), dict):
                    effective_filters.pop(fk, None)
            effective_filters.update(processed_filters)

        keys, encoded_ids = process_telemetry_filters(effective_filters)
        capture_event(
            "mem0.search",
            self,
            {
                "limit": limit,
                "version": self.api_version,
                "keys": keys,
                "encoded_ids": encoded_ids,
                "sync_type": "sync",
                "threshold": threshold,
                "advanced_filters": bool(filters and self._has_advanced_operators(filters)),
            },
        )

        # MAG owns the full retrieval pipeline. Do not run the base vector
        # search first; _mag_search() calls the vector/BM25 route itself and
        # then fuses graph/context candidates. Running it here causes duplicate
        # retrieval work and can execute legacy inline BFS before ablation flags
        # are applied.
        if getattr(self, 'mag_enabled', False):
            return self._mag_search(query, effective_filters, limit)

        original_memories = self._search_vector_store(query, effective_filters, limit, threshold)

        # Apply reranking if enabled and reranker is available
        if rerank and self.reranker and original_memories:
            try:
                reranked_memories = self.reranker.rerank(query, original_memories, limit)
                original_memories = reranked_memories
            except Exception as e:
                logger.warning(f"Reranking failed, using original results: {e}")

        return {"results": original_memories}

    def _process_metadata_filters(self, metadata_filters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process enhanced metadata filters and convert them to vector store compatible format.

        Args:
            metadata_filters: Enhanced metadata filters with operators

        Returns:
            Dict of processed filters compatible with vector store
        """
        processed_filters = {}

        def process_condition(key: str, condition: Any) -> Dict[str, Any]:
            if not isinstance(condition, dict):
                # Simple equality: {"key": "value"}
                if condition == "*":
                    # Wildcard: match everything for this field (implementation depends on vector store)
                    return {key: "*"}
                return {key: condition}

            result = {}
            for operator, value in condition.items():
                # Map platform operators to universal format that can be translated by each vector store
                operator_map = {
                    "eq": "eq", "ne": "ne", "gt": "gt", "gte": "gte",
                    "lt": "lt", "lte": "lte", "in": "in", "nin": "nin",
                    "contains": "contains", "icontains": "icontains"
                }

                if operator in operator_map:
                    result.setdefault(key, {})[operator_map[operator]] = value
                else:
                    raise ValueError(f"Unsupported metadata filter operator: {operator}")
            return result

        for key, value in metadata_filters.items():
            if key == "AND":
                # Logical AND: combine multiple conditions
                if not isinstance(value, list):
                    raise ValueError("AND operator requires a list of conditions")
                for condition in value:
                    for sub_key, sub_value in condition.items():
                        processed_filters.update(process_condition(sub_key, sub_value))
            elif key == "OR":
                # Logical OR: Pass through to vector store for implementation-specific handling
                if not isinstance(value, list) or not value:
                    raise ValueError("OR operator requires a non-empty list of conditions")
                # Store OR conditions in a way that vector stores can interpret
                processed_filters["$or"] = []
                for condition in value:
                    or_condition = {}
                    for sub_key, sub_value in condition.items():
                        or_condition.update(process_condition(sub_key, sub_value))
                    processed_filters["$or"].append(or_condition)
            elif key == "NOT":
                # Logical NOT: Pass through to vector store for implementation-specific handling
                if not isinstance(value, list) or not value:
                    raise ValueError("NOT operator requires a non-empty list of conditions")
                processed_filters["$not"] = []
                for condition in value:
                    not_condition = {}
                    for sub_key, sub_value in condition.items():
                        not_condition.update(process_condition(sub_key, sub_value))
                    processed_filters["$not"].append(not_condition)
            else:
                processed_filters.update(process_condition(key, value))

        return processed_filters

    def _has_advanced_operators(self, filters: Dict[str, Any]) -> bool:
        """
        Check if filters contain advanced operators that need special processing.

        Args:
            filters: Dictionary of filters to check

        Returns:
            bool: True if advanced operators are detected
        """
        if not isinstance(filters, dict):
            return False

        for key, value in filters.items():
            # Check for platform-style logical operators
            if key in ["AND", "OR", "NOT"]:
                return True
            # Check for comparison operators (without $ prefix for universal compatibility)
            if isinstance(value, dict):
                for op in value.keys():
                    if op in ["eq", "ne", "gt", "gte", "lt", "lte", "in", "nin", "contains", "icontains"]:
                        return True
            # Check for wildcard values
            if value == "*":
                return True
        return False

    def _search_vector_store(self, query, filters, limit, threshold=0.1):
        # Guard against None threshold (backward compat)
        if threshold is None:
            threshold = 0.1

        # Step 1: Preprocess query
        query_lemmatized = lemmatize_for_bm25(query)
        query_entities = extract_entities(query)

        # Step 2: Embed query
        embeddings = self.embedding_model.embed(query, "search")

        # Step 3: Semantic search (over-fetch for scoring pool)
        internal_limit = max(limit * 4, 60)
        semantic_results = self.vector_store.search(
            query=query, vectors=embeddings, top_k=internal_limit, filters=filters
        )

        # Step 4: Keyword search (if store supports it)
        keyword_results = self.vector_store.keyword_search(
            query=query_lemmatized, top_k=internal_limit, filters=filters
        )

        # Step 5: Compute BM25 scores from keyword results
        bm25_scores = {}
        if keyword_results is not None:
            midpoint, steepness = get_bm25_params(query, lemmatized=query_lemmatized)
            for mem in keyword_results:
                mem_id = str(mem.id) if hasattr(mem, 'id') else str(mem.get('id', ''))
                raw_score = mem.score if hasattr(mem, 'score') else mem.get('score', 0)
                if raw_score and raw_score > 0:
                    bm25_scores[mem_id] = normalize_bm25(raw_score, midpoint, steepness)

        # Step 6: Compute entity boosts
        entity_boosts = {}
        if query_entities:
            entity_boosts = self._compute_entity_boosts(query_entities, filters)

        # Step 7: Build candidate set from semantic results
        candidates = []
        for mem in semantic_results:
            mem_id = str(mem.id)
            candidates.append({
                "id": mem_id,
                "score": mem.score,
                "payload": mem.payload if hasattr(mem, 'payload') else {},
            })

        # Step 7b: Apply BM25 weight (ablation: mag_bm25_weight)
        if getattr(self, 'mag_bm25_weight', 1.0) != 1.0 and bm25_scores:
            bm25_scores = {k: v * self.mag_bm25_weight for k, v in bm25_scores.items()}

        # Step 8: Score and rank
        scored_results = score_and_rank(
            semantic_results=candidates,
            bm25_scores=bm25_scores,
            entity_boosts=entity_boosts,
            threshold=threshold,
            top_k=limit,
        )

        # ── MAG Step 8b: BFS 图拓扑检索 (路径感知 + 语义相似度) ──
        # 从 query entities 出发，遍历图边，通过 source_sentence_ids 检索原句，
        # 用 query-sentence 语义相似度过滤，收集超过阈值的句子
        if getattr(self, 'mag_enable_inline_bfs', False) and query_entities:
            try:
                session_scope = _build_session_scope(filters or {})
                q_ews = []
                for e in query_entities:
                    ename = e[1].strip().lower()
                    q_ews.append(EntityWeight(name=ename, attention_weight=0.6, entity_type=e[0]))
                    if ' ' in ename:
                        for word in ename.split():
                            if len(word) > 2:
                                q_ews.append(EntityWeight(name=word, attention_weight=0.4, entity_type='TOKEN'))

                scored_ids = {s["id"] for s in scored_results}

                # 语义相似度回调: sentence_id → cosine sim
                def _get_sim(identifier):
                    import numpy as np
                    try:
                        # 判 UUID vs 三元组文本
                        if len(str(identifier)) >= 32 and '-' in str(identifier):
                            # UUID: 从 Qdrant 取向量
                            srec = self.vector_store.client.retrieve(
                                collection_name=self.vector_store.collection_name,
                                ids=[str(identifier)], with_payload=False, with_vectors=True,
                            )
                            if srec and srec[0].vector is not None and embeddings is not None:
                                vec = srec[0].vector
                                if isinstance(vec, dict):
                                    dense = vec.get("", None)
                                else:
                                    dense = vec
                                if dense is not None and hasattr(dense, '__len__') and len(dense) > 0:
                                    v = np.array(dense, dtype=np.float32)
                                    qv = np.array(embeddings, dtype=np.float32)
                                    return float(np.dot(v, qv) / (np.linalg.norm(v) * np.linalg.norm(qv) + 1e-8))
                        else:
                            # 三元组文本: 直接 embed 后算 cosine sim
                            if embeddings is not None:
                                tv = self.embedding_model.embed(str(identifier), "search")
                                v = np.array(tv, dtype=np.float32)
                                qv = np.array(embeddings, dtype=np.float32)
                                return float(np.dot(v, qv) / (np.linalg.norm(v) * np.linalg.norm(qv) + 1e-8))
                    except Exception:
                        pass
                    return 0.0

                # 路径感知 BFS: tolerance=2, sim_threshold=0.3
                bfs_paths = self.bfs_retriever.search_paths(
                    query_entities=q_ews,
                    query_embedding=embeddings,
                    get_semantic_sim=_get_sim,
                    max_hops=3,
                    tolerance=2,
                    sim_threshold=0.3,
                    max_results=limit,
                    session_scope=session_scope,
                )

                seen_bfs_texts = set()  # 按文本去重，避免多条路径引用同一条边句子
                for path in bfs_paths:
                    path_sids = [str(s) for s in path.get("sentences", [])]
                    if not path_sids:
                        continue
                    path_score = path.get("path_score", 0.5)
                    for sid_str in path_sids:
                        if sid_str in scored_ids:
                            # 替换为 BFS 路径来源 + 沿路径分数提权
                            for s in scored_results:
                                if s["id"] == sid_str:
                                    s["score"] = max(s["score"], path_score)
                                    s.setdefault("payload", {})["_bfs_source"] = "graph_bfs"
                                    break
                        else:
                            try:
                                srec = self.vector_store.client.retrieve(
                                    collection_name=self.vector_store.collection_name,
                                    ids=[sid_str], with_payload=True,
                                )
                                if srec and len(srec) > 0:
                                    p = dict(srec[0].payload)
                                    txt = p.get("data", "").strip()
                                    if txt and txt not in seen_bfs_texts:
                                        seen_bfs_texts.add(txt)
                                        p["_bfs_source"] = "graph_bfs"
                                        scored_results.append({
                                            "id": sid_str,
                                            "score": path_score,
                                            "payload": p,
                                        })
                            except Exception:
                                pass

                # BFS 结果可能排在末尾，按分数重排确保参与 top-k 截断
                scored_results.sort(key=lambda x: x.get("score", 0), reverse=True)
            except Exception as e:
                logger.warning("MAG BFS exception: %s", str(e)[:200])
                logger.debug("MAG BFS skipped: %s", str(e)[:100])

        # Step 9: Format results
        promoted_payload_keys = [
            "user_id",
            "agent_id",
            "run_id",
            "actor_id",
            "role",
        ]
        core_and_promoted_keys = {"data", "hash", "created_at", "updated_at", "id", "text_lemmatized", "attributed_to", *promoted_payload_keys}

        original_memories = []
        for scored in scored_results:
            payload = scored.get("payload") or {}

            if not payload.get("data"):
                continue  # Skip candidates with no payload data

            memory_item_dict = MemoryItem(
                id=scored["id"],
                memory=payload.get("data", ""),
                hash=payload.get("hash"),
                created_at=payload.get("created_at"),
                updated_at=payload.get("updated_at"),
                score=scored["score"],
            ).model_dump()

            if payload.get("_bfs_source"):
                memory_item_dict["source"] = payload["_bfs_source"]

            for key in promoted_payload_keys:
                if key in payload:
                    memory_item_dict[key] = payload[key]

            additional_metadata = {k: v for k, v in payload.items() if k not in core_and_promoted_keys}
            if additional_metadata:
                if not memory_item_dict.get("metadata"):
                    memory_item_dict["metadata"] = {}
                memory_item_dict["metadata"].update(additional_metadata)

            original_memories.append(memory_item_dict)

        return original_memories

    def _compute_entity_boosts(self, query_entities, filters):
        """Compute per-memory entity boosts from entity store search.

        For each extracted entity from the query:
        1. Embed the entity text
        2. Search the entity store (threshold >= 0.5)
        3. For each matched entity, boost its linked memories

        Returns:
            Dict mapping memory_id (str) -> max entity boost [0, 0.5].
        """
        # Deduplicate entities (max 8)
        seen = set()
        deduped = []
        for entity_type, entity_text in query_entities[:8]:
            key = entity_text.strip().lower()
            if key and key not in seen:
                seen.add(key)
                deduped.append((entity_type, entity_text))

        if not deduped:
            return {}

        search_filters = {k: v for k, v in filters.items() if k in ("user_id", "agent_id", "run_id") and v}
        memory_boosts = {}

        try:
            for _, entity_text in deduped:
                entity_embedding = self.embedding_model.embed(entity_text, "search")
                matches = self.entity_store.search(
                    query=entity_text,
                    vectors=entity_embedding,
                    top_k=500,
                    filters=search_filters,
                )

                for match in matches:
                    similarity = match.score if hasattr(match, 'score') else 0.0
                    if similarity < 0.5:
                        continue

                    payload = match.payload if hasattr(match, 'payload') else {}
                    linked_memory_ids = payload.get("linked_memory_ids", [])
                    if not isinstance(linked_memory_ids, list):
                        continue

                    # Spread-attenuated boost: entities linking to many memories get attenuated
                    num_linked = max(len(linked_memory_ids), 1)
                    memory_count_weight = 1.0 / (1.0 + 0.001 * ((num_linked - 1) ** 2))
                    boost = similarity * ENTITY_BOOST_WEIGHT * memory_count_weight

                    for memory_id in linked_memory_ids:
                        if memory_id:
                            memory_key = str(memory_id)
                            memory_boosts[memory_key] = max(memory_boosts.get(memory_key, 0.0), boost)

        except Exception as e:
            logger.warning(f"Entity boost computation failed: {e}")

        return memory_boosts

    def update(self, memory_id, data, metadata: Optional[Dict[str, Any]] = None):
        """
        Update a memory by ID.

        Args:
            memory_id (str): ID of the memory to update.
            data (str): New content to update the memory with.
            metadata (dict, optional): Metadata to update with the memory. Defaults to None.

        Returns:
            dict: Success message indicating the memory was updated.

        Example:
            >>> m.update(memory_id="mem_123", data="Likes to play tennis on weekends")
            {'message': 'Memory updated successfully!'}
        """
        capture_event("mem0.update", self, {"memory_id": memory_id, "sync_type": "sync"})

        existing_embeddings = {data: self.embedding_model.embed(data, "update")}

        self._update_memory(memory_id, data, existing_embeddings, metadata)
        return {"message": "Memory updated successfully!"}

    def delete(self, memory_id):
        """
        Delete a memory by ID.

        Args:
            memory_id (str): ID of the memory to delete.
        """
        capture_event("mem0.delete", self, {"memory_id": memory_id, "sync_type": "sync"})

        existing_memory = self.vector_store.get(vector_id=memory_id)
        if existing_memory is None:
            raise ValueError(f"Memory with id {memory_id} not found")

        self._delete_memory(memory_id, existing_memory)
        return {"message": "Memory deleted successfully!"}

    def delete_all(self, user_id: Optional[str] = None, agent_id: Optional[str] = None, run_id: Optional[str] = None):
        """
        Delete all memories.

        Args:
            user_id (str, optional): ID of the user to delete memories for. Defaults to None.
            agent_id (str, optional): ID of the agent to delete memories for. Defaults to None.
            run_id (str, optional): ID of the run to delete memories for. Defaults to None.
        """
        filters: Dict[str, Any] = {}
        if user_id:
            filters["user_id"] = user_id
        if agent_id:
            filters["agent_id"] = agent_id
        if run_id:
            filters["run_id"] = run_id

        if not filters:
            raise ValueError(
                "At least one filter is required to delete all memories. If you want to delete all memories, use the `reset()` method."
            )

        keys, encoded_ids = process_telemetry_filters(filters)
        capture_event("mem0.delete_all", self, {"keys": keys, "encoded_ids": encoded_ids, "sync_type": "sync"})
        # delete all vector memories and reset the collections
        memories = self.vector_store.list(filters=filters)[0]
        for memory in memories:
            self._delete_memory(memory.id)

        logger.info(f"Deleted {len(memories)} memories")

        return {"message": "Memories deleted successfully!"}

    def history(self, memory_id):
        """
        Get the history of changes for a memory by ID.

        Args:
            memory_id (str): ID of the memory to get history for.

        Returns:
            list: List of changes for the memory.
        """
        capture_event("mem0.history", self, {"memory_id": memory_id, "sync_type": "sync"})
        return self.db.get_history(memory_id)

    def _create_memory(self, data, existing_embeddings, metadata=None):
        logger.debug(f"Creating memory with {data=}")
        if data in existing_embeddings:
            embeddings = existing_embeddings[data]
        else:
            embeddings = self.embedding_model.embed(data, memory_action="add")
        memory_id = str(uuid.uuid4())
        new_metadata = deepcopy(metadata) if metadata is not None else {}
        new_metadata["data"] = data
        new_metadata["hash"] = hashlib.md5(data.encode()).hexdigest()
        if "created_at" not in new_metadata:
            new_metadata["created_at"] = datetime.now(timezone.utc).isoformat()
        new_metadata["updated_at"] = new_metadata["created_at"]
        new_metadata["text_lemmatized"] = lemmatize_for_bm25(data)

        self.vector_store.insert(
            vectors=[embeddings],
            ids=[memory_id],
            payloads=[new_metadata],
        )
        self.db.add_history(
            memory_id,
            None,
            data,
            "ADD",
            created_at=new_metadata.get("created_at"),
            updated_at=new_metadata.get("updated_at"),
            actor_id=new_metadata.get("actor_id"),
            role=new_metadata.get("role"),
        )
        return memory_id

    def _create_procedural_memory(self, messages, metadata=None, prompt=None):
        """
        Create a procedural memory

        Args:
            messages (list): List of messages to create a procedural memory from.
            metadata (dict): Metadata to create a procedural memory from.
            prompt (str, optional): Prompt to use for the procedural memory creation. Defaults to None.
        """
        logger.info("Creating procedural memory")

        parsed_messages = [
            {"role": "system", "content": prompt or PROCEDURAL_MEMORY_SYSTEM_PROMPT},
            *messages,
            {
                "role": "user",
                "content": "Create procedural memory of the above conversation.",
            },
        ]

        try:
            procedural_memory = self.llm.generate_response(messages=parsed_messages)
            procedural_memory = remove_code_blocks(procedural_memory)
        except Exception as e:
            logger.error(f"Error generating procedural memory summary: {e}")
            raise

        if metadata is None:
            raise ValueError("Metadata cannot be done for procedural memory.")

        metadata = {**metadata, "memory_type": MemoryType.PROCEDURAL.value}
        embeddings = self.embedding_model.embed(procedural_memory, memory_action="add")
        memory_id = self._create_memory(procedural_memory, {procedural_memory: embeddings}, metadata=metadata)
        capture_event("mem0._create_procedural_memory", self, {"memory_id": memory_id, "sync_type": "sync"})

        result = {"results": [{"id": memory_id, "memory": procedural_memory, "event": "ADD"}]}

        return result

    def _update_memory(self, memory_id, data, existing_embeddings, metadata=None):
        logger.info(f"Updating memory with {data=}")

        try:
            existing_memory = self.vector_store.get(vector_id=memory_id)
        except Exception:
            logger.error(f"Error getting memory with ID {memory_id} during update.")
            raise ValueError(f"Error getting memory with ID {memory_id}. Please provide a valid 'memory_id'")

        if existing_memory is None:
            raise ValueError(f"Memory with id {memory_id} not found. Please provide a valid 'memory_id'")

        prev_value = existing_memory.payload.get("data")

        new_metadata = deepcopy(metadata) if metadata is not None else {}

        new_metadata["data"] = data
        new_metadata["hash"] = hashlib.md5(data.encode()).hexdigest()
        new_metadata["text_lemmatized"] = lemmatize_for_bm25(data)
        new_metadata["created_at"] = existing_memory.payload.get("created_at")
        new_metadata["updated_at"] = datetime.now(timezone.utc).isoformat()

        # Preserve session identifiers from existing memory only if not provided in new metadata
        if "user_id" not in new_metadata and "user_id" in existing_memory.payload:
            new_metadata["user_id"] = existing_memory.payload["user_id"]
        if "agent_id" not in new_metadata and "agent_id" in existing_memory.payload:
            new_metadata["agent_id"] = existing_memory.payload["agent_id"]
        if "run_id" not in new_metadata and "run_id" in existing_memory.payload:
            new_metadata["run_id"] = existing_memory.payload["run_id"]
        if "actor_id" in existing_memory.payload:
            new_metadata["actor_id"] = existing_memory.payload["actor_id"]
        if "role" not in new_metadata and "role" in existing_memory.payload:
            new_metadata["role"] = existing_memory.payload["role"]

        if data in existing_embeddings:
            embeddings = existing_embeddings[data]
        else:
            embeddings = self.embedding_model.embed(data, "update")

        self.vector_store.update(
            vector_id=memory_id,
            vector=embeddings,
            payload=new_metadata,
        )
        logger.info(f"Updating memory with ID {memory_id=} with {data=}")

        self.db.add_history(
            memory_id,
            prev_value,
            data,
            "UPDATE",
            created_at=new_metadata["created_at"],
            updated_at=new_metadata["updated_at"],
            actor_id=new_metadata.get("actor_id"),
            role=new_metadata.get("role"),
        )

        # Entity-store cleanup: strip this memory's id from old-text entities,
        # then re-extract entities from the new text and link them back.
        session_filters = {k: new_metadata[k] for k in ("user_id", "agent_id", "run_id") if new_metadata.get(k)}
        self._remove_memory_from_entity_store(memory_id, session_filters)
        self._link_entities_for_memory(memory_id, data, session_filters)

        return memory_id

    def _delete_memory(self, memory_id, existing_memory=None):
        logger.info(f"Deleting memory with {memory_id=}")
        if existing_memory is None:
            existing_memory = self.vector_store.get(vector_id=memory_id)
            if existing_memory is None:
                raise ValueError(f"Memory with id {memory_id} not found. Please provide a valid 'memory_id'")
        prev_value = existing_memory.payload.get("data", "")
        created_at = _normalize_iso_timestamp_to_utc(existing_memory.payload.get("created_at"))
        updated_at = datetime.now(timezone.utc).isoformat()
        payload = existing_memory.payload or {}
        session_filters = {k: payload[k] for k in ("user_id", "agent_id", "run_id") if payload.get(k)}
        self.vector_store.delete(vector_id=memory_id)
        self.db.add_history(
            memory_id,
            prev_value,
            None,
            "DELETE",
            created_at=created_at,
            updated_at=updated_at,
            actor_id=existing_memory.payload.get("actor_id"),
            role=existing_memory.payload.get("role"),
            is_deleted=1,
        )

        # Entity-store cleanup: strip this memory's id from any entity records
        # that linked to it. Non-fatal — the helper swallows errors.
        self._remove_memory_from_entity_store(memory_id, session_filters)

        return memory_id

    def reset(self):
        """
        Reset the memory store by:
            Deletes the vector store collection
            Resets the database
            Recreates the vector store with a new client
        """
        logger.warning("Resetting all memories")

        if hasattr(self.db, "connection") and self.db.connection:
            self.db.connection.execute("DROP TABLE IF EXISTS history")
            self.db.connection.close()

        self.db = SQLiteManager(self.config.history_db_path)

        if hasattr(self.vector_store, "reset"):
            self.vector_store = VectorStoreFactory.reset(self.vector_store)
        else:
            logger.warning("Vector store does not support reset. Skipping.")
            self.vector_store.delete_col()
            self.vector_store = VectorStoreFactory.create(
                self.config.vector_store.provider, self.config.vector_store.config
            )
        # Reset entity store if initialized
        if self._entity_store is not None:
            try:
                self._entity_store.reset()
            except Exception as e:
                logger.warning(f"Failed to reset entity store: {e}")
            self._entity_store = None

        capture_event("mem0.reset", self, {"sync_type": "sync"})

    def close(self):
        """Release resources held by this Memory instance (SQLite connections, etc.)."""
        if hasattr(self, "db") and self.db is not None:
            self.db.close()
            self.db = None

    def chat(self, query):
        raise NotImplementedError("Chat function not implemented yet.")

    # ==================================================================
    # MAG 句子+图索引管道
    # ==================================================================

    def _mag_sentence_pipeline(self, messages, metadata: Dict[str, Any], filters: Dict[str, Any], default_timestamp: datetime = None) -> List[Dict[str, Any]]:
        """S1分句→S2实体→S3Attention→S4存原始句子→S5LLM判别连边→S6图索引"""
        import hashlib as _hashlib
        from mem0.utils.lemmatization import lemmatize_for_bm25 as _lemmatize

        session_scope = _build_session_scope(filters)
        sentences = self.segmenter.segment(messages, default_timestamp=default_timestamp)
        if not sentences:
            return []
        sentence_texts = [s[0] for s in sentences]

        all_entities = self.entity_extractor.extract_batch(sentence_texts)
        all_ew: List[List[EntityWeight]] = []
        for text, entities in zip(sentence_texts, all_entities):
            ews = self.attention_scorer.score(text, [e[0] for e in entities], [e[1] for e in entities]) if entities and self.mag_use_attention else []
            if not self.mag_use_attention and entities:
                ews = [EntityWeight(name=e[0], attention_weight=0.5, entity_type=e[1]) for e in entities]
            all_ew.append(ews)

        try:
            embeddings = self.embedding_model.embed_batch(sentence_texts, "add")
        except Exception:
            embeddings = [self.embedding_model.embed(t, "add") for t in sentence_texts]

        mag_meta = dict(metadata)
        mag_meta[_MAG_KEY] = _MAG_VAL
        sentence_ids = []
        filtered_ew = []  # 与 sentence_ids 对齐的实体列表
        filtered_idx = []  # 原始 sentences 索引
        dedup_skipped = 0
        for i, (text, speaker, ts) in enumerate(sentences):
            sid = str(uuid.uuid4())
            mem_hash = _hashlib.md5(text.encode()).hexdigest()

            # ── 去重 (可 ablation: mag_use_dedup=False 跳过) ──
            if self.mag_use_dedup:
                if mem_hash in self._seen_hashes:
                    dedup_skipped += 1
                    continue
                self._seen_hashes.add(mem_hash)

            # ── 短句合并 (可 ablation: mag_merge_short=False 跳过) ──
            # LLM segment模式自行处理分句，跳过规则合并避免冲突
            if self.mag_merge_short and not self.mag_llm_segment:
                if len(text) < 25 and len(all_ew[i]) == 0:
                    if sentence_ids:
                        # 更新前一句的 payload: 拼接文本和实体
                        prev_sid = sentence_ids[-1]
                        try:
                            prev_rec = self.vector_store.get(vector_id=prev_sid)
                            if prev_rec and hasattr(prev_rec, "payload"):
                                prev_p = prev_rec.payload
                                merged_text = prev_p.get("data", "") + " " + text
                                merged_ents = list(prev_p.get("entities", []))
                                self.vector_store.update(
                                    vector_id=prev_sid, vector=None,
                                    payload={"data": merged_text, "entities": merged_ents},
                                )
                        except Exception:
                            pass
                    continue  # 不加入 sentence_ids，不参与检索

            sentence_ids.append(sid)
            self._mag_sentence_scopes[sid] = session_scope
            filtered_ew.append(all_ew[i])  # 对齐: 跳过 dedup/merge 后保持索引一致
            filtered_idx.append(i)  # 记录原始索引
            # 标准化 Schema: [ID] [Timestamp] [Entities] [RawText]
            payload = {
                "data": text,
                "entities": [e.to_dict() for e in all_ew[i]],
                "created_at": ts.isoformat(),
                "updated_at": ts.isoformat(),
                "sentence_id": sid,
                "speaker": speaker,
                "text_lemmatized": _lemmatize(text),
                "hash": mem_hash,
                **mag_meta,
            }
            try:
                self.vector_store.insert(vectors=[embeddings[i]], ids=[sid], payloads=[payload])
            except Exception as e:
                logger.error("MAG insert failed: %s", str(e)[:100])
                continue

            # ── SQLite 历史 (可 ablation: mag_use_history=False 跳过) ──
            if self.mag_use_history:
                try:
                    self.db.add_history(sid, None, text, "ADD",
                                        created_at=payload["created_at"],
                                        is_deleted=0)
                except Exception:
                    pass

            # ── mem0 entity_store 写入 (可 ablation: mag_use_entity_store=False 跳过) ──
            # 使 mem0 的 _compute_entity_boosts() 能通过 linked_memory_ids 找到 MAG 句子
            if self.mag_use_entity_store and all_ew[i]:
                try:
                    for ew in all_ew[i]:
                        scoped_filters = {k: v for k, v in filters.items()
                                         if k in ("user_id", "agent_id", "run_id") and v}
                        self._upsert_entity(ew.name,
                                           f"MAG_{ew.entity_type}" if ew.entity_type else "MAG_ENTITY",
                                           sid, scoped_filters)
                except Exception:
                    pass

        if dedup_skipped:
            logger.info("MAG dedup: skipped %d duplicate sentences", dedup_skipped)

        # ── S4b: 上下文窗口 — 存 prev/next_sentence_id 到 payload ──
        if self.mag_use_context_window and len(sentence_ids) >= 2:
            ctx_updated = 0
            for i, sid in enumerate(sentence_ids):
                upd = {}
                if i > 0:
                    upd["prev_sentence_id"] = sentence_ids[i - 1]
                if i < len(sentence_ids) - 1:
                    upd["next_sentence_id"] = sentence_ids[i + 1]
                if upd:
                    try:
                        self.vector_store.update(vector_id=sid, vector=None, payload=upd)
                        ctx_updated += 1
                    except Exception:
                        pass
            if ctx_updated:
                logger.debug("MAG context_window: linked %d sentences", ctx_updated)

        # ── S5: 关系连边 — 实体积累模式 (按实体数量阈值触发 LLM) ──
        if self.mag_relation_batch_size > 0:
            if self.mag_edge_entity_threshold > 0:
                # 积累模式: 跨 add() 积累句子，检测 user_id 切换自动 flush
                current_uid = filters.get("user_id", "")
                if self._pending_uid and current_uid != self._pending_uid:
                    self._flush_edge_sentences()
                self._pending_uid = current_uid
                for i, sid in enumerate(sentence_ids):
                    text = sentence_texts[i]
                    if text.strip():
                        self._pending_edge_sentences.append((sid, text))
                        self._pending_edge_entity_count += len(all_ew[i])
                if self._pending_edge_entity_count >= self.mag_edge_entity_threshold:
                    self._flush_edge_sentences()
            else:
                # 即时模式: 每个 add() 内的句子直接 LLM 连边（+ 可选指代消解）
                items = [(sid, sentence_texts[i]) for i, sid in enumerate(sentence_ids) if sentence_texts[i].strip()]
                if len(items) >= 2:
                    try:
                        # 消解+关系 (可 ablation: mag_llm_segment / mag_coref_mode)
                        if self.mag_llm_segment:
                            try:
                                ts_map = {}
                                for sid, _ in items:
                                    try:
                                        rec = self.vector_store.get(vector_id=sid)
                                        if rec and hasattr(rec, "payload"):
                                            t = rec.payload.get("created_at", "")[:10]
                                            if t: ts_map[sid] = t
                                    except Exception: pass
                                result = self.relation_detector.segment_and_relate(
                                    items, carry_over=self._llm_segment_carry, timestamps=ts_map)
                                self._llm_segment_carry = result.get("carry_over", [])
                                # 先存储原子事实（生成新 fact UUID），再用新 fact ID 建边
                                fact_pairs = self._store_atomic_facts_return_ids(
                                    result.get("facts", []), result.get("merged_away", []), items)
                                try:
                                    if fact_pairs:
                                        edge_triples = self.relation_detector.extract_from_context(fact_pairs)
                                    else:
                                        edge_triples = self.relation_detector.extract_from_context(items)
                                    for t in edge_triples:
                                        try:
                                            self.graph_store.add_relation(
                                                t.head, t.relation, t.tail, t.source_sentence_id, t.confidence,
                                                session_scope=self._mag_session_scope_for_sentence(t.source_sentence_id, session_scope),
                                            )
                                        except Exception: pass
                                except Exception as e: logger.warning("S5 relation failed: %s", str(e)[:200])
                            except Exception as e:
                                logger.warning("LLM segment+relate: %s", str(e)[:200])
                        else:
                            resolved = {}
                            if self.mag_coref_mode == "rule":
                                resolved = self.relation_detector.resolve_coreferences_rule(items)
                            elif self.mag_coref_mode == "llm":
                                try:
                                    resolved = self.relation_detector.resolve_coreferences(items)
                                except Exception as e:
                                    logger.debug("Coref LLM: %s", str(e)[:100])
                            for sid, rtext in resolved.items():
                                try:
                                    self.vector_store.update(
                                        vector_id=sid, vector=None, payload={"data": rtext})
                                except Exception: pass
                            triples = self.relation_detector.extract_from_context(items)
                            for t in triples:
                                try:
                                    self.graph_store.add_relation(
                                        t.head, t.relation, t.tail, t.source_sentence_id, t.confidence,
                                        session_scope=self._mag_session_scope_for_sentence(t.source_sentence_id, session_scope),
                                    )
                                except Exception:
                                    pass
                    except Exception as e:
                        logger.debug("S5 relation detection: %s", str(e)[:100])

        # ── S6: 实体节点立即写入图 ──
        for i, sid in enumerate(sentence_ids):
            for ew in filtered_ew[i]:
                try:
                    self.graph_store.upsert_entity(
                        ew.name,
                        ew.attention_weight,
                        sid,
                        ew.entity_type,
                        session_scope=session_scope,
                    )
                except Exception:
                    pass

        # S5+S6 完后保存图到磁盘
        self._graph_save()

        gs = self.graph_store.stats()
        logger.info("MAG: %d sentences → graph(entities=%d, edges=%d)", len(sentence_texts), gs["num_entities"], gs["num_relations"])
        return [{"id": sid, "memory": sentence_texts[filtered_idx[i]], "event": "ADD",
                 "entities": [e.to_dict() for e in filtered_ew[i]], "speaker": sentences[filtered_idx[i]][1],
                 "created_at": sentences[filtered_idx[i]][2].isoformat()} for i, sid in enumerate(sentence_ids)]

    @property
    def pending_relations_count(self) -> int:
        """待处理的关系数 (尚未调用 flush_relations)"""
        return len(self._pending_relations) if hasattr(self, '_pending_relations') else 0

    def flush_relations(self, batch_size: int = None) -> Dict[str, int]:
        """
        批量处理所有积累的待处理关系 — 一次或分批次调用 LLM 判别式连边。

        和 mem0 不同: mem0 每次 add() 都调 LLM (CHUNK_SIZE=1 → 419次调用)
        MAG: 积累所有句子，最后统一 batch 调用 (batch_size=15 → ~28次调用)

        Args:
            batch_size: 每批处理的句子数 (None = 使用初始配置的 mag_relation_batch_size)

        Returns:
            {"sentences": N, "relations_added": M, "batches": K}

        Ablation:
            batch_size=0 → 跳过，图只有实体没有边
            flush_relations() 不调用 → 图只有实体没有边
        """
        if batch_size is None:
            batch_size = self.mag_relation_batch_size

        # 积累模式：把 _pending_edge_sentences 也合并进来处理
        if self.mag_edge_entity_threshold > 0 and self._pending_edge_sentences:
            self._flush_edge_sentences()

        pending = self._pending_relations
        self._pending_relations = []

        if not pending or batch_size <= 0:
            logger.info("flush_relations: skipped (pending=%d, batch_size=%d)", len(pending), batch_size)
            return {"sentences": len(pending), "relations_added": 0, "batches": 0}

        total_relations = 0
        batch_count = 0

        for chunk_start in range(0, len(pending), batch_size):
            chunk = pending[chunk_start:chunk_start + batch_size]
            # chunk: [(sid, text), ...] — LLM 读全文上下文抽关系

            try:
                if self.mag_llm_segment:
                    result = self.relation_detector.segment_and_relate(
                        chunk, carry_over=self._llm_segment_carry)
                    self._llm_segment_carry = result.get("carry_over", [])
                    try:
                        edge_triples = self.relation_detector.extract_from_context(chunk)
                        for t in edge_triples:
                            try:
                                self.graph_store.add_relation(
                                    t.head, t.relation, t.tail, t.source_sentence_id, t.confidence,
                                    session_scope=self._mag_session_scope_for_sentence(t.source_sentence_id),
                                )
                            except Exception: pass
                    except Exception as e: logger.warning("S5 relation failed: %s", str(e)[:200])
                    self._store_atomic_facts(
                        result.get("facts", []), result.get("merged_away", []), chunk)
                else:
                    if self.mag_coref_resolve:
                        try:
                            resolved = self.relation_detector.resolve_coreferences(chunk)
                            for sid, rtext in resolved.items():
                                try:
                                    self.vector_store.update(
                                        vector_id=sid, vector=None, payload={"data": rtext})
                                except Exception: pass
                        except Exception as e:
                            logger.debug("Coref: %s", str(e)[:100])
                    triples = self.relation_detector.extract_from_context(chunk)
                if not self.mag_llm_segment:
                    for t in triples:
                        try:
                            self.graph_store.add_relation(
                                t.head, t.relation, t.tail, t.source_sentence_id, t.confidence,
                                session_scope=self._mag_session_scope_for_sentence(t.source_sentence_id),
                            )
                            total_relations += 1
                        except Exception:
                            pass
                batch_count += 1
            except Exception as e:
                logger.warning("flush_relations batch %d failed: %s", batch_count, str(e)[:200])

        gs = self.graph_store.stats()
        logger.info("flush_relations: %d sentences → %d relations (%d batches, graph: %d edges)",
                     len(pending), total_relations, batch_count, gs["num_relations"])
        gs = self.graph_store.stats()
        logger.info("flush_relations: %d relations (%d batches, graph: %d edges)",
                     total_relations, batch_count, gs["num_relations"])

        # 持久化图到磁盘
        self._graph_save()

        # 冲刷残留的 carry_over（最后一批的上下文）
        if self.mag_llm_segment and self._llm_segment_carry:
            try:
                # 用carry_over内容做最后一次消解（不建图，只更新文本）
                for sid, text in self._llm_segment_carry:
                    try:
                        self.vector_store.update(
                            vector_id=sid, vector=None, payload={"data": text})
                    except Exception: pass
            except Exception: pass
            self._llm_segment_carry = []

        return {"sentences": len(pending), "relations_added": total_relations,
                "batches": batch_count}

    def _store_atomic_facts(self, facts: List[str], merged_away: List[str],
                            original_items: List[Tuple[str, str]]):
        """存储 LLM 生成的原子事实  (alias, returns None for compat)"""
        return self._store_atomic_facts_return_ids(facts, merged_away, original_items)

    def _store_atomic_facts_return_ids(self, facts: List[str], merged_away: List[str],
                                        original_items: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """存储 LLM 原子事实，返回 [(fact_id, fact_text), ...] 供后续建边"""
        if not facts:
            return []
        # 从原句中继承时间戳和过滤字段（遍历所有句找到非空值）
        first_sid, last_sid = original_items[0][0], original_items[-1][0]
        ts = ""
        inherited_fields = {}
        prev_link = next_link = ""
        for sid, _ in original_items:
            try:
                rec = self.vector_store.get(vector_id=sid)
                if rec and hasattr(rec, "payload"):
                    p = rec.payload
                    if not prev_link:
                        prev_link = p.get("prev_sentence_id", "")
                    if not inherited_fields.get("user_id"):
                        for k in ("user_id", "agent_id", "run_id"):
                            if k in p and p[k]:
                                inherited_fields[k] = p[k]
                    if not ts:
                        orig_ts = p.get("created_at", "")
                        if orig_ts: ts = orig_ts
                    if inherited_fields.get("user_id") and prev_link and ts:
                        break  # 都找到了
            except Exception: pass
        if not ts:
            ts = datetime.now(timezone.utc).isoformat()
        try:
            rec = self.vector_store.get(vector_id=last_sid)
            if rec and hasattr(rec, "payload"):
                p = rec.payload
                if not next_link:
                    next_link = p.get("next_sentence_id", "")
        except Exception: pass

        # 存 facts，建立 prev/next 链
        import hashlib
        fact_ids = []
        fact_result = []  # [(fid, fact_text), ...]
        for fact_text in facts:
            fid = str(uuid.uuid4())
            fact_ids.append(fid)
            fact_result.append((fid, fact_text))
            # 提取实体
            entities_raw = self.entity_extractor.extract(fact_text)
            entities_payload = [{"name": e[0], "attention_weight": 0.5, "entity_type": e[1]}
                               for e in entities_raw]
            fact_emb = self.embedding_model.embed(fact_text, "add")
            # 提取 speaker（"Caroline: ..." → speaker="Caroline"）
            speaker = ""
            if ": " in fact_text:
                speaker = fact_text.split(": ", 1)[0].strip()
            payload = {
                "data": fact_text,
                "entities": entities_payload,
                "speaker": speaker,
                "sentence_id": fid,
                "created_at": ts,
                "updated_at": ts,
                "hash": hashlib.md5(fact_text.encode()).hexdigest(),
                "text_lemmatized": lemmatize_for_bm25(fact_text),
                **inherited_fields,  # user_id, agent_id, run_id
            }
            self.vector_store.insert(vectors=[fact_emb], ids=[fid], payloads=[payload])
            self._mag_sentence_scopes[fid] = _build_session_scope(inherited_fields)

        # 建立 prev/next 链接
        if fact_ids:
            # 第一个 fact → 原链 prev
            if prev_link:
                try:
                    self.vector_store.update(vector_id=fact_ids[0], vector=None,
                                             payload={"prev_sentence_id": prev_link})
                    self.vector_store.update(vector_id=prev_link, vector=None,
                                             payload={"next_sentence_id": fact_ids[0]})
                except Exception: pass
            # 最后一个 fact → 原链 next
            if next_link:
                try:
                    self.vector_store.update(vector_id=fact_ids[-1], vector=None,
                                             payload={"next_sentence_id": next_link})
                    self.vector_store.update(vector_id=next_link, vector=None,
                                             payload={"prev_sentence_id": fact_ids[-1]})
                except Exception: pass
            # facts 之间互相链接
            for i in range(len(fact_ids) - 1):
                try:
                    self.vector_store.update(vector_id=fact_ids[i], vector=None,
                                             payload={"next_sentence_id": fact_ids[i+1]})
                    self.vector_store.update(vector_id=fact_ids[i+1], vector=None,
                                             payload={"prev_sentence_id": fact_ids[i]})
                except Exception: pass

        # 被合并的旧句保留原向量，不做任何修改（BFS 需要原向量计算语义相似度）
        # merged_away 句子的 UUID 仍在边的 source_sentence_ids 中，必须保留完整数据
        return fact_result

    def _flush_edge_sentences(self):
        """批量处理积累的边候选句 — 实体数够阈值时触发。"""
        if not self._pending_edge_sentences or len(self._pending_edge_sentences) < 2:
            # 不足 2 句时不处理，但保留已积累的句子继续等
            return
        pending = self._pending_edge_sentences
        self._pending_edge_sentences = []
        self._pending_edge_entity_count = 0
        try:
            if self.mag_llm_segment:
                # LLM 联合: segment→facts + 消解
                # 收集时间戳
                ts_map = {}
                for sid, _ in pending:
                    try:
                        rec = self.vector_store.get(vector_id=sid)
                        if rec and hasattr(rec, "payload"):
                            t = rec.payload.get("created_at", "")[:10]
                            if t: ts_map[sid] = t
                    except Exception: pass
                result = self.relation_detector.segment_and_relate(
                    pending, carry_over=self._llm_segment_carry, timestamps=ts_map)
                self._llm_segment_carry = result.get("carry_over", [])
                # 先存储原子事实（生成新 fact UUID），再用新 fact ID 建边
                fact_ids = self._store_atomic_facts_return_ids(
                    result.get("facts", []), result.get("merged_away", []), pending)
                # 用新 fact 的 (id, text) 建边，保证边的 source_sentence_ids 指向有效向量
                try:
                    new_items = []
                    for fid, ftxt in fact_ids:
                        new_items.append((fid, ftxt))
                    if new_items:
                        edge_triples = self.relation_detector.extract_from_context(new_items)
                        logger.warning("S5: got %d edge_triples, graph before=%d edges",
                                       len(edge_triples), self.graph_store.stats()['num_relations'])
                        for t in edge_triples:
                            try:
                                self.graph_store.add_relation(t.head, t.relation, t.tail,
                                                              t.source_sentence_id, t.confidence,
                                                              session_scope=self._mag_session_scope_for_sentence(t.source_sentence_id))
                            except Exception as e2:
                                logger.warning("add_relation: %s", str(e2)[:100])
                        logger.warning("S5: after add, graph=%d edges",
                                       self.graph_store.stats()['num_relations'])
                except Exception as e:
                    logger.warning("S5 relation failed: %s", str(e)[:200])
            else:
                if self.mag_coref_resolve:
                    try:
                        resolved = self.relation_detector.resolve_coreferences(pending)
                        for sid, rtext in resolved.items():
                            try:
                                self.vector_store.update(
                                    vector_id=sid, vector=None, payload={"data": rtext})
                            except Exception: pass
                    except Exception as e:
                        logger.debug("Coref: %s", str(e)[:100])
                triples = self.relation_detector.extract_from_context(pending)
                for t in triples:
                    try:
                        self.graph_store.add_relation(t.head, t.relation, t.tail,
                                                      t.source_sentence_id, t.confidence,
                                                      session_scope=self._mag_session_scope_for_sentence(t.source_sentence_id))
                    except Exception:
                        pass
        except Exception as e:
            logger.warning("_flush_edge_sentences: %s", str(e)[:200])
        self._graph_save()

    def flush_edge_sentences(self):
        """公开接口：强制 flush 积累的边候选句（session 边界等场景调用）。"""
        self._flush_edge_sentences()

    def _mag_session_scope_for_sentence(
        self,
        sentence_id: str,
        default_scope: Optional[str] = None,
    ) -> str:
        """Resolve a sentence's graph session scope from memory or vector payload."""
        sid = str(sentence_id or "")
        if not sid:
            return default_scope or ""
        if sid in self._mag_sentence_scopes:
            return self._mag_sentence_scopes[sid]
        try:
            rec = self.vector_store.get(vector_id=sid)
            if rec and hasattr(rec, "payload") and isinstance(rec.payload, dict):
                scope = _build_session_scope(rec.payload)
                self._mag_sentence_scopes[sid] = scope
                return scope
        except Exception:
            pass
        return default_scope or ""

    def _graph_save(self):
        """持久化图到 JSON 文件 (线程安全: 先拷贝再原子替换)"""
        import json as _json
        g = self.graph_store.graph
        with self._graph_save_lock:
            try:
                entities = {n: dict(g.nodes[n]) for n in list(g.nodes)}
                edges = [
                    {"u": u, "v": v, "key": k, "data": dict(d)}
                    for u, v, k, d in list(g.edges(keys=True, data=True))
                ]
            except RuntimeError:
                return  # 图正在被修改，跳过本次保存
            data = {"entities": entities, "edges": edges}
            graph_dir = os.path.dirname(os.path.abspath(self._graph_path)) or "."
            os.makedirs(graph_dir, exist_ok=True)
            fd, tmp_path = tempfile.mkstemp(
                prefix=f".{os.path.basename(self._graph_path)}.",
                suffix=".tmp",
                dir=graph_dir,
            )
            try:
                with os.fdopen(fd, "w") as f:
                    _json.dump(data, f, default=str)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp_path, self._graph_path)
            except Exception:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
        logger.warning("Graph saved to %s (%d entities, %d edges)",
                       self._graph_path, len(entities), len(edges))


    def _graph_load(self):
        """从 JSON 文件恢复图。如果文件缺失或 UUID 过期，从空图开始"""
        import json as _json
        if not os.path.exists(self._graph_path):
            logger.warning("Graph file not found, starting fresh: %s", self._graph_path)
            return
        try:
            with open(self._graph_path) as f:
                data = _json.load(f)
            g = self.graph_store.graph
            for name, attrs in data.get("entities", {}).items():
                g.add_node(name, **attrs)
            for e in data.get("edges", []):
                g.add_edge(e["u"], e["v"], key=e.get("key"), **e.get("data", {}))
            # 重建反向索引
            for n in g.nodes:
                scopes = g.nodes[n].get("linked_sentence_scopes", {})
                for sid in g.nodes[n].get("linked_sentence_ids", []):
                    self.graph_store._entity_index[n].add(sid)
                    if n not in self.graph_store._sentence_entities[sid]:
                        self.graph_store._sentence_entities[sid].append(n)
                    if sid in scopes:
                        self._mag_sentence_scopes[sid] = scopes.get(sid, "")
            # 从 Qdrant 重建 _seen_hashes（去重）
            try:
                all_items = self.vector_store.list(filters={}, top_k=100000)
                rows = all_items[0] if isinstance(all_items, (list, tuple)) and all_items else all_items
                for r in (rows or []):
                    h = r.payload.get("hash") if hasattr(r, "payload") else None
                    if h:
                        self._seen_hashes.add(h)
            except Exception:
                pass

            gs = self.graph_store.stats()
            logger.info("Graph loaded from %s (%d entities, %d edges, %d hashes)",
                         self._graph_path, gs["num_entities"], gs["num_relations"],
                         len(self._seen_hashes))
        except FileNotFoundError:
            logger.debug("No persisted graph at %s", self._graph_path)
        except Exception as e:
            logger.warning("Graph load failed: %s", str(e))

    # ==================================================================
    # MAG 增强检索 (BFS + LinearRAG + 时间衰减 + 上下文)
    # ==================================================================

    def _mag_payload_matches_filters(self, payload: Dict[str, Any], filters: Optional[Dict[str, Any]]) -> bool:
        if not payload:
            return False
        for key in ("user_id", "agent_id", "run_id"):
            expected = (filters or {}).get(key)
            if expected is not None and payload.get(key) != expected:
                return False
        return True

    def _mag_get_scoped_payload(self, sentence_id: str, filters: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        try:
            rec = self.vector_store.get(vector_id=sentence_id)
            if not rec or not hasattr(rec, "payload"):
                return None
            payload = rec.payload
            if not isinstance(payload, dict):
                return None
            if not self._mag_payload_matches_filters(payload, filters):
                return None
            return payload
        except Exception:
            return None

    def _mag_search(self, query: str, filters, limit: int) -> Dict[str, Any]:
        """MAG retrieval with vector/BM25 as the primary recall route.

        Graph BFS is intentionally conservative: it can reinforce direct
        vector/BM25 hits or add a small number of supplemental candidates, but
        it must not crowd out the validator route that carries the strongest
        lexical/semantic evidence in LOCOMO.
        """
        from mem0.utils.entity_extraction import extract_entities as _ext_ent

        filters = filters or {}
        session_scope = _build_session_scope(filters)
        graph_pool_limit = min(300, max(limit * 10, 100))
        validator_limit = max(limit * 4, 60)
        query_embedding = None
        query_entities = []
        qews: List[EntityWeight] = []
        route_debug = {
            "graph_paths": 0,
            "graph_candidates": 0,
            "validator_candidates": 0,
            "validator_pool_added": 0,
            "vector_fallback_added": 0,
            "context_support_attached": 0,
            "evidence_quality_adjusted": 0,
            "diversity_pool_size": 0,
        }

        try:
            query_entities = _ext_ent(query)
        except Exception:
            query_entities = []
        for e in query_entities:
            ename = e[1].strip().lower()
            if not ename:
                continue
            qews.append(EntityWeight(name=ename, attention_weight=0.6, entity_type=e[0]))
            if " " in ename:
                for word in ename.split():
                    if len(word) > 2:
                        qews.append(EntityWeight(name=word, attention_weight=0.4, entity_type="TOKEN"))

        # ── Primary validator route: vector+BM25 seeds the pool. BFS may
        # supplement or reinforce it, but not replace these direct candidates.
        try:
            raw_results = self._search_vector_store(
                query,
                filters,
                validator_limit,
                threshold=0.1,
            )
        except Exception:
            raw_results = []

        validator_map: Dict[str, Dict[str, Any]] = {}
        for rank, r in enumerate(raw_results):
            rid = str(r.get("id", ""))
            if rid:
                validator_map[rid] = {
                    "id": rid,
                    "memory": r.get("memory", ""),
                    "score": r.get("score", 0),
                    "rank": rank + 1,
                    "created_at": r.get("created_at", ""),
                    "entities": r.get("metadata", {}).get("entities", []) if isinstance(r.get("metadata"), dict) else [],
                    "metadata": r.get("metadata", {}) if isinstance(r.get("metadata"), dict) else {},
                }
        route_debug["validator_candidates"] = len(validator_map)

        # ── Supplemental route: graph BFS candidate generation.
        graph_list: List[Dict[str, Any]] = []
        if self.mag_use_bfs and qews:
            try:
                def _sem_sim(identifier):
                    nonlocal query_embedding
                    try:
                        sid_text = str(identifier)
                        is_uuid_like = len(sid_text) >= 32 and "-" in sid_text
                        if not is_uuid_like:
                            if query_embedding is None:
                                query_embedding = self.embedding_model.embed(query, "search")
                            triple_embedding = self.embedding_model.embed(sid_text, "search")
                            import numpy as np
                            qv = np.array(query_embedding)
                            tv = np.array(triple_embedding)
                            sim = float(np.dot(qv, tv) / (np.linalg.norm(qv) * np.linalg.norm(tv) + 1e-8))
                            return max(0.0, min(1.0, sim))

                        # Qdrant server mode: vector_store.get() does not include vectors.
                        if hasattr(self.vector_store, "client"):
                            pts = self.vector_store.client.retrieve(
                                collection_name=self.collection_name,
                                ids=[sid_text],
                                with_payload=True,
                                with_vectors=True,
                            )
                            if pts and pts[0].vector:
                                payload = getattr(pts[0], "payload", None) or {}
                                if not self._mag_payload_matches_filters(payload, filters):
                                    return 0.0
                                sv = pts[0].vector
                                if isinstance(sv, dict):
                                    sv = sv.get("", list(sv.values())[0] if sv else [])
                                if query_embedding is None:
                                    query_embedding = self.embedding_model.embed(query, "search")
                                import numpy as np
                                qv = np.array(query_embedding)
                                sv_arr = np.array(sv)
                                sim = float(np.dot(qv, sv_arr) / (np.linalg.norm(qv) * np.linalg.norm(sv_arr) + 1e-8))
                                return max(0.0, min(1.0, sim))
                    except Exception:
                        pass
                    return 0.5

                bfs_paths = self.bfs_retriever.search_paths(
                    qews,
                    query_embedding=query_embedding,
                    get_semantic_sim=_sem_sim,
                    max_hops=3,
                    tolerance=2,
                    sim_threshold=0.25,
                    max_results=graph_pool_limit,
                    session_scope=session_scope,
                )
                route_debug["graph_paths"] = len(bfs_paths)

                for path_info in bfs_paths:
                    scoped_payloads = []
                    for sid in path_info.get("sentences", []):
                        payload = self._mag_get_scoped_payload(sid, filters)
                        if payload is None:
                            scoped_payloads = []
                            break
                        scoped_payloads.append((str(sid), payload))
                    if not scoped_payloads:
                        continue

                    path_len = len(scoped_payloads)
                    length_bonus = 0.03 * max(0, path_len - 1)
                    graph_score = min(1.0, path_info.get("path_score", 0.0) + length_bonus)
                    candidate_id = scoped_payloads[-1][0]
                    validator = validator_map.get(candidate_id)
                    validator_score = validator.get("score", 0.0) if validator else 0.0
                    if validator:
                        # Validated graph paths are useful as a weak boost, but
                        # the original vector/BM25 memory stays authoritative.
                        final_score = validator_score + (0.12 * graph_score)
                    else:
                        # Unvalidated graph-only paths are noisy in LOCOMO. Let
                        # them compete as backfill, not as primary evidence.
                        final_score = 0.25 * graph_score

                    ctx_lines, path_ts = [], ""
                    all_entities = []
                    for _, p in scoped_payloads:
                        txt = p.get("data", "")
                        ts = p.get("created_at", "")
                        ents = p.get("entities", [])
                        if txt:
                            ctx_lines.append(f"[{ts[:10]}] {txt}" if ts else txt)
                            if not path_ts:
                                path_ts = ts
                            all_entities.extend(ents if isinstance(ents, list) else [])
                    if not ctx_lines:
                        continue

                    triple_strs = [f"({h} {r} {t})" for h, r, t in path_info.get("triples", [])]
                    graph_memory = "\n".join(ctx_lines)
                    if triple_strs:
                        graph_memory += "\n[triples: " + "; ".join(triple_strs) + "]"
                    memory = validator.get("memory", "") if validator else graph_memory

                    graph_list.append({
                        "id": candidate_id,
                        "memory": memory,
                        "score": final_score,
                        "created_at": path_ts,
                        "source": "graph_bfs+validator" if validator else "graph_bfs",
                        "entities": all_entities,
                        "route_scores": {
                            "graph": round(graph_score, 4),
                            "validator": round(validator_score, 4),
                            "final_pre_rerank": round(final_score, 4),
                            "path_len": path_len,
                        },
                        "graph_path": path_info.get("path", []),
                    })
                    if validator:
                        graph_list[-1]["supporting_graph_context"] = graph_memory
            except Exception as e:
                logger.debug("BFS: %s", str(e)[:100])

        route_debug["graph_candidates"] = len(graph_list)

        # ── Merge validator-primary candidates first. Keep their native score
        # and text; graph hits can only reinforce them.
        merged_map: Dict[str, Dict] = {}
        validator_pool_limit = max(limit * 4, 60)
        for rid, v in list(validator_map.items())[:validator_pool_limit]:
            validator_score = v.get("score", 0)
            merged_map[rid] = {
                "id": rid,
                "memory": v.get("memory", ""),
                "score": validator_score,
                "created_at": v.get("created_at", ""),
                "source": "vector+bm25_validator",
                "entities": v.get("entities", []),
                "route_scores": {
                    "graph": 0.0,
                    "validator": round(validator_score, 4),
                    "final_pre_rerank": round(validator_score, 4),
                    "validator_rank": v.get("rank"),
                },
            }
            route_debug["validator_pool_added"] += 1

        graph_only_added = 0
        graph_only_limit = max(2, limit // 5)
        for r in graph_list:
            rid = r["id"]
            if rid in merged_map:
                graph_scores = r.get("route_scores", {})
                route_scores = merged_map[rid].setdefault("route_scores", {})
                route_scores["graph"] = max(route_scores.get("graph", 0), graph_scores.get("graph", 0))
                route_scores["graph_path_len"] = graph_scores.get("path_len", 0)
                route_scores["graph_reinforced"] = 1.0
                merged_map[rid]["source"] = _mag_add_source(merged_map[rid].get("source", ""), "graph_bfs")
                merged_map[rid]["score"] = max(merged_map[rid]["score"], r["score"])
                if r.get("supporting_graph_context"):
                    merged_map[rid]["supporting_graph_context"] = r["supporting_graph_context"]
            else:
                if graph_only_added >= graph_only_limit:
                    continue
                merged_map[rid] = r
                graph_only_added += 1

        # Fallback keeps recall when query entities miss the graph or graph pool is too small.
        fallback_needed = max(0, limit - len(merged_map))
        if fallback_needed:
            for rid, v in validator_map.items():
                if rid in merged_map:
                    continue
                merged_map[rid] = {
                    "id": rid,
                    "memory": v.get("memory", ""),
                    "score": v.get("score", 0) * 0.55,
                    "created_at": v.get("created_at", ""),
                    "source": "vector+bm25_fallback",
                    "entities": v.get("entities", []),
                    "route_scores": {
                        "graph": 0.0,
                        "validator": round(v.get("score", 0), 4),
                        "final_pre_rerank": round(v.get("score", 0) * 0.55, 4),
                        "validator_rank": v.get("rank"),
                    },
                }
                route_debug["vector_fallback_added"] += 1
                if route_debug["vector_fallback_added"] >= fallback_needed:
                    break

        merged_list = sorted(merged_map.values(), key=lambda x: x["score"], reverse=True)

        # ── Entity Match Boost: 匹配 query 实体越多的句子加分越多 ──
        if self.mag_use_entity_match and merged_list:
            from mem0.utils.entity_extraction import extract_entities as _ext_qe
            q_ents = _ext_qe(query)
            if q_ents:
                q_names = {e[1].strip().lower() for e in q_ents if e[1].strip()}
                for r in merged_list:
                    sent_ents = r.get("entities", [])
                    if not sent_ents:
                        continue
                    sent_names = set()
                    for e in sent_ents:
                        if isinstance(e, dict):
                            sent_names.add(e.get("name", "").strip().lower())
                    matches = len(q_names & sent_names)
                    if matches > 0:
                        # 重合率 × 乘性加分: score × (1 + ratio × 0.1)
                        match_ratio = matches / max(len(q_names), 1)
                        boost = r.get("score", 0) * match_ratio * 0.1
                        r["entity_match_boost"] = round(boost, 4)
                        r["entity_match_count"] = matches
                        r["score"] = r.get("score", 0) + boost
                merged_list.sort(key=lambda x: x["score"], reverse=True)

        # ── Evidence quality features: favor direct lexical/date support and
        # mildly penalize broad path contexts before cross-encoder rerank.
        for r in merged_list:
            features = _mag_candidate_evidence_features(query, r)
            route_scores = r.setdefault("route_scores", {})
            route_scores.update(features)
            base_score = r.get("score", 0.0)
            adjustment = (
                base_score * 0.10 * features["evidence_score"]
                - base_score * features["length_penalty"]
            )
            if adjustment:
                r["score"] = max(0.0, base_score + adjustment)
                route_scores["evidence_adjustment"] = round(adjustment, 4)
                route_debug["evidence_quality_adjusted"] += 1
        merged_list.sort(key=lambda x: x["score"], reverse=True)

        # ── Rerank (CrossEncoder, 可 ablation: mag_use_rerank=False) ──
        if self.mag_use_rerank and self._reranker is not None and merged_list:
            rerank_pool_size = max(limit * 3, 60)
            protected_validator_ids = set(list(validator_map.keys())[:min(len(validator_map), rerank_pool_size)])
            protected = [r for r in merged_list if r.get("id") in protected_validator_ids]
            rest = [r for r in merged_list if r.get("id") not in protected_validator_ids]
            rerank_pool = (protected + rest)[:rerank_pool_size]
            pairs = [(query, r.get("memory", "")) for r in rerank_pool if r.get("memory")]
            if pairs:
                try:
                    scores = self._reranker.predict(pairs)
                    valid = [r for r in rerank_pool if r.get("memory")]
                    import math as _math
                    for i, (r, rr) in enumerate(zip(valid, scores)):
                        rr_norm = 1.0 / (1.0 + _math.exp(-float(rr)))
                        r["rerank_score"] = rr_norm
                        source = r.get("source", "")
                        if source == "graph_bfs":
                            r["score"] = 0.75 * r.get("score", 0) + 0.25 * rr_norm
                        elif "vector+bm25" in source:
                            r["score"] = 0.45 * r.get("score", 0) + 0.55 * rr_norm
                        else:
                            r["score"] = 0.55 * r.get("score", 0) + 0.45 * rr_norm
                except Exception as e:
                    logger.debug("Rerank: %s", str(e)[:100])
            merged_list.sort(key=lambda x: x["score"], reverse=True)

        # ── LinearRAG ──
        avg_score = sum(c.get("score", 0) for c in merged_list[:limit]) / max(len(merged_list[:limit]), 1)
        if self.mag_use_linear_rag and (avg_score < self.mag_quality_threshold or len(merged_list) < limit // 2):
            try:
                supplement = self._mag_linear_rag(query, filters, merged_list, limit)
                merged_list = self._mag_dedup(merged_list + supplement)
            except Exception as e:
                logger.warning("LinearRAG: %s", str(e)[:100])

        merged_list.sort(key=lambda x: x["score"], reverse=True)

        # ── 上下文窗口: prev/next 作为父候选的 supporting text。
        # Do not add neighbor sentences as independent high-score candidates;
        # they are evidence support unless they already passed graph/vector routes.
        if self.mag_use_context_window and merged_list:
            seen_ids = {r.get("id", "") for r in merged_list}
            for r in merged_list:
                pid = r.get("id", "")
                if not pid:
                    continue
                try:
                    rec = self.vector_store.get(vector_id=pid)
                    if not rec or not hasattr(rec, "payload"):
                        continue
                    p = rec.payload
                    base_score = r.get("score", 0) * 0.95  # 邻居句轻微降权
                    for direction in ("prev_sentence_id", "next_sentence_id"):
                        neighbor_id = p.get(direction, "")
                        if not neighbor_id:
                            continue
                        # 邻居已在池中 → 双向提分
                        if neighbor_id in merged_map:
                            neighbor_payload = self._mag_get_scoped_payload(neighbor_id, filters)
                            if neighbor_payload is not None:
                                merged_map[neighbor_id]["score"] = max(
                                    merged_map[neighbor_id]["score"], base_score)
                                merged_map[neighbor_id]["source"] = _mag_add_source(
                                    merged_map[neighbor_id].get("source", ""),
                                    "ctx_boost",
                                )
                            seen_ids.add(neighbor_id)
                            continue
                        if neighbor_id in seen_ids:
                            continue
                        try:
                            nb_p = self._mag_get_scoped_payload(neighbor_id, filters)
                            if nb_p is not None:
                                support = r.setdefault("supporting_context", [])
                                support_text = nb_p.get("data", "")
                                support_ts = nb_p.get("created_at", "")
                                if support_text:
                                    support.append({
                                        "id": neighbor_id,
                                        "direction": direction.replace("_sentence_id", ""),
                                        "memory": support_text,
                                        "created_at": support_ts,
                                    })
                                    label = "previous" if direction == "prev_sentence_id" else "next"
                                    ts_prefix = f"[{support_ts[:10]}] " if support_ts else ""
                                    r["memory"] = (
                                        f"{r.get('memory', '')}\n"
                                        f"[supporting {label}: {ts_prefix}{support_text}]"
                                    )
                                    r.setdefault("route_scores", {})["context_support_count"] = len(support)
                                    route_debug["context_support_attached"] += 1
                                seen_ids.add(neighbor_id)
                        except Exception:
                            pass
                except Exception:
                    pass
            merged_list.sort(key=lambda x: x["score"], reverse=True)

        diversity_pool_size = min(len(merged_list), max(limit * 4, 60))
        route_debug["diversity_pool_size"] = diversity_pool_size
        final = _mag_diverse_topk(merged_list, limit, diversity_pool_size)

        # ── 上下文组装 ──
        ctx_lines = ["[Relevant Past Memories]\n"]
        cnt = 0
        for i, r in enumerate(final):
            mem = r.get("memory", "")
            if not mem: continue
            sid = r.get("id", "?")[:8]
            ts = r.get("created_at", "?")
            ents = r.get("entities", [])
            ents_str = ", ".join(f"{e['name']}({e['attention_weight']:.1f})" for e in sorted(ents, key=lambda x: x.get('attention_weight',0), reverse=True)[:5] if isinstance(e, dict)) if ents else ""
            line = f"{i + 1}. [{sid}] [{ts}]"
            if ents_str: line += f" [{ents_str}]"
            line += f" {mem}"
            cnt += len(line) // 3
            if cnt > 2000: break
            ctx_lines.append(line)
        ctx_lines.append(f"\n[Current Query]\n{query}")

        route_composition: Dict[str, int] = {}
        for r in final:
            source = r.get("source", "unknown")
            route_composition[source] = route_composition.get(source, 0) + 1
        route_debug["route_composition"] = route_composition

        for r in final:
            metadata = r.setdefault("metadata", {})
            if r.get("route_scores"):
                metadata["route_scores"] = r["route_scores"]
            if r.get("graph_path"):
                metadata["graph_path"] = r["graph_path"]
            if r.get("supporting_context"):
                metadata["supporting_context"] = r["supporting_context"]
            if r.get("supporting_graph_context"):
                metadata["supporting_graph_context"] = r["supporting_graph_context"]

        return {"results": final, "context": "\n".join(ctx_lines), "debug": {
            "total": len(merged_list),
            "avg_r1": round(avg_score, 4),
            "routes": route_debug,
            "graph": self.graph_store.stats(),
        }}


    def _mag_merge_mem0_bfs(self, mem0_results: List, query: str, filters, limit: int) -> List[Dict]:
        from mem0.utils.entity_extraction import extract_entities as _ext_ent
        m: Dict[str, Dict] = {}
        session_scope = _build_session_scope(filters or {})
        for r in mem0_results:
            rid = str(r.get("id", ""))
            if rid:
                m[rid] = {"id": rid, "memory": r.get("memory", ""), "score": r.get("score", 0),
                          "mem0_score": r.get("score", 0), "source": "mem0",
                          "created_at": r.get("created_at", ""), "payload": {}}
        try:
            qe = _ext_ent(query)
            if qe:
                qews = []
                for e in qe:
                    ename = e[1].strip().lower()
                    qews.append(EntityWeight(name=ename, attention_weight=0.6, entity_type=e[0]))
                    # 组合实体也拆分为单词加入 (解决 compound entity 无法匹配图节点的问题)
                    if ' ' in ename:
                        for word in ename.split():
                            if len(word) > 2:
                                qews.append(EntityWeight(name=word, attention_weight=0.4, entity_type='TOKEN'))
                for sid, bfs_s in self.bfs_retriever.search(
                    qews,
                    max_hops=2,
                    max_results=limit * 2,
                    session_scope=session_scope,
                ):
                    if sid in m:
                        m[sid]["score"] = 0.7 * m[sid]["mem0_score"] + 0.3 * bfs_s
                        m[sid]["source"] = _mag_add_source(m[sid].get("source", ""), "graph_bfs")
                    else:
                        p = {}; txt = ""; ts = ""
                        try:
                            result = self.vector_store.client.retrieve(
                                collection_name=self.vector_store.collection_name,
                                ids=[sid],
                                with_payload=True,
                            )
                            if result and len(result) > 0:
                                p = dict(result[0].payload) if hasattr(result[0], "payload") else {}
                                txt = p.get("data", "") or ""
                                ts = p.get("created_at", "")
                        except Exception:
                            pass
                        m[sid] = {"id": sid, "memory": txt, "score": bfs_s * 0.5, "mem0_score": 0,
                                  "source": "graph_bfs", "created_at": ts, "payload": p}
        except Exception as e:
            logger.debug("BFS: %s", str(e)[:100])
        return sorted(m.values(), key=lambda x: x["score"], reverse=True)

    def _mag_linear_rag(self, query, filters, first_results: List, top_k: int) -> List[Dict]:
        ids: Set[str] = set()
        for level in range(1, self.mag_max_supplement_rounds + 1):
            if level == 1:
                seeds = self._mag_seed_entities(first_results)
                if seeds:
                    session_scope = _build_session_scope(filters or {})
                    for ent in self.bfs_retriever.expand_entities(
                        seeds,
                        max_hops=1,
                        session_scope=session_scope,
                    ):
                        for sid in self.graph_store.get_sentences_for_entity(
                            ent,
                            session_scope=session_scope,
                        )[:5]:
                            ids.add(sid)
                if len(ids) >= top_k // 2:
                    break
            elif level == 2:
                ref = self._mag_reformulate(query, first_results)
                if ref and ref != query:
                    try:
                        rr = self.search(ref, top_k=top_k, filters=filters)
                        for r in (rr.get("results", []) if isinstance(rr, dict) else []):
                            ids.add(str(r.get("id", "")))
                    except Exception:
                        pass
                if len(ids) >= top_k // 4:
                    break
            elif level >= 3:
                if not first_results:
                    try:
                        all_m = self.vector_store.list(filters=filters, top_k=100)
                        rows = (all_m[0] if isinstance(all_m, (list, tuple)) and all_m and isinstance(all_m[0], list) else all_m)
                        for row in (rows or [])[:top_k]:
                            ids.add(str(row.id))
                    except Exception:
                        pass
                break
        return self._mag_fetch(list(ids), filters)

    @staticmethod
    def _mag_seed_entities(results: List, min_w: float = 0.5) -> List[str]:
        seeds: Set[str] = set()
        for r in results[:5]:
            for src in ["entities", "payload"]:
                ents = r.get(src, {}) if src == "payload" else r.get(src, [])
                if isinstance(ents, dict):
                    ents = ents.get("entities", [])
                if isinstance(ents, list):
                    for e in ents:
                        if isinstance(e, dict) and e.get("attention_weight", 0) >= min_w:
                            seeds.add(e["name"])
        return list(seeds)

    def _mag_reformulate(self, query: str, results: List) -> Optional[str]:
        if not self.llm:
            return None
        ctx = "\n".join(f"- {r.get('memory','')}" for r in results[:3] if r.get("memory"))
        if not ctx:
            ctx = "(无结果)"
        try:
            resp = self.llm.generate_response(messages=[{"role": "user", "content": (
                "原始查询未找到足够相关记忆。改写查询使更可能匹配。\n"
                f"原始: {query}\n已有结果:\n{ctx}\n改写后的查询:"
            )}])
            return resp.strip() if resp and len(resp.strip()) > 5 else None
        except Exception:
            return None

    def _mag_dedup(self, clist: List) -> List[Dict]:
        m: Dict[str, Dict] = {}
        for c in clist:
            cid = c.get("id", "")
            if not cid:
                continue
            if cid in m:
                m[cid]["score"] = max(m[cid]["score"], c.get("score", 0))
            else:
                m[cid] = dict(c)
        return sorted(m.values(), key=lambda x: x["score"], reverse=True)

    def _mag_fetch(self, sids: List[str], filters: Optional[Dict[str, Any]] = None) -> List[Dict]:
        results = []
        for sid in sids:
            p = self._mag_get_scoped_payload(sid, filters)
            if p is not None:
                results.append({"id": sid, "memory": p.get("data", ""), "score": 0.1,
                                "source": "linear_rag", "created_at": p.get("created_at", ""), "payload": p})
        return results


class AsyncMemory(MemoryBase):
    def __init__(self, config: MemoryConfig = MemoryConfig()):
        self.config = config

        self.embedding_model = EmbedderFactory.create(
            self.config.embedder.provider,
            self.config.embedder.config,
            self.config.vector_store.config,
        )
        self.vector_store = VectorStoreFactory.create(
            self.config.vector_store.provider, self.config.vector_store.config
        )
        self.llm = LlmFactory.create(self.config.llm.provider, self.config.llm.config)
        self.db = SQLiteManager(self.config.history_db_path)
        self.collection_name = self.config.vector_store.config.collection_name
        self.api_version = self.config.version
        self.custom_instructions = self.config.custom_instructions
        self._entity_store = None

        # Initialize reranker if configured
        self.reranker = None
        if config.reranker:
            self.reranker = RerankerFactory.create(
                config.reranker.provider,
                config.reranker.config
            )

        if MEM0_TELEMETRY:
            telemetry_config = _safe_deepcopy_config(self.config.vector_store.config)
            telemetry_config.collection_name = "mem0migrations"
            if self.config.vector_store.provider in ["faiss", "qdrant"]:
                provider_path = f"migrations_{self.config.vector_store.provider}"
                telemetry_config.path = os.path.join(mem0_dir, provider_path)
                os.makedirs(telemetry_config.path, exist_ok=True)
            self._telemetry_vector_store = VectorStoreFactory.create(self.config.vector_store.provider, telemetry_config)

        capture_event("mem0.init", self, {"sync_type": "async"})

    @property
    def entity_store(self):
        """Lazily initialize entity store on first use."""
        if self._entity_store is None:
            entity_config = _safe_deepcopy_config(self.config.vector_store.config)
            entity_collection = f"{self.collection_name}_entities"
            if hasattr(entity_config, 'collection_name'):
                entity_config.collection_name = entity_collection
            elif isinstance(entity_config, dict):
                entity_config['collection_name'] = entity_collection
            # For Qdrant, share the existing client to avoid RocksDB lock contention
            # when using embedded mode (path=...). QdrantConfig.client takes precedence
            # over host/port/path.
            if self.config.vector_store.provider == "qdrant" and hasattr(self.vector_store, "client"):
                try:
                    entity_config.client = self.vector_store.client
                except (AttributeError, TypeError):
                    if isinstance(entity_config, dict):
                        entity_config["client"] = self.vector_store.client
            self._entity_store = VectorStoreFactory.create(
                self.config.vector_store.provider, entity_config
            )
        return self._entity_store

    async def _upsert_entity_async(self, entity_text, entity_type, memory_id, filters):
        """Async variant of `_upsert_entity` — per-entity search-then-update-or-insert."""
        try:
            entity_embedding = await asyncio.to_thread(self.embedding_model.embed, entity_text, "add")
            search_filters = {k: v for k, v in filters.items() if k in ("user_id", "agent_id", "run_id") and v}

            existing = await asyncio.to_thread(
                self.entity_store.search,
                query=entity_text,
                vectors=entity_embedding,
                top_k=1,
                filters=search_filters,
            )

            if existing and existing[0].score >= 0.95:
                match = existing[0]
                payload = match.payload or {}
                linked_ids = payload.get("linked_memory_ids", [])
                if memory_id not in linked_ids:
                    linked_ids.append(memory_id)
                    payload["linked_memory_ids"] = linked_ids
                    await asyncio.to_thread(
                        self.entity_store.update,
                        vector_id=match.id,
                        vector=None,
                        payload=payload,
                    )
            else:
                entity_id = str(uuid.uuid4())
                entity_payload = {
                    "data": entity_text,
                    "entity_type": entity_type,
                    "linked_memory_ids": [memory_id],
                    **{k: v for k, v in search_filters.items()},
                }
                await asyncio.to_thread(
                    self.entity_store.insert,
                    vectors=[entity_embedding],
                    ids=[entity_id],
                    payloads=[entity_payload],
                )
        except Exception as e:
            logger.warning(f"Entity upsert failed for '{entity_text}' (async): {e}")

    async def _remove_memory_from_entity_store(self, memory_id, filters):
        """Async variant of `Memory._remove_memory_from_entity_store`."""
        if self._entity_store is None:
            return
        search_filters = {k: v for k, v in filters.items() if k in ("user_id", "agent_id", "run_id") and v}
        try:
            listed = await asyncio.to_thread(self.entity_store.list, filters=search_filters, top_k=10000)
            rows = listed[0] if isinstance(listed, (list, tuple)) and listed and isinstance(listed[0], list) else listed
            for row in rows or []:
                try:
                    payload = getattr(row, "payload", None) or {}
                    linked = payload.get("linked_memory_ids", [])
                    if not isinstance(linked, list) or memory_id not in linked:
                        continue
                    remaining = [mid for mid in linked if mid != memory_id]
                    if not remaining:
                        try:
                            await asyncio.to_thread(self.entity_store.delete, vector_id=row.id)
                        except Exception as e:
                            logger.debug(f"Entity delete failed for id={row.id} (async): {e}")
                    else:
                        entity_text = payload.get("data")
                        if not isinstance(entity_text, str) or not entity_text:
                            logger.debug(f"Entity id={row.id} missing 'data'; skipping update during cleanup (async)")
                            continue
                        try:
                            vec = await asyncio.to_thread(self.embedding_model.embed, entity_text, "update")
                        except Exception as e:
                            logger.debug(f"Entity re-embed failed for '{entity_text}' (async): {e}")
                            continue
                        new_payload = {**payload, "linked_memory_ids": remaining}
                        try:
                            await asyncio.to_thread(
                                self.entity_store.update,
                                vector_id=row.id,
                                vector=vec,
                                payload=new_payload,
                            )
                        except Exception as e:
                            logger.debug(f"Entity update failed for id={row.id} (async): {e}")
                except Exception as e:
                    logger.debug(f"Entity cleanup error (async): {e}")
        except Exception as e:
            logger.warning(f"Entity store cleanup failed for memory_id={memory_id} (async): {e}")

    async def _link_entities_for_memory(self, memory_id, text, filters):
        """Async variant of `Memory._link_entities_for_memory`."""
        try:
            entities = await asyncio.to_thread(extract_entities, text)
            if not entities:
                return
            seen = set()
            for entity_type, entity_text in entities:
                key = entity_text.strip().lower()
                if not key or key in seen:
                    continue
                seen.add(key)
                try:
                    await self._upsert_entity_async(entity_text, entity_type, memory_id, filters)
                except Exception as e:
                    logger.debug(f"Entity link failed for '{entity_text}' (async): {e}")
        except Exception as e:
            logger.warning(f"Entity linking failed for memory_id={memory_id} (async): {e}")

    @classmethod
    def from_config(cls, config_dict: Dict[str, Any]):
        try:
            config = cls._process_config(config_dict)
            config = MemoryConfig(**config_dict)
        except ValidationError as e:
            logger.error(f"Configuration validation error: {e}")
            raise
        return cls(config)

    @staticmethod
    def _process_config(config_dict: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return config_dict
        except ValidationError as e:
            logger.error(f"Configuration validation error: {e}")
            raise

    def _should_use_agent_memory_extraction(self, messages, metadata):
        """Determine whether to use agent memory extraction based on the logic:
        - If agent_id is present and messages contain assistant role -> True
        - Otherwise -> False

        Args:
            messages: List of message dictionaries
            metadata: Metadata containing user_id, agent_id, etc.

        Returns:
            bool: True if should use agent memory extraction, False for user memory extraction
        """
        # Check if agent_id is present in metadata
        has_agent_id = metadata.get("agent_id") is not None

        # Check if there are assistant role messages
        has_assistant_messages = any(msg.get("role") == "assistant" for msg in messages)

        # Use agent memory extraction if agent_id is present and there are assistant messages
        return has_agent_id and has_assistant_messages

    async def add(
        self,
        messages,
        *,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        infer: bool = True,
        memory_type: Optional[str] = None,
        prompt: Optional[str] = None,
        llm=None,
    ):
        """
        Create a new memory asynchronously.

        Args:
            messages (str or List[Dict[str, str]]): Messages to store in the memory.
            user_id (str, optional): ID of the user creating the memory.
            agent_id (str, optional): ID of the agent creating the memory. Defaults to None.
            run_id (str, optional): ID of the run creating the memory. Defaults to None.
            metadata (dict, optional): Metadata to store with the memory. Defaults to None.
            infer (bool, optional): Whether to infer the memories. Defaults to True.
            memory_type (str, optional): Type of memory to create. Defaults to None.
                                         Pass "procedural_memory" to create procedural memories.
            prompt (str, optional): Prompt to use for the memory creation. Defaults to None.
            llm (BaseChatModel, optional): LLM class to use for generating procedural memories. Defaults to None. Useful when user is using LangChain ChatModel.
        Returns:
            dict: A dictionary containing the result of the memory addition operation.
        """
        processed_metadata, effective_filters = _build_filters_and_metadata(
            user_id=user_id, agent_id=agent_id, run_id=run_id, input_metadata=metadata
        )

        if memory_type is not None and memory_type != MemoryType.PROCEDURAL.value:
            raise ValueError(
                f"Invalid 'memory_type'. Please pass {MemoryType.PROCEDURAL.value} to create procedural memories."
            )

        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        elif isinstance(messages, dict):
            messages = [messages]

        elif not isinstance(messages, list):
            raise Mem0ValidationError(
                message="messages must be str, dict, or list[dict]",
                error_code="VALIDATION_003",
                details={"provided_type": type(messages).__name__, "valid_types": ["str", "dict", "list[dict]"]},
                suggestion="Convert your input to a string, dictionary, or list of dictionaries."
            )

        if agent_id is not None and memory_type == MemoryType.PROCEDURAL.value:
            results = await self._create_procedural_memory(
                messages, metadata=processed_metadata, prompt=prompt, llm=llm
            )
            return results

        if self.config.llm.config.get("enable_vision"):
            messages = parse_vision_messages(messages, self.llm, self.config.llm.config.get("vision_details"))
        else:
            messages = parse_vision_messages(messages)

        vector_store_result = await self._add_to_vector_store(messages, processed_metadata, effective_filters, infer)
        return {"results": vector_store_result}

    async def _add_to_vector_store(
        self,
        messages: list,
        metadata: dict,
        effective_filters: dict,
        infer: bool,
    ):
        if not infer:
            returned_memories = []
            for message_dict in messages:
                if (
                    not isinstance(message_dict, dict)
                    or message_dict.get("role") is None
                    or message_dict.get("content") is None
                ):
                    logger.warning(f"Skipping invalid message format (async): {message_dict}")
                    continue

                if message_dict["role"] == "system":
                    continue

                per_msg_meta = deepcopy(metadata)
                per_msg_meta["role"] = message_dict["role"]

                actor_name = message_dict.get("name")
                if actor_name:
                    per_msg_meta["actor_id"] = actor_name

                msg_content = message_dict["content"]
                msg_embeddings = await asyncio.to_thread(self.embedding_model.embed, msg_content, "add")
                mem_id = await self._create_memory(msg_content, {msg_content: msg_embeddings}, per_msg_meta)

                returned_memories.append(
                    {
                        "id": mem_id,
                        "memory": msg_content,
                        "event": "ADD",
                        "actor_id": actor_name if actor_name else None,
                        "role": message_dict["role"],
                    }
                )
            return returned_memories

        # === V3 PHASED BATCH PIPELINE (async) ===

        # Phase 0: Context gathering
        session_scope = _build_session_scope(effective_filters)
        last_messages = await asyncio.to_thread(self.db.get_last_messages, session_scope, 10)
        parsed_messages = parse_messages(messages)

        # Phase 1: Existing memory retrieval
        search_filters = {k: v for k, v in effective_filters.items() if k in ("user_id", "agent_id", "run_id") and v}
        query_embedding = await asyncio.to_thread(self.embedding_model.embed, parsed_messages, "search")
        existing_results = await asyncio.to_thread(
            self.vector_store.search,
            query=parsed_messages,
            vectors=query_embedding,
            top_k=10,
            filters=search_filters,
        )

        # Map UUIDs to integers (anti-hallucination)
        existing_memories = []
        uuid_mapping = {}
        for idx, mem in enumerate(existing_results):
            uuid_mapping[str(idx)] = mem.id
            existing_memories.append({"id": str(idx), "text": mem.payload.get("data", "")})

        # Phase 2: LLM extraction (single call)
        is_agent_scoped = bool(effective_filters.get("agent_id")) and not effective_filters.get("user_id")
        system_prompt = ADDITIVE_EXTRACTION_PROMPT
        if is_agent_scoped:
            system_prompt += AGENT_CONTEXT_SUFFIX

        custom_instr = self.custom_instructions

        user_prompt = generate_additive_extraction_prompt(
            existing_memories=existing_memories,
            new_messages=parsed_messages,
            last_k_messages=last_messages,
            custom_instructions=custom_instr,
        )

        try:
            response = await asyncio.to_thread(
                self.llm.generate_response,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
            )
        except Exception as e:
            logger.error(f"LLM extraction failed (async): {e}")
            return []

        # Parse response
        try:
            response = remove_code_blocks(response)
            if not response or not response.strip():
                extracted_memories = []
            else:
                try:
                    extracted_memories = json.loads(response, strict=False).get("memory", [])
                except json.JSONDecodeError:
                    extracted_json = extract_json(response)
                    extracted_memories = json.loads(extracted_json, strict=False).get("memory", [])
        except Exception as e:
            logger.error(f"Error parsing extraction response (async): {e}")
            extracted_memories = []

        if not extracted_memories:
            await asyncio.to_thread(self.db.save_messages, messages, session_scope)
            return []

        # Phase 3: Batch embed all extracted memory texts
        mem_texts = [m.get("text", "") for m in extracted_memories if m.get("text")]
        try:
            mem_embeddings_list = await asyncio.to_thread(self.embedding_model.embed_batch, mem_texts, "add")
            embed_map = dict(zip(mem_texts, mem_embeddings_list))
        except Exception:
            embed_map = {}
            for text in mem_texts:
                try:
                    embed_map[text] = await asyncio.to_thread(self.embedding_model.embed, text, "add")
                except Exception as e:
                    logger.warning(f"Failed to embed memory text (async): {e}")

        # Phase 4: Per-memory CPU processing + Phase 5: Hash dedup
        existing_hashes = set()
        for mem in existing_results:
            h = mem.payload.get("hash") if hasattr(mem, "payload") and mem.payload else None
            if h:
                existing_hashes.add(h)

        records = []
        seen_hashes = set()
        for mem in extracted_memories:
            text = mem.get("text")
            if not text or text not in embed_map:
                continue

            mem_hash = hashlib.md5(text.encode()).hexdigest()
            if mem_hash in existing_hashes or mem_hash in seen_hashes:
                logger.debug(f"Skipping duplicate memory (hash match, async): {text[:50]}")
                continue
            seen_hashes.add(mem_hash)

            text_lemmatized = lemmatize_for_bm25(text)

            memory_id = str(uuid.uuid4())
            mem_metadata = deepcopy(metadata)
            mem_metadata["data"] = text
            mem_metadata["text_lemmatized"] = text_lemmatized
            mem_metadata["hash"] = mem_hash
            if "created_at" not in mem_metadata:
                mem_metadata["created_at"] = datetime.now(timezone.utc).isoformat()
            mem_metadata["updated_at"] = mem_metadata["created_at"]
            if mem.get("attributed_to"):
                mem_metadata["attributed_to"] = mem["attributed_to"]

            records.append((memory_id, text, embed_map[text], mem_metadata))

        if not records:
            await asyncio.to_thread(self.db.save_messages, messages, session_scope)
            return []

        # Phase 6: Batch persist
        all_vectors = [r[2] for r in records]
        all_ids = [r[0] for r in records]
        all_payloads = [r[3] for r in records]

        try:
            await asyncio.to_thread(
                self.vector_store.insert,
                vectors=all_vectors,
                ids=all_ids,
                payloads=all_payloads,
            )
        except Exception:
            for mid, vec, pay in zip(all_ids, all_vectors, all_payloads):
                try:
                    await asyncio.to_thread(self.vector_store.insert, vectors=[vec], ids=[mid], payloads=[pay])
                except Exception as e:
                    logger.error(f"Failed to insert memory {mid} (async): {e}")

        # Batch history
        history_records = [
            {
                "memory_id": r[0],
                "old_memory": None,
                "new_memory": r[1],
                "event": "ADD",
                "created_at": r[3].get("created_at"),
                "is_deleted": 0,
            }
            for r in records
        ]
        try:
            await asyncio.to_thread(self.db.batch_add_history, history_records)
        except Exception:
            for hr in history_records:
                try:
                    await asyncio.to_thread(
                        self.db.add_history, hr["memory_id"], None, hr["new_memory"], "ADD",
                        created_at=hr.get("created_at")
                    )
                except Exception as e:
                    logger.error(f"Failed to add history for {hr['memory_id']} (async): {e}")

        # Phase 7: Batch entity linking
        try:
            all_texts = [r[1] for r in records]
            all_entities = await asyncio.to_thread(extract_entities_batch, all_texts)

            # 7a: Global dedup
            global_entities = {}
            for idx, (memory_id, text, embedding, payload) in enumerate(records):
                entities = all_entities[idx] if idx < len(all_entities) else []
                for entity_type, entity_text in entities:
                    key = entity_text.strip().lower()
                    if key in global_entities:
                        global_entities[key][2].add(memory_id)
                    else:
                        global_entities[key] = [entity_type, entity_text, {memory_id}]

            if global_entities:
                ordered_keys = list(global_entities.keys())
                entity_texts = [global_entities[k][1] for k in ordered_keys]

                # 7b: Batch embed entities
                try:
                    entity_embeddings = await asyncio.to_thread(self.embedding_model.embed_batch, entity_texts, "add")
                except Exception:
                    entity_embeddings = []
                    for t in entity_texts:
                        try:
                            entity_embeddings.append(await asyncio.to_thread(self.embedding_model.embed, t, "add"))
                        except Exception:
                            entity_embeddings.append(None)

                valid = [(i, k) for i, k in enumerate(ordered_keys) if entity_embeddings[i] is not None]
                if valid:
                    valid_indices, valid_keys = zip(*valid)
                    valid_vectors = [entity_embeddings[i] for i in valid_indices]

                    # 7c: Batch search for existing entities
                    valid_texts = [global_entities[k][1] for k in valid_keys]
                    existing_matches = await asyncio.to_thread(
                        self.entity_store.search_batch,
                        queries=valid_texts,
                        vectors_list=valid_vectors,
                        top_k=1,
                        filters=search_filters,
                    )

                    # 7d: Separate into inserts vs updates
                    to_insert_vectors, to_insert_ids, to_insert_payloads = [], [], []
                    for j, key in enumerate(valid_keys):
                        entity_type, entity_text, memory_ids = global_entities[key]
                        matches = existing_matches[j] if j < len(existing_matches) else []

                        if matches and matches[0].score >= 0.95:
                            match = matches[0]
                            payload = match.payload or {}
                            linked = set(payload.get("linked_memory_ids", []))
                            linked |= memory_ids
                            payload["linked_memory_ids"] = sorted(linked)
                            try:
                                await asyncio.to_thread(
                                    self.entity_store.update,
                                    vector_id=match.id,
                                    vector=None,
                                    payload=payload,
                                )
                            except Exception as e:
                                logger.debug(f"Entity update failed for '{entity_text}' (async): {e}")
                        else:
                            to_insert_vectors.append(valid_vectors[j])
                            to_insert_ids.append(str(uuid.uuid4()))
                            to_insert_payloads.append({
                                "data": entity_text,
                                "entity_type": entity_type,
                                "linked_memory_ids": sorted(memory_ids),
                                **search_filters,
                            })

                    # 7e: Batch insert new entities
                    if to_insert_vectors:
                        try:
                            await asyncio.to_thread(
                                self.entity_store.insert,
                                vectors=to_insert_vectors,
                                ids=to_insert_ids,
                                payloads=to_insert_payloads,
                            )
                        except Exception as e:
                            logger.warning(f"Batch entity insert failed (async): {e}")
        except Exception as e:
            logger.warning(f"Batch entity linking failed (async): {e}")

        # Phase 8: Save messages + return
        await asyncio.to_thread(self.db.save_messages, messages, session_scope)

        returned_memories = [
            {"id": r[0], "memory": r[1], "event": "ADD"}
            for r in records
        ]

        keys, encoded_ids = process_telemetry_filters(effective_filters)
        capture_event(
            "mem0.add",
            self,
            {"version": self.api_version, "keys": keys, "encoded_ids": encoded_ids, "sync_type": "async"},
        )
        return returned_memories

    async def get(self, memory_id):
        """
        Retrieve a memory by ID asynchronously.

        Args:
            memory_id (str): ID of the memory to retrieve.

        Returns:
            dict: Retrieved memory.
        """
        capture_event("mem0.get", self, {"memory_id": memory_id, "sync_type": "async"})
        memory = await asyncio.to_thread(self.vector_store.get, vector_id=memory_id)
        if not memory:
            return None

        promoted_payload_keys = [
            "user_id",
            "agent_id",
            "run_id",
            "actor_id",
            "role",
        ]

        core_and_promoted_keys = {"data", "hash", "created_at", "updated_at", "id", "text_lemmatized", "attributed_to", *promoted_payload_keys}

        result_item = MemoryItem(
            id=memory.id,
            memory=memory.payload.get("data", ""),
            hash=memory.payload.get("hash"),
            created_at=memory.payload.get("created_at"),
            updated_at=memory.payload.get("updated_at"),
        ).model_dump()

        for key in promoted_payload_keys:
            if key in memory.payload:
                result_item[key] = memory.payload[key]

        additional_metadata = {k: v for k, v in memory.payload.items() if k not in core_and_promoted_keys}
        if additional_metadata:
            result_item["metadata"] = additional_metadata

        return result_item

    async def get_all(
        self,
        *,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 20,
        **kwargs,
    ):
        """
        List all memories.

        Args:
            filters (dict): Filter dict containing entity IDs and optional metadata filters.
                Must contain at least one of: user_id, agent_id, run_id.
                Example: filters={"user_id": "u1", "agent_id": "a1"}
            top_k (int, optional): The maximum number of memories to return. Defaults to 20.

        Returns:
            dict: A dictionary containing a list of memories under the "results" key.
                  Example for v1.1+: `{"results": [{"id": "...", "memory": "...", ...}]}`

        Raises:
            ValueError: If filters doesn't contain at least one of user_id, agent_id, run_id,
                or if top_k is invalid.
        """
        # Reject top-level entity params - must use filters instead
        _reject_top_level_entity_params(kwargs, "get_all")

        # Validate top_k
        _validate_search_params(top_k=top_k)

        # Validate and trim entity IDs in filters
        effective_filters = dict(filters) if filters else {}
        if "user_id" in effective_filters:
            effective_filters["user_id"] = _validate_and_trim_entity_id(
                effective_filters["user_id"], "user_id"
            )
        if "agent_id" in effective_filters:
            effective_filters["agent_id"] = _validate_and_trim_entity_id(
                effective_filters["agent_id"], "agent_id"
            )
        if "run_id" in effective_filters:
            effective_filters["run_id"] = _validate_and_trim_entity_id(
                effective_filters["run_id"], "run_id"
            )

        # Validate filters contains at least one entity ID
        if not any(key in effective_filters for key in ("user_id", "agent_id", "run_id")):
            raise ValueError(
                "filters must contain at least one of: user_id, agent_id, run_id. "
                "Example: filters={'user_id': 'u1'}"
            )

        limit = top_k

        keys, encoded_ids = process_telemetry_filters(effective_filters)
        capture_event(
            "mem0.get_all", self, {"limit": limit, "keys": keys, "encoded_ids": encoded_ids, "sync_type": "async"}
        )

        all_memories_result = await self._get_all_from_vector_store(effective_filters, limit)

        return {"results": all_memories_result}

    async def _get_all_from_vector_store(self, filters, limit):
        memories_result = await asyncio.to_thread(self.vector_store.list, filters=filters, top_k=limit)

        # Handle different vector store return formats by inspecting first element
        if isinstance(memories_result, (tuple, list)) and len(memories_result) > 0:
            first_element = memories_result[0]

            # If first element is a container, unwrap one level
            if isinstance(first_element, (list, tuple)):
                actual_memories = first_element
            else:
                # First element is a memory object, structure is already flat
                actual_memories = memories_result
        else:
            actual_memories = memories_result

        promoted_payload_keys = [
            "user_id",
            "agent_id",
            "run_id",
            "actor_id",
            "role",
        ]
        core_and_promoted_keys = {"data", "hash", "created_at", "updated_at", "id", "text_lemmatized", "attributed_to", *promoted_payload_keys}

        formatted_memories = []
        for mem in actual_memories:
            memory_item_dict = MemoryItem(
                id=mem.id,
                memory=mem.payload.get("data", ""),
                hash=mem.payload.get("hash"),
                created_at=mem.payload.get("created_at"),
                updated_at=mem.payload.get("updated_at"),
            ).model_dump(exclude={"score"})

            for key in promoted_payload_keys:
                if key in mem.payload:
                    memory_item_dict[key] = mem.payload[key]

            additional_metadata = {k: v for k, v in mem.payload.items() if k not in core_and_promoted_keys}
            if additional_metadata:
                memory_item_dict["metadata"] = additional_metadata

            formatted_memories.append(memory_item_dict)

        return formatted_memories

    async def search(
        self,
        query: str,
        *,
        top_k: int = 20,
        filters: Optional[Dict[str, Any]] = None,
        threshold: float = 0.1,
        rerank: bool = False,
        **kwargs,
    ):
        """
        Searches for memories based on a query.

        Args:
            query (str): Query to search for.
            top_k (int, optional): Maximum number of results to return. Defaults to 20.
            filters (dict): Filter dict containing entity IDs and optional metadata filters.
                Must contain at least one of: user_id, agent_id, run_id.
                Example: filters={"user_id": "u1", "agent_id": "a1"}

                Enhanced metadata filtering with operators:
                - {"key": "value"} - exact match
                - {"key": {"eq": "value"}} - equals
                - {"key": {"ne": "value"}} - not equals
                - {"key": {"in": ["val1", "val2"]}} - in list
                - {"key": {"nin": ["val1", "val2"]}} - not in list
                - {"key": {"gt": 10}} - greater than
                - {"key": {"gte": 10}} - greater than or equal
                - {"key": {"lt": 10}} - less than
                - {"key": {"lte": 10}} - less than or equal
                - {"key": {"contains": "text"}} - contains text
                - {"key": {"icontains": "text"}} - case-insensitive contains
                - {"key": "*"} - wildcard match (any value)
                - {"AND": [filter1, filter2]} - logical AND
                - {"OR": [filter1, filter2]} - logical OR
                - {"NOT": [filter1]} - logical NOT
            threshold (float, optional): Minimum score for a memory to be included. Defaults to 0.1.
            rerank (bool, optional): Whether to rerank results. Defaults to False.

        Returns:
            dict: A dictionary containing the search results under a "results" key.
                  Example for v1.1+: `{"results": [{"id": "...", "memory": "...", "score": 0.8, ...}]}`

        Raises:
            ValueError: If filters doesn't contain at least one of user_id, agent_id, run_id,
                or if threshold/top_k values are invalid.
        """
        # Reject top-level entity params - must use filters instead
        _reject_top_level_entity_params(kwargs, "search")

        # Validate search parameters (before applying defaults)
        _validate_search_params(threshold=threshold, top_k=top_k)

        # Validate and trim entity IDs in filters
        effective_filters = filters.copy() if filters else {}
        if "user_id" in effective_filters:
            effective_filters["user_id"] = _validate_and_trim_entity_id(
                effective_filters["user_id"], "user_id"
            )
        if "agent_id" in effective_filters:
            effective_filters["agent_id"] = _validate_and_trim_entity_id(
                effective_filters["agent_id"], "agent_id"
            )
        if "run_id" in effective_filters:
            effective_filters["run_id"] = _validate_and_trim_entity_id(
                effective_filters["run_id"], "run_id"
            )

        # Validate filters contains at least one entity ID
        if not any(key in effective_filters for key in ("user_id", "agent_id", "run_id")):
            raise ValueError(
                "filters must contain at least one of: user_id, agent_id, run_id. "
                "Example: filters={'user_id': 'u1'}"
            )

        limit = top_k

        # Apply enhanced metadata filtering if advanced operators are detected
        if self._has_advanced_operators(effective_filters):
            processed_filters = self._process_metadata_filters(effective_filters)
            # Remove logical/operator keys that have been reprocessed
            for logical_key in ("AND", "OR", "NOT"):
                effective_filters.pop(logical_key, None)
            for fk in list(effective_filters.keys()):
                if fk not in ("AND", "OR", "NOT", "user_id", "agent_id", "run_id") and isinstance(effective_filters.get(fk), dict):
                    effective_filters.pop(fk, None)
            effective_filters.update(processed_filters)

        keys, encoded_ids = process_telemetry_filters(effective_filters)
        capture_event(
            "mem0.search",
            self,
            {
                "limit": limit,
                "version": self.api_version,
                "keys": keys,
                "encoded_ids": encoded_ids,
                "sync_type": "async",
                "threshold": threshold,
                "advanced_filters": bool(filters and self._has_advanced_operators(filters)),
            },
        )

        original_memories = await self._search_vector_store(query, effective_filters, limit, threshold)

        # Apply reranking if enabled and reranker is available
        if rerank and self.reranker and original_memories:
            try:
                # Run reranking in thread pool to avoid blocking async loop
                reranked_memories = await asyncio.to_thread(
                    self.reranker.rerank, query, original_memories, limit
                )
                original_memories = reranked_memories
            except Exception as e:
                logger.warning(f"Reranking failed, using original results: {e}")

        return {"results": original_memories}

    def _process_metadata_filters(self, metadata_filters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process enhanced metadata filters and convert them to vector store compatible format.

        Args:
            metadata_filters: Enhanced metadata filters with operators

        Returns:
            Dict of processed filters compatible with vector store
        """
        processed_filters = {}

        def process_condition(key: str, condition: Any) -> Dict[str, Any]:
            if not isinstance(condition, dict):
                # Simple equality: {"key": "value"}
                if condition == "*":
                    # Wildcard: match everything for this field (implementation depends on vector store)
                    return {key: "*"}
                return {key: condition}

            result = {}
            for operator, value in condition.items():
                # Map platform operators to universal format that can be translated by each vector store
                operator_map = {
                    "eq": "eq", "ne": "ne", "gt": "gt", "gte": "gte",
                    "lt": "lt", "lte": "lte", "in": "in", "nin": "nin",
                    "contains": "contains", "icontains": "icontains"
                }

                if operator in operator_map:
                    result.setdefault(key, {})[operator_map[operator]] = value
                else:
                    raise ValueError(f"Unsupported metadata filter operator: {operator}")
            return result

        for key, value in metadata_filters.items():
            if key == "AND":
                # Logical AND: combine multiple conditions
                if not isinstance(value, list):
                    raise ValueError("AND operator requires a list of conditions")
                for condition in value:
                    for sub_key, sub_value in condition.items():
                        processed_filters.update(process_condition(sub_key, sub_value))
            elif key == "OR":
                # Logical OR: Pass through to vector store for implementation-specific handling
                if not isinstance(value, list) or not value:
                    raise ValueError("OR operator requires a non-empty list of conditions")
                # Store OR conditions in a way that vector stores can interpret
                processed_filters["$or"] = []
                for condition in value:
                    or_condition = {}
                    for sub_key, sub_value in condition.items():
                        or_condition.update(process_condition(sub_key, sub_value))
                    processed_filters["$or"].append(or_condition)
            elif key == "NOT":
                # Logical NOT: Pass through to vector store for implementation-specific handling
                if not isinstance(value, list) or not value:
                    raise ValueError("NOT operator requires a non-empty list of conditions")
                processed_filters["$not"] = []
                for condition in value:
                    not_condition = {}
                    for sub_key, sub_value in condition.items():
                        not_condition.update(process_condition(sub_key, sub_value))
                    processed_filters["$not"].append(not_condition)
            else:
                processed_filters.update(process_condition(key, value))

        return processed_filters

    def _has_advanced_operators(self, filters: Dict[str, Any]) -> bool:
        """
        Check if filters contain advanced operators that need special processing.

        Args:
            filters: Dictionary of filters to check

        Returns:
            bool: True if advanced operators are detected
        """
        if not isinstance(filters, dict):
            return False

        for key, value in filters.items():
            # Check for platform-style logical operators
            if key in ["AND", "OR", "NOT"]:
                return True
            # Check for comparison operators (without $ prefix for universal compatibility)
            if isinstance(value, dict):
                for op in value.keys():
                    if op in ["eq", "ne", "gt", "gte", "lt", "lte", "in", "nin", "contains", "icontains"]:
                        return True
            # Check for wildcard values
            if value == "*":
                return True
        return False

    async def _search_vector_store(self, query, filters, limit, threshold=0.1):
        if threshold is None:
            threshold = 0.1

        # Step 1: Preprocess query (CPU-bound)
        query_lemmatized = await asyncio.to_thread(lemmatize_for_bm25, query)
        query_entities = await asyncio.to_thread(extract_entities, query)

        # Step 2: Embed query
        embeddings = await asyncio.to_thread(self.embedding_model.embed, query, "search")

        # Step 3: Semantic search (over-fetch)
        internal_limit = max(limit * 4, 60)
        semantic_results = await asyncio.to_thread(
            self.vector_store.search, query=query, vectors=embeddings, top_k=internal_limit, filters=filters
        )

        # Step 4: Keyword search (if store supports it)
        keyword_results = await asyncio.to_thread(
            self.vector_store.keyword_search, query=query_lemmatized, top_k=internal_limit, filters=filters
        )

        # Step 5: Compute BM25 scores
        bm25_scores = {}
        if keyword_results is not None:
            midpoint, steepness = get_bm25_params(query, lemmatized=query_lemmatized)
            for mem in keyword_results:
                mem_id = str(mem.id) if hasattr(mem, 'id') else str(mem.get('id', ''))
                raw_score = mem.score if hasattr(mem, 'score') else mem.get('score', 0)
                if raw_score and raw_score > 0:
                    bm25_scores[mem_id] = normalize_bm25(raw_score, midpoint, steepness)

        # Step 6: Compute entity boosts
        entity_boosts = {}
        if query_entities:
            entity_boosts = await self._compute_entity_boosts_async(query_entities, filters)

        # Step 7: Build candidate set from semantic results
        candidates = []
        for mem in semantic_results:
            mem_id = str(mem.id)
            candidates.append({
                "id": mem_id,
                "score": mem.score,
                "payload": mem.payload if hasattr(mem, 'payload') else {},
            })

        # Step 8: Score and rank
        scored_results = score_and_rank(
            semantic_results=candidates,
            bm25_scores=bm25_scores,
            entity_boosts=entity_boosts,
            threshold=threshold,
            top_k=limit,
        )

        # Step 9: Format results
        promoted_payload_keys = [
            "user_id",
            "agent_id",
            "run_id",
            "actor_id",
            "role",
        ]
        core_and_promoted_keys = {"data", "hash", "created_at", "updated_at", "id", "text_lemmatized", "attributed_to", *promoted_payload_keys}

        original_memories = []
        for scored in scored_results:
            payload = scored.get("payload") or {}
            if not payload.get("data"):
                continue

            memory_item_dict = MemoryItem(
                id=scored["id"],
                memory=payload.get("data", ""),
                hash=payload.get("hash"),
                created_at=payload.get("created_at"),
                updated_at=payload.get("updated_at"),
                score=scored["score"],
            ).model_dump()

            if payload.get("_bfs_source"):
                memory_item_dict["source"] = payload["_bfs_source"]

            for key in promoted_payload_keys:
                if key in payload:
                    memory_item_dict[key] = payload[key]

            additional_metadata = {k: v for k, v in payload.items() if k not in core_and_promoted_keys}
            if additional_metadata:
                if not memory_item_dict.get("metadata"):
                    memory_item_dict["metadata"] = {}
                memory_item_dict["metadata"].update(additional_metadata)

            original_memories.append(memory_item_dict)

        return original_memories

    async def _compute_entity_boosts_async(self, query_entities, filters):
        """Async version of entity boost computation."""
        seen = set()
        deduped = []
        for entity_type, entity_text in query_entities[:8]:
            key = entity_text.strip().lower()
            if key and key not in seen:
                seen.add(key)
                deduped.append((entity_type, entity_text))

        if not deduped:
            return {}

        search_filters = {k: v for k, v in filters.items() if k in ("user_id", "agent_id", "run_id") and v}
        memory_boosts = {}

        try:
            for _, entity_text in deduped:
                entity_embedding = await asyncio.to_thread(self.embedding_model.embed, entity_text, "search")
                matches = await asyncio.to_thread(
                    self.entity_store.search,
                    query=entity_text,
                    vectors=entity_embedding,
                    top_k=500,
                    filters=search_filters,
                )

                for match in matches:
                    similarity = match.score if hasattr(match, 'score') else 0.0
                    if similarity < 0.5:
                        continue

                    payload = match.payload if hasattr(match, 'payload') else {}
                    linked_memory_ids = payload.get("linked_memory_ids", [])
                    if not isinstance(linked_memory_ids, list):
                        continue

                    num_linked = max(len(linked_memory_ids), 1)
                    memory_count_weight = 1.0 / (1.0 + 0.001 * ((num_linked - 1) ** 2))
                    boost = similarity * ENTITY_BOOST_WEIGHT * memory_count_weight

                    for memory_id in linked_memory_ids:
                        if memory_id:
                            memory_key = str(memory_id)
                            memory_boosts[memory_key] = max(memory_boosts.get(memory_key, 0.0), boost)

        except Exception as e:
            logger.warning(f"Entity boost computation failed: {e}")

        return memory_boosts

    async def update(self, memory_id, data, metadata: Optional[Dict[str, Any]] = None):
        """
        Update a memory by ID asynchronously.

        Args:
            memory_id (str): ID of the memory to update.
            data (str): New content to update the memory with.
            metadata (dict, optional): Metadata to update with the memory. Defaults to None.

        Returns:
            dict: Success message indicating the memory was updated.

        Example:
            >>> await m.update(memory_id="mem_123", data="Likes to play tennis on weekends")
            {'message': 'Memory updated successfully!'}
        """
        capture_event("mem0.update", self, {"memory_id": memory_id, "sync_type": "async"})

        embeddings = await asyncio.to_thread(self.embedding_model.embed, data, "update")
        existing_embeddings = {data: embeddings}

        await self._update_memory(memory_id, data, existing_embeddings, metadata)
        return {"message": "Memory updated successfully!"}

    async def delete(self, memory_id):
        """
        Delete a memory by ID asynchronously.

        Args:
            memory_id (str): ID of the memory to delete.
        """
        capture_event("mem0.delete", self, {"memory_id": memory_id, "sync_type": "async"})

        existing_memory = await asyncio.to_thread(self.vector_store.get, vector_id=memory_id)
        if existing_memory is None:
            raise ValueError(f"Memory with id {memory_id} not found")

        await self._delete_memory(memory_id, existing_memory)
        return {"message": "Memory deleted successfully!"}

    async def delete_all(self, user_id=None, agent_id=None, run_id=None):
        """
        Delete all memories asynchronously.

        Args:
            user_id (str, optional): ID of the user to delete memories for. Defaults to None.
            agent_id (str, optional): ID of the agent to delete memories for. Defaults to None.
            run_id (str, optional): ID of the run to delete memories for. Defaults to None.
        """
        filters = {}
        if user_id:
            filters["user_id"] = user_id
        if agent_id:
            filters["agent_id"] = agent_id
        if run_id:
            filters["run_id"] = run_id

        if not filters:
            raise ValueError(
                "At least one filter is required to delete all memories. If you want to delete all memories, use the `reset()` method."
            )

        keys, encoded_ids = process_telemetry_filters(filters)
        capture_event("mem0.delete_all", self, {"keys": keys, "encoded_ids": encoded_ids, "sync_type": "async"})
        memories = await asyncio.to_thread(self.vector_store.list, filters=filters)

        delete_tasks = []
        for memory in memories[0]:
            delete_tasks.append(self._delete_memory(memory.id))

        await asyncio.gather(*delete_tasks)

        logger.info(f"Deleted {len(memories[0])} memories")

        return {"message": "Memories deleted successfully!"}

    async def history(self, memory_id):
        """
        Get the history of changes for a memory by ID asynchronously.

        Args:
            memory_id (str): ID of the memory to get history for.

        Returns:
            list: List of changes for the memory.
        """
        capture_event("mem0.history", self, {"memory_id": memory_id, "sync_type": "async"})
        return await asyncio.to_thread(self.db.get_history, memory_id)

    async def _create_memory(self, data, existing_embeddings, metadata=None):
        logger.debug(f"Creating memory with {data=}")
        if data in existing_embeddings:
            embeddings = existing_embeddings[data]
        else:
            embeddings = await asyncio.to_thread(self.embedding_model.embed, data, memory_action="add")

        memory_id = str(uuid.uuid4())
        new_metadata = deepcopy(metadata) if metadata is not None else {}
        new_metadata["data"] = data
        new_metadata["hash"] = hashlib.md5(data.encode()).hexdigest()
        if "created_at" not in new_metadata:
            new_metadata["created_at"] = datetime.now(timezone.utc).isoformat()
        new_metadata["updated_at"] = new_metadata["created_at"]

        await asyncio.to_thread(
            self.vector_store.insert,
            vectors=[embeddings],
            ids=[memory_id],
            payloads=[new_metadata],
        )

        await asyncio.to_thread(
            self.db.add_history,
            memory_id,
            None,
            data,
            "ADD",
            created_at=new_metadata.get("created_at"),
            updated_at=new_metadata.get("updated_at"),
            actor_id=new_metadata.get("actor_id"),
            role=new_metadata.get("role"),
        )

        return memory_id

    async def _create_procedural_memory(self, messages, metadata=None, llm=None, prompt=None):
        """
        Create a procedural memory asynchronously

        Args:
            messages (list): List of messages to create a procedural memory from.
            metadata (dict): Metadata to create a procedural memory from.
            llm (llm, optional): LLM to use for the procedural memory creation. Defaults to None.
            prompt (str, optional): Prompt to use for the procedural memory creation. Defaults to None.
        """
        try:
            from langchain_core.messages.utils import (
                convert_to_messages,  # type: ignore
            )
        except Exception:
            logger.error(
                "Import error while loading langchain-core. Please install 'langchain-core' to use procedural memory."
            )
            raise

        logger.info("Creating procedural memory")

        parsed_messages = [
            {"role": "system", "content": prompt or PROCEDURAL_MEMORY_SYSTEM_PROMPT},
            *messages,
            {"role": "user", "content": "Create procedural memory of the above conversation."},
        ]

        try:
            if llm is not None:
                parsed_messages = convert_to_messages(parsed_messages)
                response = await asyncio.to_thread(llm.invoke, input=parsed_messages)
                procedural_memory = response.content
            else:
                procedural_memory = await asyncio.to_thread(self.llm.generate_response, messages=parsed_messages)
                procedural_memory = remove_code_blocks(procedural_memory)

        except Exception as e:
            logger.error(f"Error generating procedural memory summary: {e}")
            raise

        if metadata is None:
            raise ValueError("Metadata cannot be done for procedural memory.")

        metadata = {**metadata, "memory_type": MemoryType.PROCEDURAL.value}
        embeddings = await asyncio.to_thread(self.embedding_model.embed, procedural_memory, memory_action="add")
        memory_id = await self._create_memory(procedural_memory, {procedural_memory: embeddings}, metadata=metadata)
        capture_event("mem0._create_procedural_memory", self, {"memory_id": memory_id, "sync_type": "async"})

        result = {"results": [{"id": memory_id, "memory": procedural_memory, "event": "ADD"}]}

        return result

    async def _update_memory(self, memory_id, data, existing_embeddings, metadata=None):
        logger.info(f"Updating memory with {data=}")

        try:
            existing_memory = await asyncio.to_thread(self.vector_store.get, vector_id=memory_id)
        except Exception:
            logger.error(f"Error getting memory with ID {memory_id} during update.")
            raise ValueError(f"Error getting memory with ID {memory_id}. Please provide a valid 'memory_id'")

        if existing_memory is None:
            raise ValueError(f"Memory with id {memory_id} not found. Please provide a valid 'memory_id'")

        prev_value = existing_memory.payload.get("data")

        new_metadata = deepcopy(metadata) if metadata is not None else {}

        new_metadata["data"] = data
        new_metadata["hash"] = hashlib.md5(data.encode()).hexdigest()
        new_metadata["text_lemmatized"] = lemmatize_for_bm25(data)
        new_metadata["created_at"] = existing_memory.payload.get("created_at")
        new_metadata["updated_at"] = datetime.now(timezone.utc).isoformat()

        # Preserve session identifiers from existing memory only if not provided in new metadata
        if "user_id" not in new_metadata and "user_id" in existing_memory.payload:
            new_metadata["user_id"] = existing_memory.payload["user_id"]
        if "agent_id" not in new_metadata and "agent_id" in existing_memory.payload:
            new_metadata["agent_id"] = existing_memory.payload["agent_id"]
        if "run_id" not in new_metadata and "run_id" in existing_memory.payload:
            new_metadata["run_id"] = existing_memory.payload["run_id"]

        if "actor_id" in existing_memory.payload:
            new_metadata["actor_id"] = existing_memory.payload["actor_id"]
        if "role" not in new_metadata and "role" in existing_memory.payload:
            new_metadata["role"] = existing_memory.payload["role"]

        if data in existing_embeddings:
            embeddings = existing_embeddings[data]
        else:
            embeddings = await asyncio.to_thread(self.embedding_model.embed, data, "update")

        await asyncio.to_thread(
            self.vector_store.update,
            vector_id=memory_id,
            vector=embeddings,
            payload=new_metadata,
        )
        logger.info(f"Updating memory with ID {memory_id=} with {data=}")

        await asyncio.to_thread(
            self.db.add_history,
            memory_id,
            prev_value,
            data,
            "UPDATE",
            created_at=new_metadata["created_at"],
            updated_at=new_metadata["updated_at"],
            actor_id=new_metadata.get("actor_id"),
            role=new_metadata.get("role"),
        )

        # Entity-store cleanup: strip this memory's id from old-text entities,
        # then re-extract entities from the new text and link them back.
        session_filters = {k: new_metadata[k] for k in ("user_id", "agent_id", "run_id") if new_metadata.get(k)}
        await self._remove_memory_from_entity_store(memory_id, session_filters)
        await self._link_entities_for_memory(memory_id, data, session_filters)

        return memory_id

    async def _delete_memory(self, memory_id, existing_memory=None):
        logger.info(f"Deleting memory with {memory_id=}")
        if existing_memory is None:
            existing_memory = await asyncio.to_thread(self.vector_store.get, vector_id=memory_id)
            if existing_memory is None:
                raise ValueError(f"Memory with id {memory_id} not found. Please provide a valid 'memory_id'")
        prev_value = existing_memory.payload.get("data", "")
        created_at = _normalize_iso_timestamp_to_utc(existing_memory.payload.get("created_at"))
        updated_at = datetime.now(timezone.utc).isoformat()
        payload = existing_memory.payload or {}
        session_filters = {k: payload[k] for k in ("user_id", "agent_id", "run_id") if payload.get(k)}

        await asyncio.to_thread(self.vector_store.delete, vector_id=memory_id)
        await asyncio.to_thread(
            self.db.add_history,
            memory_id,
            prev_value,
            None,
            "DELETE",
            created_at=created_at,
            updated_at=updated_at,
            actor_id=existing_memory.payload.get("actor_id"),
            role=existing_memory.payload.get("role"),
            is_deleted=1,
        )

        # Entity-store cleanup: strip this memory's id from any entity records
        # that linked to it. Non-fatal — the helper swallows errors.
        await self._remove_memory_from_entity_store(memory_id, session_filters)

        return memory_id

    async def reset(self):
        """
        Reset the memory store asynchronously by:
            Deletes the vector store collection
            Resets the database
            Recreates the vector store with a new client
        """
        logger.warning("Resetting all memories")
        await asyncio.to_thread(self.vector_store.delete_col)

        gc.collect()

        if hasattr(self.vector_store, "client") and hasattr(self.vector_store.client, "close"):
            await asyncio.to_thread(self.vector_store.client.close)

        if hasattr(self.db, "connection") and self.db.connection:
            await asyncio.to_thread(lambda: self.db.connection.execute("DROP TABLE IF EXISTS history"))
            await asyncio.to_thread(self.db.connection.close)

        self.db = SQLiteManager(self.config.history_db_path)

        self.vector_store = VectorStoreFactory.create(
            self.config.vector_store.provider, self.config.vector_store.config
        )

        capture_event("mem0.reset", self, {"sync_type": "async"})

    def close(self):
        """Release resources held by this AsyncMemory instance."""
        if hasattr(self, "db") and self.db is not None:
            self.db.close()
            self.db = None

    async def chat(self, query):
        raise NotImplementedError("Chat function not implemented yet.")
