from __future__ import annotations

from memory.query_engine import QueryEngine

from .baselines import RetrievalOnlyBaseline


class MethodRegistry:
    GRAPH_METHODS = {"graph_full", "basic_retrieval", "no_causal", "no_temporal", "flat_graph"}
    BASELINE_METHODS = {"vector_only", "keyword_only", "scan_only"}

    ABLATIONS = {
        "basic_retrieval": "basic_retrieval",
        "no_causal": "no_causal",
        "no_temporal": "no_temporal",
        "flat_graph": "flat_graph",
    }

    @classmethod
    def create(cls, method: str, builder):
        if method in cls.GRAPH_METHODS:
            ablation_config = {}
            if method in cls.ABLATIONS:
                ablation_config[cls.ABLATIONS[method]] = True
            return QueryEngine(
                builder.trg,
                builder.node_index,
                entity_session_map=getattr(builder, "entity_session_map", None),
                entity_dia_map=getattr(builder, "entity_dia_map", None),
                llm_controller=builder.llm_controller,
                ablation_config=ablation_config,
            )

        if method in cls.BASELINE_METHODS:
            return RetrievalOnlyBaseline(builder, method)

        raise ValueError(f"Unsupported method: {method}")
