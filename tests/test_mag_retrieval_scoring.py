from benchmarks.common.mem0_client import format_search_results
from mag.core import (
    _mag_add_source,
    _mag_bfs_shadow_gate,
    _mag_build_query_plan,
    _mag_candidate_evidence_features,
    _mag_dedupe_final_context,
    _mag_diverse_topk,
    _mag_time_constraints_match,
)


def test_evidence_features_reward_query_and_temporal_coverage():
    features = _mag_candidate_evidence_features(
        "When did Melanie sign up for a pottery class?",
        {
            "memory": "[2023-07-02] Melanie signed up for a pottery class.",
            "created_at": "2023-07-02T00:00:00+00:00",
        },
    )

    assert features["query_coverage"] > 0.5
    assert features["temporal_cue"] == 1.0
    assert features["evidence_score"] > 0.6


def test_query_plan_marks_aggregate_temporal_and_entities():
    plan = _mag_build_query_plan(
        "What recipes did Caroline recommend in September?",
        [("PERSON", "Caroline")],
    )

    assert plan["answer_shape"] == "list"
    assert plan["evidence_mode"] == "temporal_aggregate"
    assert plan["time_constraints"] == ["september"]
    assert plan["target_entities"] == [{"type": "PERSON", "name": "Caroline"}]
    assert "recommendation" in plan["relation_hints"]
    assert "recipes" in plan["must_have_terms"]


def test_query_plan_marks_indirect_recommendation_lists():
    plan = _mag_build_query_plan(
        "What book recommendations has Joanna given to Nate?",
        [("PROPER", "Joanna"), ("PROPER", "Nate")],
    )

    assert plan["answer_shape"] == "list"
    assert plan["evidence_mode"] == "aggregate"
    assert "recommendation" in plan["relation_hints"]


def test_bfs_shadow_gate_flags_unsupported_graph_only_candidate():
    query = "What book recommendations has Joanna given to Nate?"
    plan = _mag_build_query_plan(query, [("PROPER", "Joanna"), ("PROPER", "Nate")])
    gate = _mag_bfs_shadow_gate(
        query,
        plan,
        {
            "memory": "A weather forecast mentioned storms near Chicago.",
            "entities": [{"name": "Chicago"}],
            "route_scores": {"path_len": 2},
        },
        validator_backed=False,
    )

    assert gate["would_block"] is True
    assert "missing_target_entity" in gate["reasons"]
    assert "low_query_coverage" in gate["reasons"]


def test_bfs_shadow_gate_never_blocks_validator_backed_candidate():
    query = "When did Evan start lifting weights?"
    plan = _mag_build_query_plan(query, [("PROPER", "Evan")])
    gate = _mag_bfs_shadow_gate(
        query,
        plan,
        {
            "memory": "Evan started lifting weights in June 2023.",
            "entities": [{"name": "Evan"}],
            "route_scores": {"path_len": 3},
        },
        validator_backed=True,
    )

    assert gate["would_block"] is False
    assert gate["reasons"] == ["validator_backed"]


def test_time_constraints_match_month_names_against_iso_dates():
    assert _mag_time_constraints_match(["may", "2023"], "Evan took a road trip.", "2023-05-24T00:00:00")
    assert not _mag_time_constraints_match(["may", "2023"], "Evan took a road trip.", "2023-08-24T00:00:00")
    assert not _mag_time_constraints_match(["may", "2023"], "Maybe Evan took a road trip.", "2023-08-24T00:00:00")


def test_diverse_topk_keeps_high_score_but_reduces_duplicate_cluster():
    candidates = [
        {"id": "a", "memory": "John likes basketball jerseys and basketball shoes.", "score": 1.0},
        {"id": "b", "memory": "John likes basketball jerseys and basketball shoes a lot.", "score": 0.99},
        {"id": "c", "memory": "Tim owns a signed basketball collectible.", "score": 0.86},
    ]

    selected = _mag_diverse_topk(candidates, limit=2, pool_size=3)

    assert [item["id"] for item in selected] == ["a", "c"]


def test_diverse_topk_deduplicates_candidate_ids():
    candidates = [
        {"id": "a", "memory": "John likes basketball jerseys.", "score": 1.0},
        {"id": "a", "memory": "John likes basketball jerseys duplicate.", "score": 0.99},
        {"id": "b", "memory": "Tim owns a signed basketball.", "score": 0.9},
    ]

    selected = _mag_diverse_topk(candidates, limit=3, pool_size=3)

    assert [item["id"] for item in selected] == ["a", "b"]


def test_final_context_deduplicates_path_and_support_sentence_ids():
    final = [
        {
            "id": "s2",
            "memory": "path memory",
            "created_at": "2023-01-02T00:00:00",
            "context_segments": [
                {"id": "s1", "memory": "Alice visited Rome.", "created_at": "2023-01-01T00:00:00"},
                {"id": "s2", "memory": "Alice bought pasta.", "created_at": "2023-01-02T00:00:00"},
            ],
            "supporting_context": [
                {"id": "s3", "direction": "next", "memory": "Alice cooked pasta.", "created_at": "2023-01-03T00:00:00"},
            ],
        },
        {
            "id": "s3",
            "memory": "Alice cooked pasta.",
            "created_at": "2023-01-03T00:00:00",
        },
        {
            "id": "s4",
            "memory": "Bob likes tea.",
            "created_at": "2023-01-04T00:00:00",
            "supporting_context": [
                {"id": "s1", "direction": "prev", "memory": "Alice visited Rome.", "created_at": "2023-01-01T00:00:00"},
            ],
        },
    ]

    cleaned = _mag_dedupe_final_context(final)

    assert [item["id"] for item in cleaned] == ["s2", "s4"]
    joined = "\n".join(item["memory"] for item in cleaned)
    assert joined.count("Alice visited Rome.") == 1
    assert joined.count("Alice cooked pasta.") == 1
    assert "supporting_context" not in cleaned[1]


def test_add_source_deduplicates_route_labels():
    assert _mag_add_source("vector+bm25_validator+graph_bfs", "graph_bfs") == "vector+bm25_validator+graph_bfs"
    assert _mag_add_source("vector+bm25_validator", "ctx_boost") == "vector+bm25_validator+ctx_boost"
    assert _mag_add_source("vector+bm25_fallback", "") == "vector+bm25_fallback"


def test_format_search_results_preserves_mag_metadata():
    formatted, query_debug = format_search_results({
        "query_debug": {"routes": {"graph_candidates": 2}},
        "results": [{
            "id": "m1",
            "memory": "Caroline went to a support group.",
            "score": 0.9,
            "source": "vector+bm25_validator+graph_bfs",
            "metadata": {"route_scores": {"graph": 0.8}},
            "user_id": "u1",
        }],
    })

    assert query_debug == {"routes": {"graph_candidates": 2}}
    assert formatted[0]["metadata"]["route_scores"]["graph"] == 0.8
    assert formatted[0]["user_id"] == "u1"
