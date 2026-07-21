from benchmarks.common.mem0_client import format_search_results
from mag.core import _mag_add_source, _mag_candidate_evidence_features, _mag_diverse_topk


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


def test_diverse_topk_keeps_high_score_but_reduces_duplicate_cluster():
    candidates = [
        {"id": "a", "memory": "John likes basketball jerseys and basketball shoes.", "score": 1.0},
        {"id": "b", "memory": "John likes basketball jerseys and basketball shoes a lot.", "score": 0.99},
        {"id": "c", "memory": "Tim owns a signed basketball collectible.", "score": 0.86},
    ]

    selected = _mag_diverse_topk(candidates, limit=2, pool_size=3)

    assert [item["id"] for item in selected] == ["a", "c"]


def test_add_source_deduplicates_route_labels():
    assert _mag_add_source("vector+bm25_validator+graph_bfs", "graph_bfs") == "vector+bm25_validator+graph_bfs"
    assert _mag_add_source("vector+bm25_validator", "ctx_boost") == "vector+bm25_validator+ctx_boost"


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
