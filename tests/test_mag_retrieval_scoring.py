from mag.core import _mag_candidate_evidence_features, _mag_diverse_topk


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
