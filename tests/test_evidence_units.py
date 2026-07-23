import json
from datetime import datetime, timezone

from mag.evidence_units import EvidenceUnitBuilder


class FakeLLM:
    def __init__(self, payload):
        self.payload = payload

    def generate_response(self, **kwargs):
        return json.dumps(self.payload)


def test_evidence_unit_builder_merges_adjacent_source_bound_sentences():
    builder = EvidenceUnitBuilder(
        llm_client=FakeLLM(
            {
                "units": [
                    {
                        "source_sentence_ids": ["s0", "s1"],
                        "text": "Alice signed up for the pottery class. Alice is excited about the pottery class.",
                        "resolved_references": [
                            {"expression": "She", "referent": "Alice"},
                            {"expression": "it", "referent": "the pottery class"},
                        ],
                        "merge_reason": "second sentence depends on the first sentence",
                        "confidence": 0.9,
                    }
                ]
            }
        )
    )

    timestamp = datetime(2023, 7, 1, tzinfo=timezone.utc)
    units = builder.build(
        [
            ("Alice signed up for the pottery class.", "user", timestamp),
            ("She is excited about it.", "user", timestamp),
        ]
    )

    assert len(units) == 1
    assert units[0].source_sentence_ids == ["s0", "s1"]
    assert units[0].source_texts == [
        "Alice signed up for the pottery class.",
        "She is excited about it.",
    ]
    assert units[0].speaker == "user"


def test_evidence_unit_builder_rejects_non_adjacent_merge_and_preserves_inputs():
    builder = EvidenceUnitBuilder(
        llm_client=FakeLLM(
            {
                "units": [
                    {
                        "source_sentence_ids": ["s0", "s2"],
                        "text": "Alice likes pottery. Alice likes jazz.",
                        "confidence": 0.9,
                    }
                ]
            }
        )
    )

    timestamp = datetime(2023, 7, 1, tzinfo=timezone.utc)
    units = builder.build(
        [
            ("Alice likes pottery.", "user", timestamp),
            ("Bob likes chess.", "assistant", timestamp),
            ("Alice likes jazz.", "user", timestamp),
        ]
    )

    assert [unit.text for unit in units] == [
        "Alice likes pottery.",
        "Bob likes chess.",
        "Alice likes jazz.",
    ]
    assert [unit.source_sentence_ids for unit in units] == [["s0"], ["s1"], ["s2"]]


def test_evidence_unit_builder_falls_back_for_uncovered_sentence():
    builder = EvidenceUnitBuilder(
        llm_client=FakeLLM(
            {
                "units": [
                    {
                        "source_sentence_ids": ["s0"],
                        "text": "Alice likes pottery.",
                        "confidence": 0.9,
                    }
                ]
            }
        )
    )

    timestamp = datetime(2023, 7, 1, tzinfo=timezone.utc)
    units = builder.build(
        [
            ("Alice likes pottery.", "user", timestamp),
            ("Bob likes chess.", "assistant", timestamp),
        ]
    )

    assert [unit.text for unit in units] == ["Alice likes pottery.", "Bob likes chess."]
    assert [unit.source_sentence_ids for unit in units] == [["s0"], ["s1"]]
