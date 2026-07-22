import json

from mag.graph.extraction import DirectTripleExtractor


class FakeLLM:
    def __init__(self, payload):
        self.payload = payload

    def generate_response(self, **kwargs):
        return json.dumps(self.payload)


def test_direct_triple_extraction_uses_llm_source_sentence_ids():
    detector = DirectTripleExtractor(
        llm_client=FakeLLM({
            "triples": [
                {
                    "head": "Alice",
                    "relation": "painted",
                    "tail": "sunsets",
                    "source_sentence_id": "s1",
                    "confidence": 0.9,
                },
                {
                    "head": "Alice",
                    "relation": "likes",
                    "tail": "piano",
                    "source_sentence_id": "unknown",
                    "confidence": 0.9,
                },
            ]
        })
    )

    triples = detector.extract_triples_direct([("s1", "Alice painted sunsets.")])

    assert len(triples) == 1
    assert triples[0].head == "Alice"
    assert triples[0].relation == "painted"
    assert triples[0].tail == "sunsets"
    assert triples[0].source_sentence_id == "s1"
