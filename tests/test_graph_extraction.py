import json

from mag.graph.extraction import DirectTripleExtractor


class FakeLLM:
    def __init__(self, payload):
        self.payload = payload

    def generate_response(self, **kwargs):
        return json.dumps(self.payload)


class SequenceLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def generate_response(self, **kwargs):
        self.calls.append(kwargs)
        response = self.responses.pop(0)
        if isinstance(response, str):
            return response
        return json.dumps(response)


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


def test_direct_triple_extraction_uses_source_facts():
    detector = DirectTripleExtractor(
        llm_client=FakeLLM({
            "facts": [
                {
                    "fact_id": "f1",
                    "source_sentence_id": "s1",
                    "fact": "Alice painted sunsets.",
                    "confidence": 0.9,
                }
            ],
            "triples": [
                {
                    "head": "Alice",
                    "relation": "painted",
                    "tail": "sunsets",
                    "source_fact_id": "f1",
                    "source_sentence_id": "s1",
                    "confidence": 0.9,
                },
                {
                    "head": "Alice",
                    "relation": "likes",
                    "tail": "piano",
                    "source_fact_id": "missing",
                    "source_sentence_id": "s1",
                    "confidence": 0.9,
                },
            ],
        })
    )

    triples = detector.extract_triples_direct([("s1", "Alice painted sunsets.")])

    assert len(detector.last_extracted_facts) == 1
    assert detector.last_extracted_facts[0].fact_id == "f1"
    assert detector.last_extracted_facts[0].fact == "Alice painted sunsets."
    assert len(triples) == 1
    assert triples[0].source_fact_id == "f1"
    assert triples[0].source_fact == "Alice painted sunsets."


def test_direct_triple_extraction_uses_context_without_extracting_context_facts():
    detector = DirectTripleExtractor(
        llm_client=FakeLLM({
            "facts": [
                {
                    "fact_id": "f_ctx",
                    "source_sentence_id": "ctx1",
                    "fact": "Caroline asked which concert Melanie attended.",
                    "confidence": 0.9,
                },
                {
                    "fact_id": "f1",
                    "source_sentence_id": "s1",
                    "source_sentence_ids": ["ctx0", "ctx1", "s1"],
                    "fact": "Melanie attended a concert featuring Matt Patterson.",
                    "confidence": 0.9,
                },
            ],
            "triples": [
                {
                    "head": "Melanie",
                    "relation": "attended concert featuring",
                    "tail": "Matt Patterson",
                    "source_fact_id": "f1",
                    "source_sentence_id": "s1",
                    "source_sentence_ids": ["ctx0", "ctx1", "s1"],
                    "confidence": 0.9,
                },
                {
                    "head": "Caroline",
                    "relation": "asked about",
                    "tail": "concert",
                    "source_fact_id": "f_ctx",
                    "source_sentence_id": "ctx1",
                    "confidence": 0.9,
                },
            ],
        })
    )

    triples = detector.extract_triples_direct(
        [("s1", "Melanie: It was Matt Patterson.")],
        context_items=[
            ("ctx0", "Melanie: I celebrated my daughter's birthday with a concert."),
            ("ctx1", "Caroline: What concert was it?"),
        ],
    )

    assert len(detector.last_extracted_facts) == 1
    assert detector.last_extracted_facts[0].source_sentence_id == "s1"
    assert detector.last_extracted_facts[0].source_sentence_ids == ["ctx0", "ctx1", "s1"]
    assert len(triples) == 1
    assert triples[0].head == "Melanie"
    assert triples[0].tail == "Matt Patterson"
    assert triples[0].source_sentence_ids == ["ctx0", "ctx1", "s1"]


def test_direct_triple_extraction_retries_truncated_batch_by_splitting():
    detector = DirectTripleExtractor(
        llm_client=SequenceLLM(
            [
                '{"facts": [{"fact_id": "f1", "fact": "unterminated',
                {
                    "facts": [
                        {
                            "fact_id": "f1",
                            "source_sentence_id": "s1",
                            "fact": "Alice painted sunsets.",
                            "confidence": 0.9,
                        }
                    ],
                    "triples": [
                        {
                            "head": "Alice",
                            "relation": "painted",
                            "tail": "sunsets",
                            "source_fact_id": "f1",
                            "source_sentence_id": "s1",
                            "confidence": 0.9,
                        }
                    ],
                },
                {
                    "facts": [
                        {
                            "fact_id": "f1",
                            "source_sentence_id": "s2",
                            "fact": "Bob likes piano.",
                            "confidence": 0.8,
                        }
                    ],
                    "triples": [
                        {
                            "head": "Bob",
                            "relation": "likes",
                            "tail": "piano",
                            "source_fact_id": "f1",
                            "source_sentence_id": "s2",
                            "confidence": 0.8,
                        }
                    ],
                },
            ]
        )
    )

    triples = detector.extract_triples_direct(
        [("s1", "Alice painted sunsets."), ("s2", "Bob likes piano.")]
    )

    assert [triple.source_sentence_id for triple in triples] == ["s1", "s2"]
    assert len(detector.last_extracted_facts) == 2
    assert len(detector.llm_client.calls) == 3
    assert detector.llm_client.calls[0]["max_tokens"] >= 8192
