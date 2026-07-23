from mag.core import MAGMemory
from mag.schema import ExtractedFact, Triple


class FakeEmbeddingModel:
    def embed_batch(self, texts, action):
        return [[float(len(text))] for text in texts]

    def embed(self, text, action):
        return [float(len(text))]


class FakeRecord:
    def __init__(self, payload):
        self.payload = payload


class FakeVectorStore:
    def __init__(self):
        self.inserted = []
        self.updated = []
        self.records = {}
        self.source_payload = {
            "data": "Alice painted sunsets. She loved it.",
            "created_at": "2023-07-01T00:00:00+00:00",
            "updated_at": "2023-07-01T00:00:00+00:00",
            "speaker": "user",
            "user_id": "u1",
            "source_raw_sentence_ids": ["s0", "s1"],
            "source_raw_texts": ["Alice painted sunsets.", "She loved it."],
        }

    def get(self, vector_id):
        if vector_id == "unit_1":
            return FakeRecord(self.source_payload)
        if vector_id in self.records:
            return FakeRecord(self.records[vector_id])
        return None

    def insert(self, *, vectors, ids, payloads):
        self.inserted.append({"vectors": vectors, "ids": ids, "payloads": payloads})
        for vector_id, payload in zip(ids, payloads):
            self.records[vector_id] = payload

    def update(self, *, vector_id, vector, payload):
        self.updated.append({"vector_id": vector_id, "vector": vector, "payload": payload})
        self.records[vector_id] = payload


class FakeDB:
    def __init__(self):
        self.history = []

    def add_history(self, *args, **kwargs):
        self.history.append((args, kwargs))


def test_mag_indexes_extracted_facts_as_graphiti_style_memories():
    memory = object.__new__(MAGMemory)
    memory.mag_index_graph_facts = True
    memory.mag_index_entity_summaries = True
    memory.mag_use_history = True
    memory.embedding_model = FakeEmbeddingModel()
    memory.vector_store = FakeVectorStore()
    memory.db = FakeDB()
    memory._mag_sentence_scopes = {"unit_1": "user_id=u1"}

    fact_ids = memory._mag_index_graph_fact_memories(
        [
            ExtractedFact(
                fact_id="f1",
                fact="Alice painted sunsets.",
                source_sentence_id="unit_1",
                confidence=0.9,
            )
        ],
        [
            Triple(
                head="Alice",
                relation="painted",
                tail="sunsets",
                source_sentence_id="unit_1",
                source_fact_id="f1",
                source_fact="Alice painted sunsets.",
                confidence=0.9,
            )
        ],
        {"user_id": "u1"},
        "user_id=u1",
    )

    fact_memory_id = fact_ids[("unit_1", "f1")]
    payload = memory.vector_store.inserted[0]["payloads"][0]
    entity_payloads = [
        item["payloads"][0]
        for item in memory.vector_store.inserted[1:]
        if item["payloads"][0].get("graph_object") == "entity_node"
    ]

    assert payload["graph_object"] == "edge_fact"
    assert payload["data"] == "Alice painted sunsets."
    assert payload["source_unit_id"] == "unit_1"
    assert payload["valid_at"] == "2023-07-01T00:00:00+00:00"
    assert payload["invalid_at"] == ""
    assert payload["triples"] == [{"head": "Alice", "relation": "painted", "tail": "sunsets"}]
    assert payload["user_id"] == "u1"
    assert memory._mag_sentence_scopes[fact_memory_id] == "user_id=u1"
    assert {payload["entity_name"] for payload in entity_payloads} == {"Alice", "sunsets"}
    assert all("Alice painted sunsets." in payload["entity_summary"] for payload in entity_payloads)


def test_mag_entity_summary_memory_updates_existing_node():
    memory = object.__new__(MAGMemory)
    memory.mag_index_entity_summaries = True
    memory.embedding_model = FakeEmbeddingModel()
    memory.vector_store = FakeVectorStore()
    memory._mag_sentence_scopes = {}

    source_payload = {"created_at": "2023-07-01T00:00:00+00:00", "user_id": "u1"}
    memory._mag_upsert_entity_summary_memory(
        "Alice",
        "Alice painted sunsets.",
        "fact_1",
        "2023-07-01T00:00:00+00:00",
        source_payload,
        "user_id=u1",
    )
    memory._mag_upsert_entity_summary_memory(
        "Alice",
        "Alice bought a camera.",
        "fact_2",
        "2023-07-02T00:00:00+00:00",
        source_payload,
        "user_id=u1",
    )

    assert len(memory.vector_store.updated) == 1
    payload = memory.vector_store.updated[0]["payload"]
    assert payload["graph_object"] == "entity_node"
    assert payload["summary_facts"] == ["Alice painted sunsets.", "Alice bought a camera."]
    assert payload["source_fact_memory_ids"] == ["fact_1", "fact_2"]
