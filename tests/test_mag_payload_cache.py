from mag.core import MAGMemory


class FakePoint:
    def __init__(self, point_id, payload):
        self.id = point_id
        self.payload = payload


class FakeQdrantClient:
    def __init__(self):
        self.calls = []

    def retrieve(self, *, collection_name, ids, with_payload, with_vectors):
        self.calls.append(
            {
                "collection_name": collection_name,
                "ids": list(ids),
                "with_payload": with_payload,
                "with_vectors": with_vectors,
            }
        )
        payloads = {
            "s1": {"data": "kept", "user_id": "u1"},
            "s2": {"data": "wrong scope", "user_id": "u2"},
        }
        return [FakePoint(point_id, payloads[point_id]) for point_id in ids if point_id in payloads]


class FakeVectorStore:
    def __init__(self):
        self.client = FakeQdrantClient()

    def get(self, vector_id):
        raise AssertionError("single-id fallback should not be used when batch retrieve works")


def test_mag_prefetch_scoped_payloads_batches_dedupes_and_caches_misses():
    memory = object.__new__(MAGMemory)
    memory.vector_store = FakeVectorStore()
    memory.collection_name = "test_collection"
    payload_cache = {}

    memory._mag_prefetch_scoped_payloads(
        ["s1", "s1", "s2", "s3"],
        {"user_id": "u1"},
        payload_cache,
    )

    assert memory.vector_store.client.calls == [
        {
            "collection_name": "test_collection",
            "ids": ["s1", "s2", "s3"],
            "with_payload": True,
            "with_vectors": False,
        }
    ]
    assert payload_cache["s1"]["data"] == "kept"
    assert payload_cache["s2"] is None
    assert payload_cache["s3"] is None

    assert memory._mag_get_scoped_payload("s1", {"user_id": "u1"}, payload_cache)["data"] == "kept"
    assert len(memory.vector_store.client.calls) == 1
