from mag.graph.bfs_search import BFSRetriever
from mag.graph.store import GraphStore
from mag.schema import EntityWeight


def test_graph_store_filters_entity_links_and_edges_by_session_scope():
    graph = GraphStore()

    graph.upsert_entity("Alice", 0.8, "s_alice_a", "PERSON", session_scope="user_id=a")
    graph.upsert_entity("Bob", 0.7, "s_bob_a", "PERSON", session_scope="user_id=a")
    graph.upsert_entity("Carol", 0.7, "s_carol_b", "PERSON", session_scope="user_id=b")
    graph.add_relation("Alice", "knows", "Bob", "s_edge_a", session_scope="user_id=a")
    graph.add_relation("Alice", "knows", "Carol", "s_edge_b", session_scope="user_id=b")

    assert set(graph.get_sentences_for_entity("alice", session_scope="user_id=a")) == {"s_alice_a"}
    assert set(graph.get_sentences_for_entity("alice", session_scope="user_id=b")) == set()
    assert set(graph.get_neighbors_scoped("alice", session_scope="user_id=a")) == {"bob"}
    assert set(graph.get_neighbors_scoped("alice", session_scope="user_id=b")) == {"carol"}


def test_bfs_search_stays_inside_requested_session_scope():
    graph = GraphStore()

    graph.upsert_entity("Alice", 0.8, "s_alice_a", "PERSON", session_scope="user_id=a")
    graph.upsert_entity("Bob", 0.7, "s_bob_a", "PERSON", session_scope="user_id=a")
    graph.upsert_entity("Carol", 0.7, "s_carol_b", "PERSON", session_scope="user_id=b")
    graph.add_relation("Alice", "knows", "Bob", "s_edge_a", session_scope="user_id=a")
    graph.add_relation("Alice", "knows", "Carol", "s_edge_b", session_scope="user_id=b")

    bfs = BFSRetriever(graph)
    results = bfs.search(
        [EntityWeight(name="alice", attention_weight=1.0, entity_type="PERSON")],
        max_hops=1,
        max_results=10,
        session_scope="user_id=a",
    )

    result_ids = {sid for sid, _ in results}
    assert "s_alice_a" in result_ids
    assert "s_bob_a" in result_ids
    assert "s_carol_b" not in result_ids


def test_bfs_search_uses_multiple_edges_between_same_nodes():
    graph = GraphStore()

    graph.upsert_entity("Alice", 0.8, "s1", "PERSON", session_scope="user_id=a")
    graph.upsert_entity("piano", 0.7, "s1", "OBJECT", session_scope="user_id=a")
    graph.upsert_entity("Alice", 0.8, "s2", "PERSON", session_scope="user_id=a")
    graph.upsert_entity("piano", 0.7, "s2", "OBJECT", session_scope="user_id=a")
    graph.add_relation("Alice", "plays", "piano", "s1", confidence=0.9, session_scope="user_id=a")
    graph.add_relation("Alice", "bought", "piano", "s2", confidence=0.9, session_scope="user_id=a")

    bfs = BFSRetriever(graph)
    results = bfs.search(
        [EntityWeight(name="alice", attention_weight=1.0, entity_type="PERSON")],
        max_hops=1,
        max_results=10,
        session_scope="user_id=a",
    )

    assert {"s1", "s2"} <= {sid for sid, _ in results}


def test_path_search_reads_reverse_multiedges_with_distinct_sources():
    graph = GraphStore()

    graph.add_relation("Alice", "plays", "piano", "s1", confidence=0.9, session_scope="user_id=a")
    graph.add_relation("Alice", "bought", "piano", "s2", confidence=0.9, session_scope="user_id=a")

    bfs = BFSRetriever(graph)
    paths = bfs.search_paths(
        [EntityWeight(name="piano", attention_weight=1.0, entity_type="OBJECT")],
        query_embedding=None,
        get_semantic_sim=lambda _: 1.0,
        max_hops=1,
        max_results=10,
        session_scope="user_id=a",
    )

    source_ids = {sid for path in paths for sid in path["sentences"]}
    assert {"s1", "s2"} <= source_ids


def test_graph_store_preserves_source_fact_metadata():
    graph = GraphStore()
    graph.add_relation(
        "Alice",
        "painted",
        "sunsets",
        "s1",
        confidence=0.9,
        session_scope="user_id=a",
        source_fact_id="f1",
        source_fact="Alice painted sunsets.",
    )

    relations = graph.get_all_relations_for_entity("alice", session_scope="user_id=a")

    assert len(relations) == 1
    assert relations[0]["source_fact_id"] == "f1"
    assert relations[0]["source_fact"] == "Alice painted sunsets."


def test_graph_store_preserves_graphiti_fact_memory_ids():
    graph = GraphStore()
    graph.add_relation(
        "Alice",
        "painted",
        "sunsets",
        "fact_1",
        confidence=0.9,
        session_scope="user_id=a",
        source_fact_id="f1",
        source_fact="Alice painted sunsets.",
        source_unit_id="unit_1",
        source_fact_memory_id="fact_1",
    )

    relations = graph.get_all_relations_for_entity("alice", session_scope="user_id=a")

    assert relations[0]["source_sentence_ids"] == ["fact_1"]
    assert relations[0]["source_fact_memory_ids"] == ["fact_1"]
    assert relations[0]["source_unit_id"] == "unit_1"


def test_path_search_prefers_fact_memory_ids_on_edges():
    graph = GraphStore()
    graph.add_relation(
        "Alice",
        "painted",
        "sunsets",
        "unit_1",
        confidence=0.9,
        session_scope="user_id=a",
        source_fact_memory_id="fact_1",
    )

    bfs = BFSRetriever(graph)
    paths = bfs.search_paths(
        [EntityWeight(name="alice", attention_weight=1.0, entity_type="PERSON")],
        query_embedding=None,
        get_semantic_sim=lambda identifier: 1.0 if identifier == "Alice painted sunsets" else 0.0,
        max_hops=1,
        max_results=10,
        session_scope="user_id=a",
    )

    source_ids = {sid for path in paths for sid in path["sentences"]}
    assert source_ids == {"fact_1"}
