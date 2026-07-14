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
