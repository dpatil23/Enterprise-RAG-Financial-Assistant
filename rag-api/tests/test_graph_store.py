import pytest
from app.core.graph_store import GraphStore


@pytest.fixture(scope="module")
def store():
    gs = GraphStore()
    if not gs.verify_connectivity():
        pytest.skip("Neo4j database is not reachable at configured URI.")
    yield gs


def test_neo4j_connectivity(store):
    assert store.verify_connectivity() is True


def test_idempotent_ingestion_and_provenance(store):
    entities = [
        {"name": "Apple Inc.", "type": "Company", "description": "Tech company"},
        {"name": "Tim Cook", "type": "Person", "description": "CEO"},
        {"name": "Beats Electronics", "type": "Company", "description": "Audio hardware subsidiary"},
        {"name": "Foxconn", "type": "Company", "description": "Contract manufacturer"},
    ]
    relationships = [
        {"source": "Tim Cook", "target": "Apple Inc.", "type": "CEO_OF", "description": "Chief Executive Officer"},
        {"source": "Beats Electronics", "target": "Apple Inc.", "type": "SUBSIDIARY_OF", "description": "Wholly owned"},
        {"source": "Foxconn", "target": "Apple Inc.", "type": "SUPPLIES_TO", "description": "Assembly supplier"},
    ]

    # Ingest first time
    res1 = store.add_entities_and_relations(
        entities=entities,
        relationships=relationships,
        doc_id="apple_10k",
        chunk_id="apple_10k_chunk_0",
    )
    assert res1["nodes"] >= 4

    counts_after_first = store.count_nodes_and_edges()

    # Ingest second time (idempotency test)
    res2 = store.add_entities_and_relations(
        entities=entities,
        relationships=relationships,
        doc_id="apple_10k",
        chunk_id="apple_10k_chunk_0",
    )
    counts_after_second = store.count_nodes_and_edges()

    # Node and relationship count should remain identical!
    assert counts_after_second["node_count"] == counts_after_first["node_count"]
    assert counts_after_second["relationship_count"] == counts_after_first["relationship_count"]


def test_multi_hop_traversal(store):
    # Foxconn -> Apple <- Beats
    paths = store.find_multi_hop_paths("Foxconn", "Beats Electronics", max_hops=2)
    assert len(paths) > 0
    p = paths[0]
    assert "Apple Inc." in p["path_nodes"]
    assert p["hops"] == 2
