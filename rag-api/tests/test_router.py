from app.services.query_router import QueryRouter


def test_query_router_heuristics():
    router = QueryRouter()

    # Graph questions
    r1 = router._heuristic_classification("Which subsidiaries of Apple supply Samsung?")
    assert r1["route"] == "graph"

    r2 = router._heuristic_classification("Who is the CEO and board chairman?")
    assert r2["route"] == "graph"

    # Vector questions
    r3 = router._heuristic_classification("What was the reported net income for 2023?")
    assert r3["route"] == "vector"

    # Hybrid questions
    r4 = router._heuristic_classification("How does the subsidiary supply chain impact reported operating margin?")
    assert r4["route"] == "both"


def test_query_router_classify_query():
    router = QueryRouter()
    res = router.classify_query("Who is the CEO of Apple Inc.?")
    assert "route" in res
    assert res["route"] in ["vector", "graph", "both"]
    assert "duration_ms" in res
