from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_check():
    """The /health endpoint should always return 200 with status healthy."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_root():
    """The root endpoint should return a welcome message."""
    response = client.get("/")
    assert response.status_code == 200
    assert "docs" in response.json()


def test_ask_with_no_documents():
    """Asking a question before uploading any docs should return a graceful message."""
    response = client.post("/api/v1/ask", json={"question": "What is the revenue?"})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "answer" in data["data"]
    assert "route" in data["data"]


def test_ask_with_force_route():
    """Asking a question with forced route should obey the override."""
    response = client.post("/api/v1/ask", json={
        "question": "Which subsidiaries supply components to Samsung?",
        "force_route": "graph"
    })
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["data"]["route"] == "graph"


def test_upload_non_pdf_rejected():
    """Uploading a non-PDF file should return a 400 error."""
    response = client.post(
        "/api/v1/upload",
        files={"file": ("test.txt", b"some text content", "text/plain")},
    )
    assert response.status_code == 400
    assert "PDF" in response.json()["detail"]


def test_ask_empty_question_rejected():
    """Asking an empty question should return a 400 error."""
    response = client.post("/api/v1/ask", json={"question": "   "})
    assert response.status_code == 400
