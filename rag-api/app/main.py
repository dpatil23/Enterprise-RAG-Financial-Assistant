from fastapi import FastAPI
from app.api.routes import ingest, query

app = FastAPI(
    title="Enterprise Hybrid GraphRAG API",
    description="""
## 🚀 Enterprise Hybrid Graph + Vector RAG API

Upload financial filings (SEC 10-K, 10-Q, annual reports) and query them using an integrated Knowledge Graph and Vector Search engine.

### Core Capabilities:
1. **POST /api/v1/upload** — Ingest financial PDF. Extracts text into FAISS vector store and parses structured entities & relations into Neo4j with provenance tracking.
2. **POST /api/v1/ask** — Multi-modal query answering using Query Router (`vector`, `graph`, `both`) with per-claim citation validation.
3. **GET /health** — Service health monitoring.

### Tech Stack:
- **Knowledge Graph:** Neo4j Community (Cypher Graph Database)
- **Vector DB:** FAISS (Facebook AI Similarity Search)
- **Embeddings:** `all-MiniLM-L6-v2` (sentence-transformers)
- **LLM Reasoning:** LLaMA 3.3 via Groq
- **API Framework:** FastAPI
    """,
    version="2.0.0",
)

# Register routers
app.include_router(ingest.router, prefix="/api/v1", tags=["Ingestion"])
app.include_router(query.router, prefix="/api/v1", tags=["Query"])


@app.get("/health", tags=["Health"])
def health_check():
    """Health check endpoint for monitoring and CI/CD pipelines."""
    return {"status": "healthy", "version": "2.0.0"}


@app.get("/", tags=["Root"])
def root():
    return {
        "message": "Enterprise Hybrid GraphRAG API is running!",
        "docs": "/docs",
        "health": "/health",
    }
