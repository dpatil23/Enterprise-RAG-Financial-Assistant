from typing import Optional, Literal
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from app.services.query_service import answer_question

router = APIRouter()


class QueryRequest(BaseModel):
    question: str = Field(..., description="Natural language question about uploaded financial filings")
    force_route: Optional[Literal["vector", "graph", "both"]] = Field(
        None, description="Optional manual override of retrieval strategy"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"question": "What is the total revenue reported in the filing?"},
                {"question": "Which subsidiaries of Apple supply components to Samsung?", "force_route": "graph"},
                {"question": "How does the supply chain operate and what risks are listed?", "force_route": "both"}
            ]
        }
    }


@router.post("/ask", summary="Ask a question via Hybrid Graph + Vector RAG")
def ask_question(request: QueryRequest):
    """
    Query the Hybrid RAG engine.
    
    1. **Query Router:** Intelligently classifies questions into `vector`, `graph`, or `both`.
    2. **Multi-Hop Traversal:** Navigates Neo4j knowledge graph for corporate structures & executive connections.
    3. **Vector Similarity:** Searches FAISS dense index for precise document paragraphs and financial metrics.
    4. **Citation Validation:** Validates chunk IDs and provenance for each claim.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        result = answer_question(
            question=request.question,
            force_route=request.force_route,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

    return {"status": "success", "data": result}
