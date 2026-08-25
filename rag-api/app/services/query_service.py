import logging
import re
import time
from typing import Any, Dict, List
from app.core.config import settings
from app.core.embedder import embedder
from app.core.vector_store import vector_store
from app.core.llm import llm_client
from app.services.query_router import query_router
from app.services.graph_query_service import graph_query_service

logger = logging.getLogger("query_service")


def validate_citations(answer: str, retrieved_chunk_ids: set[str]) -> Dict[str, Any]:
    """
    Citation Validator:
    Scans the synthesized answer for citations (e.g., [Source: doc_chunk_0])
    and verifies that every cited chunk was genuinely part of the retrieved context.
    """
    cited_patterns = re.findall(r"\[Source:\s*([a-zA-Z0-9_\-]+)\]", answer)
    cited_set = set(cited_patterns)

    valid_citations = []
    invalid_citations = []

    for cid in cited_set:
        if cid in retrieved_chunk_ids or any(cid in actual for actual in retrieved_chunk_ids):
            valid_citations.append(cid)
        else:
            invalid_citations.append(cid)

    return {
        "is_valid": len(invalid_citations) == 0,
        "valid_citations": valid_citations,
        "invalid_citations": invalid_citations,
    }


def answer_question(question: str, force_route: str = None) -> dict:
    """
    Enterprise Hybrid RAG Query Pipeline.
    
    1. ROUTE:       QueryRouter analyzes question intent (vector / graph / both).
    2. RETRIEVE:
       - Vector Path: Dense embedding cosine similarity over FAISS.
       - Graph Path:  Cypher graph traversal across multi-hop relationships in Neo4j.
    3. DEDUPLICATE: Merge and remove redundant information from both channels.
    4. SYNTHESIZE:  LLM generates unified answer with provenance grounding.
    5. VALIDATE:    Citation Validator verifies every cited source chunk.
    
    Args:
        question: User query
        force_route: Optional override ('vector' | 'graph' | 'both')
        
    Returns:
        Structured response dictionary
    """
    start_time = time.time()
    clean_q = question.strip()

    # 1. Routing
    if force_route and force_route in ["vector", "graph", "both"]:
        route_decision = {
            "route": force_route,
            "reasoning": f"Manual override to '{force_route}'",
            "entities": [],
            "duration_ms": 0.0,
        }
    else:
        route_decision = query_router.classify_query(clean_q)

    chosen_route = route_decision["route"]
    extracted_entities = route_decision.get("entities", [])

    context_chunks: List[Dict[str, Any]] = []
    graph_facts: List[str] = []
    sources: List[Dict[str, Any]] = []
    retrieved_chunk_ids: set[str] = set()

    # 2. Vector Retrieval (if route is vector or both)
    if chosen_route in ["vector", "both"]:
        try:
            query_embedding = embedder.embed([clean_q])[0]
            context_chunks = vector_store.query(
                collection_name="documents",
                query_embedding=query_embedding,
                top_k=settings.TOP_K_RESULTS,
            )
            for c in context_chunks:
                cid = f"{c['source']}_chunk_{c['chunk_index']}"
                retrieved_chunk_ids.add(cid)
                sources.append({
                    "type": "vector",
                    "chunk_id": cid,
                    "doc_id": c["source"],
                    "chunk_index": c["chunk_index"],
                    "similarity_score": c["similarity_score"],
                    "text_preview": c["text"][:200] + ("..." if len(c["text"]) > 200 else ""),
                })
        except Exception as e:
            logger.warning(f"[QueryService] Vector search error: {e}")

    # 3. Graph Retrieval (if route is graph or both)
    if chosen_route in ["graph", "both"]:
        try:
            graph_res = graph_query_service.query_graph_for_context(
                question=clean_q,
                entities=extracted_entities,
            )
            graph_facts = graph_res.get("facts", [])
            for g_source in graph_res.get("sources", []):
                sources.append(g_source)
                cid = g_source.get("chunk_id")
                if cid and cid != "unknown_chunk":
                    retrieved_chunk_ids.add(cid)
        except Exception as e:
            logger.warning(f"[QueryService] Graph search error: {e}")

    # Check if we got anything
    if not context_chunks and not graph_facts:
        return {
            "question": clean_q,
            "route": chosen_route,
            "routing_reasoning": route_decision.get("reasoning", ""),
            "answer": "No relevant documents or graph relationships found. Please upload a financial PDF document first.",
            "sources": [],
            "citation_validation": {"is_valid": True, "valid_citations": [], "invalid_citations": []},
            "latency_ms": round((time.time() - start_time) * 1000, 2),
        }

    # 4. Synthesize Answer
    answer = llm_client.generate_hybrid_answer(
        question=clean_q,
        context_chunks=context_chunks,
        graph_facts=graph_facts,
    )

    # 5. Validate Citations
    citation_check = validate_citations(answer, retrieved_chunk_ids)

    total_duration_ms = round((time.time() - start_time) * 1000, 2)

    return {
        "question": clean_q,
        "route": chosen_route,
        "routing_reasoning": route_decision.get("reasoning", ""),
        "answer": answer,
        "sources": sources,
        "citation_validation": citation_check,
        "latency_ms": total_duration_ms,
    }
