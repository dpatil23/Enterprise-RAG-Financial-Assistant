import json
import logging
import re
import time
from typing import Any, Dict, Optional
from openai import OpenAI
from app.core.config import settings

logger = logging.getLogger("query_router")

ROUTER_SYSTEM_PROMPT = """You are an intelligent query routing engine for an enterprise financial RAG system.
Your job is to classify the user's question into one of three retrieval strategies:

1. "vector": For specific factual lookups, financial metric values, paragraph explanations, or direct statement lookups.
   Examples:
   - "What was the company's total revenue in 2023?"
   - "What are the primary risk factors described in Item 1A?"
   - "What accounting policies are used for revenue recognition?"

2. "graph": For multi-hop relationship queries, organizational hierarchies, corporate ownership, supply chains, executive affiliations, or cross-entity connections.
   Examples:
   - "Which subsidiaries of Apple supply Samsung?"
   - "Who is the CEO of Company X's largest subsidiary?"
   - "What suppliers connect Foxconn and Apple?"
   - "Are there any shared board members between Subsidiary A and Subsidiary B?"

3. "both": For complex thematic questions, comparative analyses, or queries requiring both structured relationships and textual details.
   Examples:
   - "How does Company X's supply chain ecosystem operate and what are the related financial costs?"
   - "Provide a breakdown of executive leadership and their associated compensation packages."
   - "Which competitors are listed and what market risks do they pose?"

Return JSON ONLY in this format:
{
  "route": "vector" | "graph" | "both",
  "reasoning": "Brief 1-sentence rationale for the routing decision",
  "entities": ["list", "of", "key", "entities", "identified", "in", "the", "question"]
}"""


class QueryRouter:
    """
    Classifies queries to direct them to Vector Retrieval, Graph Traversal, or Hybrid synthesis.
    """

    def __init__(self, client: Optional[OpenAI] = None):
        self._client = client
        self._init_client()
        self.routing_logs: list[dict] = []

    def _init_client(self) -> None:
        api_key = settings.LLM_API_KEY or settings.GROQ_API_KEY
        base_url = settings.LLM_BASE_URL
        if settings.GROQ_API_KEY and not settings.LLM_API_KEY:
            base_url = "https://api.groq.com/openai/v1"
            api_key = settings.GROQ_API_KEY

        if api_key:
            try:
                self._client = OpenAI(base_url=base_url, api_key=api_key)
            except Exception as e:
                logger.warning(f"[QueryRouter] Client init failed: {e}")

    def _get_client(self) -> Optional[OpenAI]:
        if self._client is None and (settings.LLM_API_KEY or settings.GROQ_API_KEY):
            self._init_client()
        return self._client

    def _get_model(self) -> str:
        if settings.LLM_API_KEY:
            return settings.LLM_MODEL
        if settings.GROQ_API_KEY:
            return settings.GROQ_MODEL
        return settings.LLM_MODEL

    def classify_query(self, question: str) -> Dict[str, Any]:
        """
        Classify question into 'vector', 'graph', or 'both', and extract key entities.
        """
        start_time = time.time()
        clean_q = question.strip()

        client = self._get_client()

        # Fallback heuristic if LLM client is unavailable
        if client is None:
            route = self._heuristic_classification(clean_q)
            duration_ms = round((time.time() - start_time) * 1000, 2)
            log_entry = {
                "question": clean_q,
                "route": route["route"],
                "reasoning": route["reasoning"],
                "entities": route["entities"],
                "duration_ms": duration_ms,
            }
            self.routing_logs.append(log_entry)
            return log_entry

        try:
            response = client.chat.completions.create(
                model=self._get_model(),
                messages=[
                    {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
                    {"role": "user", "content": f"Classify this question into valid JSON format:\n{clean_q}"},
                ],
                temperature=0.0,
                max_tokens=500,
            )
            content = response.choices[0].message.content.strip()
            # Clean thinking tags or code fences if present
            content_clean = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
            match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content_clean, flags=re.DOTALL)
            if match:
                json_str = match.group(1)
            else:
                brace_match = re.search(r"(\{.*\})", content_clean, flags=re.DOTALL)
                json_str = brace_match.group(1) if brace_match else content_clean
            
            parsed = json.loads(json_str)
            route = parsed.get("route", "both").lower()
            if route not in ["vector", "graph", "both"]:
                route = "both"

            duration_ms = round((time.time() - start_time) * 1000, 2)
            log_entry = {
                "question": clean_q,
                "route": route,
                "reasoning": parsed.get("reasoning", ""),
                "entities": parsed.get("entities", []),
                "duration_ms": duration_ms,
            }
            self.routing_logs.append(log_entry)
            logger.info(f"[QueryRouter] Routed to '{route}' in {duration_ms}ms: {clean_q}")
            return log_entry
        except Exception as e:
            logger.warning(f"[QueryRouter] LLM classification error: {e}. Using heuristic fallback.")

        # Heuristic fallback on exception
        route = self._heuristic_classification(clean_q)
        duration_ms = round((time.time() - start_time) * 1000, 2)
        log_entry = {
            "question": clean_q,
            "route": route["route"],
            "reasoning": route["reasoning"],
            "entities": route["entities"],
            "duration_ms": duration_ms,
        }
        self.routing_logs.append(log_entry)
        return log_entry

    def _heuristic_classification(self, question: str) -> Dict[str, Any]:
        """Simple rule-based classifier when LLM is unavailable."""
        q_lower = question.lower()

        graph_signals = [
            "subsidiary", "subsidiaries", "ceo", "board", "executive", "officer",
            "supplies to", "supplier", "parent company", "affiliate", "connected to",
            "relationship between", "path between", "who reports to", "competitor",
            "competes with", "acqui", "partner", "ownership", "hierarchy", "shareholder"
        ]
        vector_signals = [
            "what was the revenue", "net income", "operating margin", "ebitda",
            "how much", "percentage", "rate", "according to item", "note 1", "note 2",
            "risk factor", "accounting policy", "amortization", "depreciation"
        ]

        has_graph = any(sig in q_lower for sig in graph_signals)
        has_vec = any(sig in q_lower for sig in vector_signals)

        if has_graph and has_vec:
            return {
                "route": "both",
                "reasoning": "Detected both structural relationship signals and metric lookups.",
                "entities": [],
            }
        elif has_graph:
            return {
                "route": "graph",
                "reasoning": "Detected entity-relationship query keywords.",
                "entities": [],
            }
        else:
            return {
                "route": "vector",
                "reasoning": "Defaulted to semantic vector search for text/metric retrieval.",
                "entities": [],
            }


# Singleton instance
query_router = QueryRouter()
