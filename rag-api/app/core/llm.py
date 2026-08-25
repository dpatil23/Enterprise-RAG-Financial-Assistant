import re
import logging
from typing import Any, Dict, List, Optional
from openai import OpenAI
from app.core.config import settings

logger = logging.getLogger("llm_client")


class LLMClient:
    """
    Universal OpenAI-compatible LLM client supporting Puter.js (Claude 3.5 Sonnet, GPT-4o, Grok-2),
    Groq (LLaMA 3.3), native OpenAI, and local Ollama.
    """

    def __init__(self):
        self._client: Optional[OpenAI] = None
        self._init_client()

    def _init_client(self) -> None:
        api_key = settings.LLM_API_KEY or settings.GROQ_API_KEY
        base_url = settings.LLM_BASE_URL

        # If user configured Groq specifically but left LLM_BASE_URL default
        if settings.GROQ_API_KEY and not settings.LLM_API_KEY:
            base_url = "https://api.groq.com/openai/v1"
            api_key = settings.GROQ_API_KEY

        if api_key:
            try:
                self._client = OpenAI(
                    base_url=base_url,
                    api_key=api_key,
                )
                logger.info(
                    f"[LLMClient] Initialized with provider '{settings.LLM_PROVIDER}', model: {self.get_model_name()} at {base_url}"
                )
            except Exception as e:
                logger.warning(f"[LLMClient] LLM client initialization failed: {e}")

    def get_client(self) -> Optional[OpenAI]:
        if self._client is None and (settings.LLM_API_KEY or settings.GROQ_API_KEY):
            self._init_client()
        return self._client

    def get_model_name(self) -> str:
        if settings.LLM_API_KEY:
            return settings.LLM_MODEL
        if settings.GROQ_API_KEY:
            return settings.GROQ_MODEL
        return settings.LLM_MODEL

    def generate_answer(self, question: str, context_chunks: list[dict]) -> str:
        """Vector-only answer generation."""
        return self.generate_hybrid_answer(
            question=question,
            context_chunks=context_chunks,
            graph_facts=[],
        )

    def generate_hybrid_answer(
        self,
        question: str,
        context_chunks: List[Dict[str, Any]],
        graph_facts: List[str],
    ) -> str:
        """
        Build a multi-modal context block containing both structured Knowledge Graph facts
        and unstructured text chunks, generating a fully grounded answer with citations.
        """
        client = self.get_client()
        if client is None:
            # Fallback if no API key is provided
            if graph_facts and context_chunks:
                return (
                    f"[Hybrid RAG (Local Mode)]: Found {len(graph_facts)} graph facts and {len(context_chunks)} "
                    f"vector passages relevant to '{question}'. (Set LLM_API_KEY to enable full Claude/GPT-4o synthesis)."
                )
            elif graph_facts:
                facts_summary = "; ".join(graph_facts[:3])
                return f"[Graph RAG (Local Mode)]: Retrieved relationships: {facts_summary}."
            elif context_chunks:
                snippet = context_chunks[0]['text'][:200]
                return f"[Vector RAG (Local Mode)]: Most relevant excerpt: \"{snippet}...\". (Set LLM_API_KEY for synthesis)."
            else:
                return "I could not find enough relevant context in the uploaded documents to answer this question."

        # Build Knowledge Graph section
        graph_section = ""
        if graph_facts:
            graph_lines = "\n".join([f"- {fact}" for fact in graph_facts])
            graph_section = f"### STRUCTURED KNOWLEDGE GRAPH FACTS:\n{graph_lines}\n\n"

        # Build Vector Text Chunks section
        vector_section = ""
        if context_chunks:
            chunk_blocks = [
                f"[Source Chunk ID: {c.get('source', 'doc')}_chunk_{c.get('chunk_index', 0)} | Doc: {c.get('source', '')}]\n{c.get('text', '')}"
                for c in context_chunks
            ]
            vector_section = "### DOCUMENT EXCERPTS / PASSAGES:\n" + "\n\n---\n\n".join(chunk_blocks)

        prompt = f"""You are a senior financial analyst and Enterprise Knowledge Graph expert.
Your job is to answer user questions with absolute factual accuracy based ONLY on the provided Knowledge Graph facts and Document excerpts.

{graph_section}{vector_section}

USER QUESTION:
{question}

INSTRUCTIONS:
1. Provide the direct, structured financial answer immediately. Do NOT output internal monologue or <think> tags.
2. Answer strictly using information in the CONTEXT above. Do not assume or extrapolate external financial facts.
3. If the context does not contain enough information to answer, state:
   "I could not find a direct answer in the provided document."
4. Cite your sources for key statements:
   - For facts from document excerpts, cite the chunk ID, e.g., `[Source: document_chunk_0]`.
   - For relationship paths from the knowledge graph, mention the verified relationship path, e.g., `(Verified via Knowledge Graph: Entity A -> Entity B)`.
5. Provide a well-structured, clear, and professional response.

ANSWER:"""

        try:
            response = client.chat.completions.create(
                model=self.get_model_name(),
                messages=[
                    {"role": "system", "content": "You are a concise financial analyst. Return direct structured markdown answers without <think> tags."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=2500,
            )
            content = response.choices[0].message.content.strip()
            
            # Robustly strip thinking tags (both closed and unclosed)
            if "</think>" in content:
                clean_content = content.split("</think>")[-1].strip()
            elif "<think>" in content:
                # If cut off inside think, extract the drafted answer from inside
                draft_match = re.search(r"(?:Based on|Refined:|Draft:|\n\n\*\*|Total revenues)(.*)", content, re.DOTALL | re.IGNORECASE)
                if draft_match:
                    clean_content = draft_match.group(0).strip()
                else:
                    clean_content = re.sub(r"<think>", "", content).strip()
            else:
                clean_content = content
                
            return clean_content if clean_content else content
        except Exception as e:
            if "429" in str(e) or "rate_limit" in str(e).lower():
                logger.warning(f"[LLMClient] Rate limit hit. Falling back to Gemini. Original error: {e}")
                try:
                    gemini_api_key = os.environ.get("GEMINI_API_KEY", "")
                    if not gemini_api_key:
                        raise ValueError("GEMINI_API_KEY environment variable is not set for fallback.")

                    gemini_client = OpenAI(
                        api_key=gemini_api_key,
                        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
                    )
                    response = gemini_client.chat.completions.create(
                        model="gemini-3.6-flash",
                        messages=[
                            {"role": "system", "content": "You are a concise financial analyst. Return direct structured markdown answers without <think> tags."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.1,
                        max_tokens=2500,
                    )
                    return response.choices[0].message.content.strip()
                except Exception as gemini_e:
                    logger.error(f"[LLMClient] Gemini fallback also failed: {gemini_e}")
                    return f"An error occurred while generating the answer via LLM: {str(e)} | Fallback error: {str(gemini_e)}"
            else:
                logger.error(f"[LLMClient] API completion error: {e}")
                return f"An error occurred while generating the answer via LLM: {str(e)}"


# Singleton instance
llm_client = LLMClient()
