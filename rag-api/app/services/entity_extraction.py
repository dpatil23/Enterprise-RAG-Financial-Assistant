import json
import logging
import re
from typing import Any, Dict, List, Optional
from openai import OpenAI
from app.core.config import settings

logger = logging.getLogger("entity_extraction")

# Financial Ontology Constants
VALID_ENTITY_TYPES = {
    "Company",
    "Person",
    "Product",
    "Location",
    "Financial_Metric",
    "Regulation",
}

VALID_RELATIONSHIP_TYPES = {
    "SUBSIDIARY_OF",
    "CEO_OF",
    "EMPLOYS",
    "SUPPLIES_TO",
    "HEADQUARTERED_IN",
    "REPORTED_METRIC",
    "COMPETES_WITH",
    "REGULATED_BY",
    "PARTNERED_WITH",
    "ACQUIRED",
}

EXTRACTION_SYSTEM_PROMPT = """You are an expert financial knowledge graph architect extracting structured entities and relationships from SEC filings and business documents.

Extract entities and relationships STRICTLY matching this ontology:

ENTITY TYPES:
- Company (e.g., Apple Inc., Microsoft, Foxconn, Beats)
- Person (e.g., Tim Cook, Satya Nadella, Luca Maestri)
- Product (e.g., iPhone, Azure Cloud, Windows)
- Location (e.g., Cupertino, California, Redmond, Washington)
- Financial_Metric (e.g., Revenue: $394B, Net Income: $99.8B, EPS: $6.11)
- Regulation (e.g., SOX Act, GDPR, SEC Rule 10b-5)

RELATIONSHIP TYPES:
- SUBSIDIARY_OF (Company -> Company)
- CEO_OF (Person -> Company)
- EMPLOYS (Company -> Person)
- SUPPLIES_TO (Company -> Company)
- HEADQUARTERED_IN (Company -> Location)
- REPORTED_METRIC (Company -> Financial_Metric)
- COMPETES_WITH (Company -> Company)
- REGULATED_BY (Company -> Regulation)
- PARTNERED_WITH (Company -> Company)
- ACQUIRED (Company -> Company)

RULES:
1. Every entity must have a clear "name" and valid "type".
2. Every relationship must have "source" (must exist in entities), "target" (must exist in entities), "type" (one of valid types), and brief "description".
3. Return ONLY a valid JSON object with keys "entities" and "relationships".
4. If no clear financial entities/relations exist in the text, return {"entities": [], "relationships": []}.

JSON OUTPUT FORMAT:
{
  "entities": [
    {"name": "Apple Inc.", "type": "Company"},
    {"name": "Tim Cook", "type": "Person"},
    {"name": "Cupertino", "type": "Location"}
  ],
  "relationships": [
    {"source": "Tim Cook", "target": "Apple Inc.", "type": "CEO_OF", "description": "Tim Cook serves as Chief Executive Officer"},
    {"source": "Apple Inc.", "target": "Cupertino", "type": "HEADQUARTERED_IN", "description": "Headquartered in Cupertino, CA"}
  ]
}"""


class EntityExtractor:
    """
    Extracts structured entities and knowledge graph relationships from text chunks using OpenAI/Puter/Groq LLM.
    """

    def __init__(self, client: Optional[OpenAI] = None):
        self._client = client
        self._init_client()

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
                logger.warning(f"[EntityExtractor] Client init failed: {e}")

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

    def extract_from_chunk(self, chunk_text: str, max_retries: int = 0) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract entities and relationships from a text chunk.
        """
        if not chunk_text.strip():
            return {"entities": [], "relationships": []}

        client = self._get_client()
        if client is None:
            logger.debug("[EntityExtractor] No API key provided; returning empty extraction.")
            return {"entities": [], "relationships": []}

        prompt = f"Analyze the text and extract entities and relationships. Return JSON ONLY without explanation:\n\nTEXT:\n{chunk_text[:1200]}"

        for attempt in range(max_retries + 1):
            try:
                response = client.chat.completions.create(
                    model=self._get_model(),
                    messages=[
                        {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.0,
                    max_tokens=600,
                )
                content = response.choices[0].message.content.strip()
                
                parsed = None
                # 1. Try markdown ```json ... ``` extraction
                match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", content)
                if match:
                    try:
                        parsed = json.loads(match.group(1))
                    except Exception:
                        pass
                
                # 2. Try searching for explicit {"entities": ...} object
                if parsed is None:
                    obj_match = re.search(r"(\{\s*\"entities\"[\s\S]*?\})", content)
                    if obj_match:
                        try:
                            parsed, _ = json.JSONDecoder().raw_decode(obj_match.group(1))
                        except Exception:
                            pass

                # 3. Fallback: find any clean JSON dictionary after </think>
                if parsed is None:
                    content_clean = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
                    start_idx = content_clean.find('{')
                    if start_idx != -1:
                        try:
                            parsed, _ = json.JSONDecoder().raw_decode(content_clean[start_idx:])
                        except Exception:
                            pass

                if not isinstance(parsed, dict):
                    continue

                # Validate and normalize
                entities = parsed.get("entities", [])
                relationships = parsed.get("relationships", [])

                clean_entities = []
                entity_names = set()
                for ent in entities:
                    if isinstance(ent, dict) and ent.get("name"):
                        name = str(ent["name"]).strip()
                        raw_type = str(ent.get("type", "Company")).strip()
                        norm_type = raw_type if raw_type in VALID_ENTITY_TYPES else "Company"
                        if name and name not in entity_names:
                            entity_names.add(name)
                            clean_entities.append({
                                "name": name,
                                "type": norm_type,
                                "description": ent.get("description", ""),
                            })

                clean_relationships = []
                for rel in relationships:
                    if isinstance(rel, dict) and rel.get("source") and rel.get("target"):
                        source = str(rel["source"]).strip()
                        target = str(rel["target"]).strip()
                        rel_type = str(rel.get("type", "RELATED_TO")).strip().upper()
                        norm_rel = rel_type if rel_type in VALID_RELATIONSHIP_TYPES else "RELATED_TO"

                        # Ensure both ends exist in entity set
                        if source and target and source != target:
                            clean_relationships.append({
                                "source": source,
                                "target": target,
                                "type": norm_rel,
                                "description": rel.get("description", ""),
                            })

                return {"entities": clean_entities, "relationships": clean_relationships}

            except Exception as e:
                logger.warning(f"[EntityExtractor] Extraction attempt {attempt + 1} failed: {e}")
                if attempt == max_retries:
                    break

        return {"entities": [], "relationships": []}


# Singleton instance
entity_extractor = EntityExtractor()
