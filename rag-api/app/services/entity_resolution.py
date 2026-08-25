import logging
import numpy as np
from typing import Any, Dict, List, Tuple
from app.core.embedder import embedder

logger = logging.getLogger("entity_resolution")


class EntityResolver:
    """
    Entity Resolution and Canonicalization engine.
    
    Prevents graph fragmentation by mapping surface-form variants (e.g. 'Apple Inc.', 'Apple', 'AAPL')
    to a single canonical entity node.
    
    Approach:
    1. Exact match / lower-case normalization.
    2. Substring & corporate suffix rules (e.g., 'Apple Inc.' -> 'Apple').
    3. Dense semantic embedding similarity (> 0.85 cosine similarity) for fuzzy matching.
    4. Maintains an alias-to-canonical mapping.
    """

    def __init__(self, similarity_threshold: float = 0.85):
        self.similarity_threshold = similarity_threshold
        # canonical_name -> list of aliases
        self.canonical_entities: Dict[str, List[str]] = {}
        # alias -> canonical_name
        self.alias_map: Dict[str, str] = {}
        # canonical_name -> embedding vector (np.ndarray)
        self.canonical_embeddings: Dict[str, np.ndarray] = {}

    def _clean_suffix(self, name: str) -> str:
        """Strip common corporate suffixes to extract the core brand name."""
        cleaned = name.strip()
        suffixes = [
            " Corporation", " Corp.", " Corp",
            " Incorporated", " Inc.", " Inc",
            " Limited", " Ltd.", " Ltd",
            " LLC", " L.L.C.", " Co.", " Co", " S.A.", " AG", " NV"
        ]
        for suffix in suffixes:
            if cleaned.lower().endswith(suffix.lower()):
                cleaned = cleaned[: -len(suffix)].strip()
                break
        return cleaned if cleaned else name.strip()

    def resolve_entity(self, entity_name: str, entity_type: str = "Company") -> str:
        """
        Resolve a single entity name to its canonical form.
        If it matches an existing canonical entity above the threshold, return that.
        Otherwise register it as a new canonical entity.
        """
        raw_name = entity_name.strip()
        if not raw_name:
            return raw_name

        lower_name = raw_name.lower()
        base_name = self._clean_suffix(raw_name) if entity_type == "Company" else raw_name
        lower_base = base_name.lower()

        # 1. Check exact or alias match on raw or base name
        if raw_name in self.alias_map:
            return self.alias_map[raw_name]
        if lower_name in self.alias_map:
            return self.alias_map[lower_name]
        if base_name in self.alias_map:
            self._add_alias(self.alias_map[base_name], raw_name)
            return self.alias_map[base_name]
        if lower_base in self.alias_map:
            self._add_alias(self.alias_map[lower_base], raw_name)
            return self.alias_map[lower_base]

        # 2. Check if base_name is already a known canonical entity
        if base_name in self.canonical_entities:
            self._add_alias(base_name, raw_name)
            return base_name

        # 3. Vector similarity matching against existing canonical entities
        if self.canonical_embeddings:
            try:
                candidate_embedding = np.array(embedder.embed([base_name])[0], dtype=np.float32)
                norm_c = np.linalg.norm(candidate_embedding)
                if norm_c > 0:
                    candidate_embedding = candidate_embedding / norm_c

                best_match = None
                best_score = -1.0

                for canon_name, emb in self.canonical_embeddings.items():
                    score = float(np.dot(candidate_embedding, emb))
                    if score > best_score:
                        best_score = score
                        best_match = canon_name

                if best_match and best_score >= self.similarity_threshold:
                    logger.debug(
                        f"[EntityResolver] Resolved '{raw_name}' -> '{best_match}' (score: {best_score:.3f})"
                    )
                    self._add_alias(best_match, raw_name)
                    return best_match
            except Exception as e:
                logger.warning(f"[EntityResolver] Embedding comparison error: {e}")

        # 4. Register base_name as the new canonical entity
        self._register_new_canonical(base_name)
        if raw_name != base_name:
            self._add_alias(base_name, raw_name)
        return base_name

    def _register_new_canonical(self, canonical_name: str) -> None:
        self.canonical_entities[canonical_name] = [canonical_name]
        self.alias_map[canonical_name] = canonical_name
        self.alias_map[canonical_name.lower()] = canonical_name

        try:
            emb = np.array(embedder.embed([canonical_name])[0], dtype=np.float32)
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
            self.canonical_embeddings[canonical_name] = emb
        except Exception as e:
            logger.debug(f"[EntityResolver] Could not compute embedding for canonical entity: {e}")

    def _add_alias(self, canonical_name: str, alias: str) -> None:
        if canonical_name not in self.canonical_entities:
            self.canonical_entities[canonical_name] = [canonical_name]
        if alias not in self.canonical_entities[canonical_name]:
            self.canonical_entities[canonical_name].append(alias)
        self.alias_map[alias] = canonical_name
        self.alias_map[alias.lower()] = canonical_name

    def resolve_batch(
        self,
        entities: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Resolve all entities in batch, updating relationships to point to canonical names.
        """
        local_map: Dict[str, str] = {}
        canonical_entities_map: Dict[str, Dict[str, Any]] = {}

        for ent in entities:
            orig_name = ent.get("name", "").strip()
            ent_type = ent.get("type", "Entity")
            if not orig_name:
                continue
            canonical = self.resolve_entity(orig_name, ent_type)
            local_map[orig_name] = canonical

            if canonical not in canonical_entities_map:
                canonical_entities_map[canonical] = {
                    "name": canonical,
                    "type": ent_type,
                    "description": ent.get("description", ""),
                }

        resolved_relationships = []
        for rel in relationships:
            source = rel.get("source", "").strip()
            target = rel.get("target", "").strip()
            canon_source = local_map.get(source, self.resolve_entity(source))
            canon_target = local_map.get(target, self.resolve_entity(target))

            if canon_source and canon_target and canon_source != canon_target:
                resolved_relationships.append({
                    "source": canon_source,
                    "target": canon_target,
                    "type": rel.get("type", "RELATED_TO"),
                    "description": rel.get("description", ""),
                })

        return list(canonical_entities_map.values()), resolved_relationships


# Singleton instance
entity_resolver = EntityResolver()
