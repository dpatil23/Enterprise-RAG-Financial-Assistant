import logging
from typing import Any, Dict, List, Optional
from app.core.graph_store import graph_store
from app.services.entity_resolution import entity_resolver

logger = logging.getLogger("graph_query_service")


class GraphQueryService:
    """
    Translates user questions and entities into safe parameterized Cypher queries,
    executes graph traversals, and formats structured knowledge facts for LLM synthesis.
    """

    def __init__(self):
        self.store = graph_store

    def query_graph_for_context(
        self,
        question: str,
        entities: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Main entrypoint for querying the graph given a question and extracted entity candidates.
        
        Returns:
            Dict containing formatted context strings, structured path objects, and source chunk IDs.
        """
        entities = entities or []
        # Resolve entities to canonical names in graph
        resolved_entities = [
            entity_resolver.resolve_entity(ent) for ent in entities if ent.strip()
        ]
        # Also discover any explicit graph node names in question text
        matched_nodes = self._search_nodes_by_keywords(question)
        for mn in matched_nodes:
            resolved_entities.append(mn["name"])

        # Remove duplicates preserving order
        unique_entities = list(dict.fromkeys(resolved_entities))

        graph_facts: List[str] = []
        structured_sources: List[Dict[str, Any]] = []
        cited_chunks: set[str] = set()

        if len(unique_entities) >= 2:
            # Multi-entity query: find paths and direct relations between pairs
            e1 = unique_entities[0]
            e2 = unique_entities[1]

            # 1. Check direct relations
            direct_rels = self.store.find_direct_relationships(e1, e2)
            for r in direct_rels:
                fact_str = f"{r['source']} is {r['relationship']} -> {r['target']}"
                if r.get("description"):
                    fact_str += f" ({r['description']})"
                graph_facts.append(fact_str)
                chunk_id = r.get("chunk_id", "unknown_chunk")
                cited_chunks.add(chunk_id)
                structured_sources.append({
                    "type": "graph",
                    "path": fact_str,
                    "chunk_id": chunk_id,
                    "doc_id": r.get("doc_id", "unknown_doc"),
                })

            # 2. Check multi-hop paths (up to 2 or 3 hops)
            paths = self.store.find_multi_hop_paths(e1, e2, max_hops=3)
            for p in paths:
                nodes = p.get("path_nodes", [])
                rels = p.get("relationship_types", [])
                chunk_ids = p.get("chunk_ids", [])
                
                # Format: Node1 -[REL]-> Node2 -[REL]-> Node3
                path_repr_parts = []
                for i in range(len(nodes)):
                    path_repr_parts.append(nodes[i])
                    if i < len(rels):
                        path_repr_parts.append(f"-[{rels[i]}]->")
                path_repr = " ".join(path_repr_parts)

                if path_repr not in graph_facts:
                    graph_facts.append(path_repr)
                    for cid in chunk_ids:
                        if cid and cid != "unknown":
                            cited_chunks.add(cid)
                    structured_sources.append({
                        "type": "graph",
                        "path": path_repr,
                        "chunk_id": chunk_ids[0] if chunk_ids else "unknown",
                        "hops": p.get("hops", 1),
                    })

        # Also get 1-hop neighborhoods for each mentioned entity
        for ent in unique_entities:
            neighbors = self.store.get_entity_neighbors(ent, limit=10)
            for n in neighbors:
                fact_str = f"{n['source']} -[{n['relationship']}]-> {n['target']} ({n.get('target_type', 'Entity')})"
                if n.get("description"):
                    fact_str += f": {n['description']}"
                if fact_str not in graph_facts:
                    graph_facts.append(fact_str)
                    chunk_id = n.get("chunk_id", "unknown_chunk")
                    if chunk_id and chunk_id != "unknown_chunk":
                        cited_chunks.add(chunk_id)
                    structured_sources.append({
                        "type": "graph",
                        "path": fact_str,
                        "chunk_id": chunk_id,
                    })

        # Fallback: if no entities extracted, search for keyword matches in node names
        if not graph_facts and not unique_entities:
            general_nodes = self._search_nodes_by_keywords(question)
            for n in general_nodes:
                neighbors = self.store.get_entity_neighbors(n["name"], limit=5)
                for item in neighbors:
                    fact_str = f"{item['source']} -[{item['relationship']}]-> {item['target']}"
                    if fact_str not in graph_facts:
                        graph_facts.append(fact_str)
                        structured_sources.append({
                            "type": "graph",
                            "path": fact_str,
                            "chunk_id": item.get("chunk_id", "unknown_chunk"),
                        })

        return {
            "facts": graph_facts,
            "sources": structured_sources,
            "cited_chunks": list(cited_chunks),
            "entities_queried": unique_entities,
        }

    def _search_nodes_by_keywords(self, question: str) -> List[Dict[str, Any]]:
        """Find node names that appear as substrings in the question (case-insensitive)."""
        cypher = """
        MATCH (n)
        WHERE toLower($q) CONTAINS toLower(n.name)
        RETURN n.name AS name, labels(n)[0] AS type
        LIMIT 10
        """
        return self.store.query_graph(cypher, {"q": question.lower()})


# Singleton instance
graph_query_service = GraphQueryService()
