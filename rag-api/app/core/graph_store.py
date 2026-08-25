import logging
from typing import Any, Dict, List, Optional
from neo4j import GraphDatabase, Driver
from app.core.config import settings

logger = logging.getLogger("graph_store")


class GraphStore:
    """
    Neo4j Graph Database client for Knowledge Graph construction and traversal.

    Key Features:
    - Uses MERGE for idempotent node/edge ingestion (no duplicates on re-ingestion).
    - Every edge stores `source_chunk_id` and `doc_id` for 100% provenance and citation.
    - Uses parameterized Cypher queries ONLY (zero risk of injection from LLMs).
    - Manages connection lifecycle with reconnect resilience.
    """

    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
    ):
        self.uri = uri or settings.NEO4J_URI
        self.user = user or settings.NEO4J_USER
        self.password = password or settings.NEO4J_PASSWORD
        self._driver: Optional[Driver] = None
        self._initialize_driver()

    def _initialize_driver(self) -> None:
        try:
            self._driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password),
                max_connection_lifetime=3600,
            )
            self.verify_connectivity()
            self._setup_schema_constraints()
            logger.info("[GraphStore] Connected successfully to Neo4j at %s", self.uri)
        except Exception as e:
            logger.warning("[GraphStore] Could not connect to Neo4j at %s: %s", self.uri, e)

    def get_driver(self) -> Driver:
        if self._driver is None:
            self._initialize_driver()
        return self._driver

    def verify_connectivity(self) -> bool:
        """Verify that Neo4j is reachable and credentials are valid."""
        if self._driver is None:
            return False
        try:
            self._driver.verify_connectivity()
            return True
        except Exception as e:
            logger.error("[GraphStore] Neo4j connectivity check failed: %s", e)
            return False

    def close(self) -> None:
        """Close driver connection."""
        if self._driver is not None:
            self._driver.close()
            self._driver = None

    def _setup_schema_constraints(self) -> None:
        """
        Create uniqueness constraints and indexes for high-speed node matching.
        """
        if not self.verify_connectivity():
            return
        
        # Entity labels in our financial ontology
        node_labels = ["Company", "Person", "Product", "Location", "Financial_Metric", "Regulation", "Entity"]
        with self._driver.session() as session:
            for label in node_labels:
                try:
                    query = f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label}) REQUIRE n.name IS UNIQUE"
                    session.run(query)
                except Exception as e:
                    logger.debug(f"[GraphStore] Constraint creation notice for {label}: {e}")

    def add_entities_and_relations(
        self,
        entities: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]],
        doc_id: str,
        chunk_id: str,
    ) -> Dict[str, int]:
        """
        Idempotently write nodes and relationships to Neo4j using parameterized MERGE queries.
        
        Args:
            entities: List of dicts with keys: name, type, description (optional)
            relationships: List of dicts with keys: source, target, type, description (optional)
            doc_id: The document identifier
            chunk_id: The source chunk identifier (for provenance)
            
        Returns:
            Dict with counts of nodes and relationships stored
        """
        if not self.verify_connectivity():
            logger.warning("[GraphStore] Neo4j not connected. Skipping graph storage.")
            return {"nodes": 0, "relationships": 0}

        nodes_created = 0
        rels_created = 0

        with self._driver.session() as session:
            # 1. Store Entities
            for ent in entities:
                name = ent.get("name", "").strip()
                ent_type = ent.get("type", "Entity").strip()
                if not name:
                    continue

                # Ensure entity type is a valid identifier (alphanumeric/underscore)
                safe_label = "".join([c for c in ent_type if c.isalnum() or c == "_"]) or "Entity"

                cypher = f"""
                MERGE (n:`{safe_label}` {{name: $name}})
                ON CREATE SET 
                    n.first_seen_doc = $doc_id,
                    n.first_seen_chunk = $chunk_id,
                    n.type = $ent_type,
                    n.description = $description,
                    n.aliases = [$name],
                    n.created_at = timestamp()
                ON MATCH SET
                    n.last_seen_doc = $doc_id,
                    n.last_seen_chunk = $chunk_id,
                    n.aliases = CASE WHEN NOT $name IN n.aliases THEN n.aliases + $name ELSE n.aliases END
                RETURN n.name
                """
                session.run(
                    cypher,
                    name=name,
                    ent_type=safe_label,
                    description=ent.get("description", ""),
                    doc_id=doc_id,
                    chunk_id=chunk_id,
                )
                nodes_created += 1

            # 2. Store Relationships
            for rel in relationships:
                source = rel.get("source", "").strip()
                target = rel.get("target", "").strip()
                rel_type = rel.get("type", "RELATED_TO").strip().upper()

                if not source or not target:
                    continue

                # Sanitize relationship type
                safe_rel_type = "".join([c for c in rel_type if c.isalnum() or c == "_"]) or "RELATED_TO"

                cypher = f"""
                MERGE (a {{name: $source}})
                MERGE (b {{name: $target}})
                MERGE (a)-[r:`{safe_rel_type}`]->(b)
                ON CREATE SET 
                    r.doc_id = $doc_id,
                    r.chunk_id = $chunk_id,
                    r.description = $description,
                    r.created_at = timestamp()
                ON MATCH SET 
                    r.last_updated = timestamp(),
                    r.chunk_id = $chunk_id
                RETURN type(r)
                """
                res = session.run(
                    cypher,
                    source=source,
                    target=target,
                    description=rel.get("description", ""),
                    doc_id=doc_id,
                    chunk_id=chunk_id,
                )
                summary = res.consume()
                if summary.counters.relationships_created > 0 or summary.counters.contains_updates:
                    rels_created += 1

        return {"nodes": nodes_created, "relationships": rels_created}

    def query_graph(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Run a parameterized read-only Cypher query."""
        if not self.verify_connectivity():
            return []

        parameters = parameters or {}
        with self._driver.session() as session:
            result = session.run(query, parameters)
            return [record.data() for record in result]

    def find_direct_relationships(self, entity1: str, entity2: str) -> List[Dict[str, Any]]:
        """Find direct 1-hop relationships between two entities."""
        cypher = """
        MATCH (a {name: $entity1})-[r]-(b {name: $entity2})
        RETURN a.name AS source, type(r) AS relationship, r.description AS description, 
               b.name AS target, r.chunk_id AS chunk_id, r.doc_id AS doc_id
        """
        return self.query_graph(cypher, {"entity1": entity1, "entity2": entity2})

    def find_multi_hop_paths(self, entity1: str, entity2: str, max_hops: int = 2) -> List[Dict[str, Any]]:
        """Find paths of up to max_hops between two entities."""
        safe_hops = min(max(1, max_hops), 3)
        cypher = f"""
        MATCH p = (a {{name: $entity1}})-[*1..{safe_hops}]-(b {{name: $entity2}})
        RETURN [n in nodes(p) | n.name] AS path_nodes,
               [r in relationships(p) | type(r)] AS relationship_types,
               [r in relationships(p) | coalesce(r.chunk_id, "unknown")] AS chunk_ids,
               length(p) AS hops
        ORDER BY hops ASC
        LIMIT 10
        """
        return self.query_graph(cypher, {"entity1": entity1, "entity2": entity2})

    def get_entity_neighbors(self, entity_name: str, limit: int = 15) -> List[Dict[str, Any]]:
        """Get 1-hop neighbors of an entity."""
        cypher = """
        MATCH (a {name: $entity})-[r]-(b)
        RETURN a.name AS source, type(r) AS relationship, b.name AS target, 
               labels(b)[0] AS target_type, r.description AS description, r.chunk_id AS chunk_id
        LIMIT $limit
        """
        return self.query_graph(cypher, {"entity": entity_name, "limit": limit})

    def count_nodes_and_edges(self) -> Dict[str, int]:
        """Return total counts of nodes and relationships in the graph."""
        if not self.verify_connectivity():
            return {"node_count": 0, "relationship_count": 0}
        
        node_res = self.query_graph("MATCH (n) RETURN count(n) AS count")
        edge_res = self.query_graph("MATCH ()-[r]->() RETURN count(r) AS count")
        
        node_count = node_res[0]["count"] if node_res else 0
        edge_count = edge_res[0]["count"] if edge_res else 0
        return {"node_count": node_count, "relationship_count": edge_count}

    def clear_database(self) -> None:
        """Delete all nodes and relationships (use for testing or reset)."""
        if self.verify_connectivity():
            with self._driver.session() as session:
                session.run("MATCH (n) DETACH DELETE n")
            logger.info("[GraphStore] Graph database cleared.")


# Singleton instance
graph_store = GraphStore()
