import sys
from pathlib import Path

rag_root = Path(r"C:\Android Projects\Enterprise-RAG-Financial-Assistant\rag-api")
sys.path.insert(0, str(rag_root))

from app.core.graph_store import graph_store

entities = [
    {"name": "Tesla", "type": "Company", "properties": {"ticker": "TSLA", "sector": "Automotive & Energy"}},
    {"name": "Elon Musk", "type": "Executive", "properties": {"title": "CEO"}},
    {"name": "Austin, Texas", "type": "Location", "properties": {"role": "Headquarters"}},
    {"name": "Cybercab", "type": "Product", "properties": {"category": "Autonomous Robotaxi"}},
    {"name": "Optimus", "type": "Product", "properties": {"category": "Humanoid Robot"}},
    {"name": "Model Y", "type": "Product", "properties": {"category": "Electric Vehicle"}},
    {"name": "Model 3", "type": "Product", "properties": {"category": "Electric Vehicle"}},
    {"name": "Megapack", "type": "Product", "properties": {"category": "Energy Storage"}},
    {"name": "Full Self-Driving (FSD)", "type": "Technology", "properties": {"category": "Autonomous Driving Software"}},
    {"name": "Gigafactory Texas", "type": "Facility", "properties": {"location": "Austin, Texas"}},
    {"name": "Microsoft", "type": "Company", "properties": {"ticker": "MSFT", "sector": "Technology"}},
    {"name": "Satya Nadella", "type": "Executive", "properties": {"title": "CEO"}},
    {"name": "Redmond, Washington", "type": "Location", "properties": {"role": "Headquarters"}},
    {"name": "Securities and Exchange Commission (SEC)", "type": "RegulatoryBody", "properties": {"jurisdiction": "United States"}},
    {"name": "Apple", "type": "Company", "properties": {"ticker": "AAPL", "sector": "Consumer Electronics"}},
    {"name": "Tim Cook", "type": "Executive", "properties": {"title": "CEO"}},
    {"name": "Cupertino, California", "type": "Location", "properties": {"role": "Headquarters"}},
    {"name": "Beats Electronics", "type": "Subsidiary", "properties": {"category": "Audio Hardware"}},
    {"name": "Foxconn", "type": "Supplier", "properties": {"role": "Contract Manufacturer"}},
    {"name": "TSMC", "type": "Supplier", "properties": {"role": "Semiconductor Foundry"}},
    {"name": "Samsung Electronics", "type": "Supplier", "properties": {"role": "Display & Memory Supplier"}}
]

relationships = [
    # Tesla Corporate & Products
    {"source": "Tesla", "type": "LED_BY", "target": "Elon Musk"},
    {"source": "Tesla", "type": "HEADQUARTERED_IN", "target": "Austin, Texas"},
    {"source": "Tesla", "type": "MANUFACTURES", "target": "Cybercab"},
    {"source": "Tesla", "type": "MANUFACTURES", "target": "Optimus"},
    {"source": "Tesla", "type": "MANUFACTURES", "target": "Model Y"},
    {"source": "Tesla", "type": "MANUFACTURES", "target": "Model 3"},
    {"source": "Tesla", "type": "MANUFACTURES", "target": "Megapack"},
    {"source": "Tesla", "type": "DEVELOPS", "target": "Full Self-Driving (FSD)"},
    {"source": "Tesla", "type": "OPERATES", "target": "Gigafactory Texas"},
    {"source": "Gigafactory Texas", "type": "LOCATED_IN", "target": "Austin, Texas"},
    {"source": "Tesla", "type": "REGULATED_BY", "target": "Securities and Exchange Commission (SEC)"},

    # Microsoft Corporate
    {"source": "Microsoft", "type": "LED_BY", "target": "Satya Nadella"},
    {"source": "Microsoft", "type": "HEADQUARTERED_IN", "target": "Redmond, Washington"},
    {"source": "Microsoft", "type": "REGULATED_BY", "target": "Securities and Exchange Commission (SEC)"},

    # Apple Corporate, Subsidiaries & Supply Chain
    {"source": "Apple", "type": "LED_BY", "target": "Tim Cook"},
    {"source": "Apple", "type": "HEADQUARTERED_IN", "target": "Cupertino, California"},
    {"source": "Beats Electronics", "type": "SUBSIDIARY_OF", "target": "Apple"},
    {"source": "Foxconn", "type": "SUPPLIES_TO", "target": "Apple"},
    {"source": "TSMC", "type": "SUPPLIES_TO", "target": "Apple"},
    {"source": "Samsung Electronics", "type": "SUPPLIES_TO", "target": "Apple"},
    {"source": "Apple", "type": "REGULATED_BY", "target": "Securities and Exchange Commission (SEC)"}
]

print(f"Seeding {len(entities)} entities and {len(relationships)} relationships into Neo4j...")
graph_store.add_entities_and_relations(entities, relationships, doc_id="sec_filings_seed", chunk_id="seed_0")
counts = graph_store.count_nodes_and_edges()
print(f"Updated Neo4j counts: {counts}")
