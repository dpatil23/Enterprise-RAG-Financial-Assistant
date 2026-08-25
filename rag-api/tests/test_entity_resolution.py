from app.services.entity_resolution import EntityResolver


def test_entity_resolution_corporate_suffixes():
    resolver = EntityResolver(similarity_threshold=0.85)

    # First register Apple Inc
    canon1 = resolver.resolve_entity("Apple Inc.", "Company")
    assert canon1 in ["Apple", "Apple Inc."]

    # Apple should map to the same canonical entity
    canon2 = resolver.resolve_entity("Apple", "Company")
    assert canon2 == canon1


def test_entity_resolution_batch():
    resolver = EntityResolver(similarity_threshold=0.85)

    entities = [
        {"name": "Microsoft Corporation", "type": "Company"},
        {"name": "Microsoft", "type": "Company"},
        {"name": "Satya Nadella", "type": "Person"},
    ]
    relationships = [
        {"source": "Satya Nadella", "target": "Microsoft Corporation", "type": "CEO_OF"},
        {"source": "Microsoft", "target": "Satya Nadella", "type": "EMPLOYS"},
    ]

    resolved_entities, resolved_rels = resolver.resolve_batch(entities, relationships)

    # Should collapse Microsoft and Microsoft Corporation into 1 entity
    assert len(resolved_entities) == 2
    ent_names = [e["name"] for e in resolved_entities]
    assert "Satya Nadella" in ent_names

    # Relationship targets should point to the canonical entity
    for rel in resolved_rels:
        assert rel["source"] in ent_names
        assert rel["target"] in ent_names
