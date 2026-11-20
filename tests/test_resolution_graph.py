from __future__ import annotations

from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from graphrag_pipeline.graph import KnowledgeGraph, load_graphml, scan_mentions
from graphrag_pipeline.ingest import chunk_document, document_from_text
from graphrag_pipeline.resolution import EntityResolver, initials, match_key, normalise_name
from graphrag_pipeline.schemas import COOCCURRENCE, ChunkExtraction, Entity, EntityMention, Relation, RelationMention


def mention(name: str, etype: str = "ORGANIZATION", chunk: str = "d#0", description: str = "") -> EntityMention:
    return EntityMention(name=name, type=etype, chunk_id=chunk, description=description)  # type: ignore[arg-type]


def resolve(*names: tuple[str, str], relations: list[RelationMention] | None = None, text: str = "") -> dict[str, str]:
    extraction = ChunkExtraction(chunk_id="d#0", entities=[mention(n, t) for n, t in names],
                                 relations=relations or [])
    chunks = chunk_document(document_from_text("d", text)) if text else []
    resolution = EntityResolver().resolve([extraction], chunks)
    return {name: resolution.entities[resolution.key_to_id[match_key(name)]].name for name, _ in names}


def test_normalisation() -> None:
    assert normalise_name("The Velltrix Systems's") == "velltrix systems"
    assert match_key("Acme Robotics Inc.") == "acme robotics"
    assert match_key("Ember-9") == "ember 9"
    assert initials("Quorvale Institute of Technology") == "qit"


@given(st.text(max_size=60))
def test_normalise_is_idempotent(text: str) -> None:
    once = normalise_name(text)
    assert normalise_name(once) == once
    assert match_key(match_key(text)) == match_key(text)


def test_acronyms_short_forms_surnames_and_plurals_merge() -> None:
    merged = resolve(
        ("Quorvale Institute of Technology", "ORGANIZATION"),
        ("QIT", "ORGANIZATION"),
        ("Velltrix Systems", "ORGANIZATION"),
        ("Velltrix", "ORGANIZATION"),
        ("Mara Okafor", "PERSON"),
        ("Okafor", "PERSON"),
        ("Halcyon Sensor", "PRODUCT"),
        ("Halcyon Sensors", "PRODUCT"),
    )
    assert merged["QIT"] == "Quorvale Institute of Technology"
    assert merged["Velltrix"] == "Velltrix Systems"
    assert merged["Okafor"] == "Mara Okafor"
    assert merged["Halcyon Sensors"] == merged["Halcyon Sensor"]


def test_places_never_merge_into_organisations_and_ambiguity_blocks_merges() -> None:
    merged = resolve(
        ("Quorvale", "LOCATION"),
        ("Quorvale Dynamics", "ORGANIZATION"),
        ("Lumen", "CONCEPT"),
        ("Lumen Assay", "ORGANIZATION"),
        ("Lumen Q3", "PRODUCT"),
        ("Tomas Rehn", "PERSON"),
        ("Tobiah Renn", "PERSON"),
    )
    assert merged["Quorvale"] == "Quorvale"
    assert merged["Lumen"] == "Lumen"
    assert merged["Tomas Rehn"] != merged["Tobiah Renn"]


def test_explicit_acronym_in_text() -> None:
    merged = resolve(("Carro Data Bureau", "ORGANIZATION"), ("CDB", "CONCEPT"),
                     text="The Carro Data Bureau (CDB) opened.")
    assert merged["CDB"] == "Carro Data Bureau"


def test_relation_roles_vote_on_types() -> None:
    relations = [
        RelationMention(source="Ilse Varga", target="Kestrelmoor", type="BORN_IN", chunk_id="d#0", evidence="e",
                        start=0, end=1, method="passive"),
        RelationMention(source="Ilse Varga", target="Kestrelmoor Polytechnic", type="STUDIED_AT", chunk_id="d#0",
                        evidence="e", start=0, end=1, method="prep"),
    ]
    extraction = ChunkExtraction(
        chunk_id="d#0",
        entities=[mention("Ilse Varga", "ORGANIZATION"), mention("Kestrelmoor", "ORGANIZATION"),
                  mention("Kestrelmoor Polytechnic")],
        relations=relations,
    )
    entities = {e.name: e for e in EntityResolver().resolve([extraction]).entities.values()}
    assert entities["Ilse Varga"].type == "PERSON"
    assert entities["Kestrelmoor"].type == "LOCATION"
    assert len({e.id for e in entities.values()}) == 3


def _tiny_graph() -> KnowledgeGraph:
    doc = document_from_text("acme", "# Acme\n\nAcme Robotics is based in Lindport. Jane Doyle founded Acme Robotics.")
    chunks = chunk_document(doc)
    cid = chunks[0].id
    relations = [
        RelationMention(source="Acme Robotics", target="Lindport", type="LOCATED_IN", chunk_id=cid,
                        evidence="Acme Robotics is based in Lindport.", start=8, end=44, method="passive"),
        RelationMention(source="Jane Doyle", target="Acme Robotics", type="FOUNDED", chunk_id=cid,
                        evidence="Jane Doyle founded Acme Robotics.", start=45, end=78, method="svo"),
        RelationMention(source="Jane Doyle", target="Lindport", type=COOCCURRENCE, chunk_id=cid, evidence="x",
                        start=0, end=1, weight=0.2, method="cooccurrence"),
    ]
    extraction = ChunkExtraction(
        chunk_id=cid,
        entities=[mention("Acme Robotics", chunk=cid), mention("Lindport", "LOCATION", cid),
                  mention("Jane Doyle", "PERSON", cid)],
        relations=relations,
    )
    return KnowledgeGraph.build([doc], chunks, [extraction], EntityResolver().resolve([extraction], chunks))


def test_graph_keeps_provenance_on_every_edge() -> None:
    kg = _tiny_graph()
    assert kg.stats()["typed_relations"] == 2 and kg.stats()["cooccurrence_relations"] == 1
    for relation in kg.relations():
        assert relation.chunk_id == "acme#0" and relation.doc_id == "acme" and relation.evidence
    assert kg.lookup("acme robotics") == "acme-robotics"
    assert set(kg.chunk_entities["acme#0"]) == {"acme-robotics", "lindport", "jane-doyle"}
    undirected = kg.undirected()
    assert undirected["jane-doyle"]["lindport"]["weight"] == pytest.approx(0.2)
    assert not kg.undirected(include_cooccurrence=False).has_edge("jane-doyle", "lindport")
    best = kg.relations_between("lindport", "acme-robotics")[0]
    assert best.type == "LOCATED_IN" and best.source == "acme-robotics"


def test_json_and_graphml_round_trips(tmp_path: Path) -> None:
    kg = _tiny_graph()
    loaded = KnowledgeGraph.load_json(kg.save_json(tmp_path / "g.json"))
    assert loaded.to_dict() == kg.to_dict()
    graph = load_graphml(kg.save_graphml(tmp_path / "g.graphml"))
    assert set(graph.nodes) == set(kg.graph.nodes)
    assert graph.number_of_edges() == kg.graph.number_of_edges()
    assert graph.nodes["acme-robotics"]["chunk_ids"] == ["acme#0"]
    assert {d["type"] for _, _, d in graph.edges(data=True)} == {"LOCATED_IN", "FOUNDED", COOCCURRENCE}


def test_graph_rejects_dangling_relations_and_bad_formats() -> None:
    kg = KnowledgeGraph()
    kg.add_entity(Entity(id="a", name="A", type="CONCEPT"))
    with pytest.raises(KeyError):
        kg.add_relation(Relation(id="r1", source="a", target="b", type="X", weight=1, evidence="", chunk_id="c",
                                 doc_id="d", start=0, end=0, method="svo"))
    with pytest.raises(ValueError):
        KnowledgeGraph.from_dict({"format": "other"})


def test_scan_mentions_matches_whole_names_only() -> None:
    doc = document_from_text("d", "Acme Robotics hired staff. Acmeville is elsewhere. ACME ROBOTICS shouts.")
    entity = Entity(id="acme", name="Acme Robotics", type="ORGANIZATION", aliases=["Acme"])
    found = scan_mentions(chunk_document(doc), [entity])
    assert found == {"d#0": {"acme"}}
    assert scan_mentions(chunk_document(document_from_text("e", "Nothing here.")), [entity]) == {}
