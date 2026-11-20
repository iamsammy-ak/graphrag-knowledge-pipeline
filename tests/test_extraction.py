from __future__ import annotations

import pytest

from graphrag_pipeline.extraction import SpacyExtractor, get_extractor
from graphrag_pipeline.extraction.vocab import canonical_relation, refine_type
from graphrag_pipeline.ingest import chunk_document, document_from_text, mask_markdown
from graphrag_pipeline.schemas import COOCCURRENCE

from .conftest import requires_spacy


def triples(text: str, title: str = "Doc", cooccurrence: bool = True) -> set[tuple[str, str, str]]:
    doc = document_from_text("doc", f"# {title}\n\n{text}")
    extraction = SpacyExtractor(cooccurrence=cooccurrence).extract(chunk_document(doc))
    return {(r.source, r.type, r.target) for e in extraction for r in e.relations}


@requires_spacy
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Velltrix Systems acquired Nimbrel Labs in 2019.", ("Velltrix Systems", "ACQUIRED", "Nimbrel Labs")),
        ("Nimbrel Labs was acquired by Velltrix Systems.", ("Velltrix Systems", "ACQUIRED", "Nimbrel Labs")),
        ("Velltrix Systems is headquartered in Quorvale.", ("Velltrix Systems", "LOCATED_IN", "Quorvale")),
        (
            "Felix Ambrose, the chief executive of Greyfen Materials, spoke today.",
            ("Felix Ambrose", "LEADS", "Greyfen Materials"),
        ),
        ("Lena Morrow is the chief executive of Pinecrest Bank.", ("Lena Morrow", "LEADS", "Pinecrest Bank")),
        ("Harrowgate Capital invested in Corvane Motors.", ("Harrowgate Capital", "INVESTED_IN", "Corvane Motors")),
        ("The Wayfarer is a truck powered by Ember-9 cells.", ("Wayfarer", "USES", "Ember-9")),
    ],
)
def test_dependency_patterns(text: str, expected: tuple[str, str, str]) -> None:
    assert expected in triples(text)


@requires_spacy
def test_pronouns_and_definite_descriptions_resolve_to_earlier_entities() -> None:
    text = (
        "Mara Okafor was born in Dunmarrow. She studied physics at the Quorvale Institute of Technology. "
        "Brightwell Cells is a battery manufacturer. The company buys its electrolyte from Greyfen Materials."
    )
    found = triples(text)
    assert ("Mara Okafor", "STUDIED_AT", "Quorvale Institute of Technology") in found
    assert ("Greyfen Materials", "SUPPLIES", "Brightwell Cells") in found


@requires_spacy
def test_possessive_role_and_relative_clause() -> None:
    found = triples(
        "Ostrel Energy is a storage company. Its chief executive is Ravi Castellan, who joined from Brightwell Cells."
    )
    assert ("Ravi Castellan", "LEADS", "Ostrel Energy") in found
    assert ("Ravi Castellan", "WORKS_FOR", "Brightwell Cells") in found


@requires_spacy
def test_cooccurrence_fallback_is_low_weight_and_optional() -> None:
    doc = document_from_text("d", "Velltrix Systems and Quorvale were mentioned alongside Calder Optics at the fair.")
    extraction = SpacyExtractor().extract(chunk_document(doc))[0]
    co = [r for r in extraction.relations if r.type == COOCCURRENCE]
    assert co and all(r.weight == pytest.approx(0.2) and r.method == "cooccurrence" for r in co)
    without = SpacyExtractor(cooccurrence=False).extract(chunk_document(doc))[0]
    assert not [r for r in without.relations if r.type == COOCCURRENCE]


@requires_spacy
def test_mentions_and_evidence_carry_valid_offsets() -> None:
    doc = document_from_text("d", "# Acme\n\nAcme Robotics is headquartered in Lindport. **Jane Doyle** founded it.")
    chunk = chunk_document(doc)[0]
    extraction = SpacyExtractor().extract([chunk])[0]
    masked = mask_markdown(chunk.text)
    assert extraction.entities
    for mention in extraction.entities:
        assert mention.start is not None and mention.end is not None
        assert masked[mention.start : mention.end] == mention.name
    for relation in extraction.relations:
        assert " ".join(masked[relation.start : relation.end].split()) == relation.evidence


@requires_spacy
def test_no_entities_means_no_relations() -> None:
    assert triples("the weather was mild and the river was calm.") == set()


def test_canonical_relation_vocabulary() -> None:
    assert canonical_relation("former_engineer_at") == ("WORKS_FOR", False)
    assert canonical_relation("power") == ("USES", True)
    assert canonical_relation("buy_from") == ("SUPPLIES", True)
    assert canonical_relation("company_in", "LOCATION") == ("LOCATED_IN", False)
    assert canonical_relation("company_in") == ("COMPANY_IN", False)
    assert canonical_relation("Chief Executive Of") == ("LEADS", False)


def test_refine_type_uses_surface_cues() -> None:
    assert refine_type("Nimbrel Labs", "PERSON") == "ORGANIZATION"
    assert refine_type("Vey Harbour", "ORGANIZATION") == "LOCATION"
    assert refine_type("Glint SDK", "ORGANIZATION") == "PRODUCT"
    assert refine_type("Ember-9", "CONCEPT") == "PRODUCT"
    assert refine_type("Mara Okafor", "PERSON") == "PERSON"


def test_get_extractor_selection() -> None:
    assert get_extractor("auto").name == "spacy-rules"
    with pytest.raises(ValueError):
        get_extractor("llm")
    with pytest.raises(ValueError):
        get_extractor("magic")
