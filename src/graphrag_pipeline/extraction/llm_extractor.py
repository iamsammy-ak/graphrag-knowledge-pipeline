"""LLM extraction into a Pydantic schema, with every relation grounded in the chunk text.

The model returns entities (name, type, description) and relations (source, target, type, evidence quote). The quote
is then located in the chunk; a relation whose evidence cannot be found is dropped rather than trusted, so every
LLM edge in the graph points at real text. Used only when an API key is configured.
"""

from __future__ import annotations

import difflib
import re
from collections.abc import Sequence
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field, field_validator

from ..llm import LLM, LLMError, complete_structured
from ..schemas import ENTITY_TYPES, Chunk, ChunkExtraction, EntityMention, EntityType, RelationMention
from .vocab import CANONICAL_RELATIONS, canonical_relation, normalise_key

if TYPE_CHECKING:
    from . import Extractor

_TYPE_SYNONYMS: dict[str, EntityType] = {
    "ORG": "ORGANIZATION",
    "ORGANISATION": "ORGANIZATION",
    "COMPANY": "ORGANIZATION",
    "INSTITUTION": "ORGANIZATION",
    "UNIVERSITY": "ORGANIZATION",
    "PER": "PERSON",
    "PEOPLE": "PERSON",
    "GPE": "LOCATION",
    "LOC": "LOCATION",
    "PLACE": "LOCATION",
    "CITY": "LOCATION",
    "COUNTRY": "LOCATION",
    "REGION": "LOCATION",
    "TECHNOLOGY": "PRODUCT",
    "SOFTWARE": "PRODUCT",
}


def coerce_entity_type(value: str) -> EntityType:
    """Map a model's free-form type label onto our entity types (unknown labels become CONCEPT)."""
    label = re.sub(r"[^A-Z]", "", (value or "").upper())
    if label in ENTITY_TYPES:
        return label  # type: ignore[return-value]
    return _TYPE_SYNONYMS.get(label, "CONCEPT")


class LLMEntityItem(BaseModel):
    name: str
    type: str = "CONCEPT"
    description: str = ""

    @field_validator("name", "description", mode="before")
    @classmethod
    def _strip(cls, value: object) -> str:
        return str(value or "").strip()


class LLMRelationItem(BaseModel):
    source: str
    target: str
    type: str
    evidence: str = ""

    @field_validator("source", "target", "type", "evidence", mode="before")
    @classmethod
    def _strip(cls, value: object) -> str:
        return str(value or "").strip()


class LLMExtractionPayload(BaseModel):
    """The JSON shape the model must return for one chunk."""

    entities: list[LLMEntityItem] = Field(default_factory=list)
    relations: list[LLMRelationItem] = Field(default_factory=list)


SYSTEM_PROMPT = (
    "You extract a knowledge graph from a passage. Return named entities and the relations the passage states "
    "explicitly between them.\n"
    f"Entity types: {', '.join(ENTITY_TYPES)}.\n"
    f"Relation types (direction source -> target): {', '.join(CANONICAL_RELATIONS)}. "
    "Use another UPPER_SNAKE_CASE type only when none fits.\n"
    "For every relation, `evidence` must be an exact quote from the passage that states it. "
    "Do not infer facts that the passage does not state. Resolve pronouns to the entity they refer to."
)


def locate_evidence(text: str, evidence: str) -> tuple[int, int] | None:
    """Find a model's evidence quote in the chunk: exact, case-insensitive, whitespace-tolerant, then fuzzy."""
    evidence = evidence.strip().strip('"').strip()
    if not evidence:
        return None
    index = text.find(evidence)
    if index >= 0:
        return index, index + len(evidence)
    index = text.lower().find(evidence.lower())
    if index >= 0:
        return index, index + len(evidence)
    pattern = r"\s+".join(re.escape(word) for word in evidence.split())
    match = re.search(pattern, text, re.I)
    if match:
        return match.start(), match.end()
    matcher = difflib.SequenceMatcher(None, text.lower(), evidence.lower(), autojunk=False)
    block = matcher.find_longest_match(0, len(text), 0, len(evidence))
    if block.size >= max(12, int(0.8 * len(evidence))):
        return block.a, block.a + block.size
    return None


class LLMExtractor:
    """Structured LLM extraction with evidence grounding; falls back per chunk when the model fails."""

    name = "llm"

    def __init__(self, llm: LLM, fallback: Extractor | None = None) -> None:
        self.llm = llm
        self.fallback = fallback
        self.dropped_relations = 0
        self.failed_chunks = 0

    def _messages(self, chunk: Chunk) -> list[dict[str, str]]:
        context = f"Document: {chunk.title}\n\n" if chunk.title else ""
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{context}Passage:\n{chunk.text}"},
        ]

    def extract_one(self, chunk: Chunk) -> ChunkExtraction:
        payload = complete_structured(self.llm, self._messages(chunk), LLMExtractionPayload)
        result = ChunkExtraction(chunk_id=chunk.id)
        known: dict[str, EntityMention] = {}
        for item in payload.entities:
            if not item.name or item.name.lower() in known:
                continue
            span = locate_evidence(chunk.text, item.name)
            mention = EntityMention(
                name=item.name,
                type=coerce_entity_type(item.type),
                chunk_id=chunk.id,
                start=span[0] if span else None,
                end=span[1] if span else None,
                description=item.description[:160],
                method="llm",
            )
            known[item.name.lower()] = mention
            result.entities.append(mention)
        for rel in payload.relations:
            if not rel.source or not rel.target or rel.source.lower() == rel.target.lower():
                continue
            span = locate_evidence(chunk.text, rel.evidence)
            if span is None:
                self.dropped_relations += 1
                continue
            for name in (rel.source, rel.target):
                if name.lower() not in known:
                    mention = EntityMention(name=name, type="CONCEPT", chunk_id=chunk.id, method="llm")
                    known[name.lower()] = mention
                    result.entities.append(mention)
            target_type = known[rel.target.lower()].type
            rel_type, reverse = canonical_relation(rel.type, target_type)
            if rel_type not in CANONICAL_RELATIONS:
                rel_type = normalise_key(rel.type).upper() or "RELATED_TO"
            source, target = (rel.target, rel.source) if reverse else (rel.source, rel.target)
            result.relations.append(
                RelationMention(
                    source=source,
                    target=target,
                    type=rel_type,
                    chunk_id=chunk.id,
                    evidence=" ".join(chunk.text[span[0] : span[1]].split()),
                    start=span[0],
                    end=span[1],
                    weight=1.0,
                    method="llm",
                )
            )
        return result

    def extract(self, chunks: Sequence[Chunk]) -> list[ChunkExtraction]:
        results: list[ChunkExtraction] = []
        for chunk in chunks:
            try:
                results.append(self.extract_one(chunk))
            except LLMError:
                self.failed_chunks += 1
                if self.fallback is None:
                    raise
                results.append(self.fallback.extract([chunk])[0])
        return results
