"""Pydantic data contracts shared by every stage of the pipeline.

Provenance is a first-class field everywhere: chunks keep their character offsets in the source document,
mentions and relations keep the chunk they came from, and relations keep the evidence sentence with its span.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

EntityType = Literal["PERSON", "ORGANIZATION", "LOCATION", "PRODUCT", "EVENT", "CONCEPT"]
ENTITY_TYPES: tuple[str, ...] = ("PERSON", "ORGANIZATION", "LOCATION", "PRODUCT", "EVENT", "CONCEPT")

#: Relation type used for low-weight "appear in the same sentence" edges.
COOCCURRENCE = "RELATED_TO"


# --------------------------------------------------------------------------- ingestion
class Document(BaseModel):
    """A source document. ``text`` is the raw file content with newlines normalised to ``\\n``."""

    id: str
    title: str
    text: str
    source: str = ""


class Chunk(BaseModel):
    """A contiguous slice of a document: ``document.text[start:end] == text``."""

    id: str
    doc_id: str
    index: int
    start: int
    end: int
    text: str
    title: str = ""


# --------------------------------------------------------------------------- extraction
class EntityMention(BaseModel):
    """One surface mention of an entity inside a chunk (offsets are relative to the chunk text)."""

    name: str
    type: EntityType
    chunk_id: str
    start: int | None = None
    end: int | None = None
    description: str = ""
    method: str = "ner"


class RelationMention(BaseModel):
    """A relation between two surface names, with the evidence span that supports it."""

    source: str
    target: str
    type: str
    chunk_id: str
    evidence: str
    start: int
    end: int
    weight: float = 1.0
    method: str


class ChunkExtraction(BaseModel):
    """Everything an extractor found in one chunk."""

    chunk_id: str
    entities: list[EntityMention] = Field(default_factory=list)
    relations: list[RelationMention] = Field(default_factory=list)


# --------------------------------------------------------------------------- graph
class Entity(BaseModel):
    """A resolved (canonical) entity: one graph node."""

    id: str
    name: str
    type: EntityType
    aliases: list[str] = Field(default_factory=list)
    description: str = ""
    chunk_ids: list[str] = Field(default_factory=list)
    mention_count: int = 0


class Relation(BaseModel):
    """A typed, directed graph edge between two canonical entities, with provenance."""

    id: str
    source: str
    target: str
    type: str
    weight: float
    evidence: str
    chunk_id: str
    doc_id: str
    start: int
    end: int
    method: str

    @property
    def is_cooccurrence(self) -> bool:
        return self.type == COOCCURRENCE


class Community(BaseModel):
    """A Louvain community at one resolution level (level 0 is the coarsest)."""

    id: str
    level: int
    resolution: float
    members: list[str]
    parent: str | None = None


class Finding(BaseModel):
    """One statement in a community report, with the chunks that support it."""

    text: str
    chunk_ids: list[str] = Field(default_factory=list)


class CommunityReport(BaseModel):
    """A readable summary of one community, used by global search."""

    community_id: str
    level: int
    title: str
    summary: str
    findings: list[Finding] = Field(default_factory=list)
    top_entities: list[str] = Field(default_factory=list)
    importance: float = 0.0
    method: str = "extractive"

    def as_text(self) -> str:
        """The report as one plain-text block (used for relevance ranking)."""
        lines = [self.title, self.summary, *(f.text for f in self.findings)]
        return "\n".join(line for line in lines if line)


# --------------------------------------------------------------------------- search results
class LinkedEntity(BaseModel):
    """An entity recognised in a query."""

    entity_id: str
    name: str
    type: EntityType
    matched: str
    score: float


class EvidenceHit(BaseModel):
    """A retrieved chunk with its score and the entities it mentions."""

    chunk_id: str
    doc_id: str
    score: float
    text: str
    entities: list[str] = Field(default_factory=list)


class TraversalEdge(BaseModel):
    """One edge of the reasoning tree shown for a local search, backed by a retrieved chunk."""

    source: str
    source_name: str
    target: str
    target_name: str
    relation: str
    forward: bool
    chunk_id: str
    depth: int


class LocalSearchResult(BaseModel):
    query: str
    retriever: str
    seeds: list[LinkedEntity] = Field(default_factory=list)
    hits: list[EvidenceHit] = Field(default_factory=list)
    traversal: list[TraversalEdge] = Field(default_factory=list)
    answer: str | None = None


class PathHop(BaseModel):
    """One hop of an explained path. ``forward`` is False when the edge points against travel direction."""

    source: str
    source_name: str
    target: str
    target_name: str
    relation: str
    forward: bool
    chunk_id: str
    evidence: str
    weight: float


class EntityPath(BaseModel):
    hops: list[PathHop]
    cost: float


class PathSearchResult(BaseModel):
    query: str
    source: LinkedEntity | None = None
    target: LinkedEntity | None = None
    paths: list[EntityPath] = Field(default_factory=list)
    message: str = ""


class GlobalPoint(BaseModel):
    text: str
    community_id: str
    score: float
    chunk_ids: list[str] = Field(default_factory=list)


class RankedReport(BaseModel):
    community_id: str
    title: str
    score: float


class GlobalSearchResult(BaseModel):
    query: str
    level: int
    reports: list[RankedReport] = Field(default_factory=list)
    points: list[GlobalPoint] = Field(default_factory=list)
    answer: str | None = None
    method: str = "extractive"

