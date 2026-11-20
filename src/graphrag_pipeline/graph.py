"""The knowledge graph: a ``networkx.MultiDiGraph`` of canonical entities with provenance on every node and edge.

* Nodes are entities (id, name, type, aliases, description, the chunks that mention them).
* Edges are relations (type, weight, evidence sentence, chunk id, document id, character span, extraction method).
  A multigraph keeps every piece of evidence as its own edge, so two documents stating the same fact give two edges.
* The graph also keeps the chunks themselves and a chunk -> entities index, which is what retrieval needs.

It saves to one JSON file (everything, reloadable) and to GraphML (graph only, for Gephi, yEd or Neo4j imports).
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import networkx as nx

from .ingest import mask_markdown
from .resolution import Resolution, match_key
from .schemas import COOCCURRENCE, Chunk, ChunkExtraction, Document, Entity, Relation

FORMAT = "graphrag-pipeline/1"
_LIST_ATTRS = ("aliases", "chunk_ids")


class KnowledgeGraph:
    """Entities, relations and chunks, with the indices that local, path and global search need."""

    def __init__(self) -> None:
        self.graph: nx.MultiDiGraph = nx.MultiDiGraph()
        self.chunks: dict[str, Chunk] = {}
        self.documents: dict[str, dict[str, str]] = {}
        self.chunk_entities: dict[str, list[str]] = {}
        self.key_index: dict[str, str] = {}
        self._projection: dict[bool, nx.Graph] = {}

    # ------------------------------------------------------------------ construction
    @classmethod
    def build(
        cls,
        documents: Sequence[Document],
        chunks: Sequence[Chunk],
        extractions: Sequence[ChunkExtraction],
        resolution: Resolution,
    ) -> KnowledgeGraph:
        """Assemble the graph from resolved extractions and scan every chunk for known names."""
        kg = cls()
        kg.documents = {d.id: {"title": d.title, "source": d.source} for d in documents}
        kg.chunks = {c.id: c for c in chunks}
        kg.key_index = dict(resolution.key_to_id)
        for entity in resolution.entities.values():
            kg.add_entity(entity)
        mentions: dict[str, set[str]] = defaultdict(set)
        for extraction in extractions:
            for mention in extraction.entities:
                entity_id = resolution.entity_id(mention.name)
                if entity_id is not None:
                    mentions[extraction.chunk_id].add(entity_id)
        for chunk_id, found in scan_mentions(chunks, resolution.entities.values()).items():
            mentions[chunk_id] |= found
        counter = 0
        for extraction in extractions:
            chunk = kg.chunks[extraction.chunk_id]
            for rel in extraction.relations:
                source, target = resolution.entity_id(rel.source), resolution.entity_id(rel.target)
                if source is None or target is None or source == target:
                    continue
                counter += 1
                kg.add_relation(
                    Relation(
                        id=f"r{counter:05d}",
                        source=source,
                        target=target,
                        type=rel.type,
                        weight=rel.weight,
                        evidence=rel.evidence,
                        chunk_id=chunk.id,
                        doc_id=chunk.doc_id,
                        start=rel.start,
                        end=rel.end,
                        method=rel.method,
                    )
                )
                mentions[chunk.id] |= {source, target}
        kg.chunk_entities = {cid: sorted(mentions.get(cid, set())) for cid in kg.chunks}
        entity_chunks: dict[str, set[str]] = defaultdict(set)
        for chunk_id, ids in kg.chunk_entities.items():
            for entity_id in ids:
                entity_chunks[entity_id].add(chunk_id)
        for entity_id, chunk_ids in entity_chunks.items():
            kg.graph.nodes[entity_id]["chunk_ids"] = sorted(chunk_ids)
        return kg

    def add_entity(self, entity: Entity) -> None:
        self.graph.add_node(entity.id, **entity.model_dump(exclude={"id"}))
        self._projection.clear()

    def add_relation(self, relation: Relation) -> None:
        if relation.source not in self.graph or relation.target not in self.graph:
            raise KeyError(f"relation {relation.id} refers to an unknown entity")
        attrs = relation.model_dump(exclude={"id", "source", "target"})
        self.graph.add_edge(relation.source, relation.target, key=relation.id, **attrs)
        self._projection.clear()

    # ------------------------------------------------------------------ access
    def entity(self, entity_id: str) -> Entity:
        return Entity(id=entity_id, **self.graph.nodes[entity_id])

    def entities(self) -> list[Entity]:
        return [self.entity(n) for n in self.graph.nodes]

    def relations(self, include_cooccurrence: bool = True) -> list[Relation]:
        return [
            Relation(id=key, source=u, target=v, **data)
            for u, v, key, data in self.graph.edges(keys=True, data=True)
            if include_cooccurrence or data["type"] != COOCCURRENCE
        ]

    def relations_between(self, a: str, b: str) -> list[Relation]:
        """Relations in either direction between two entities, strongest first."""
        found: list[Relation] = []
        for u, v in ((a, b), (b, a)):
            for key, data in (self.graph.get_edge_data(u, v) or {}).items():
                found.append(Relation(id=key, source=u, target=v, **data))
        return sorted(found, key=lambda r: (-r.weight, r.type == COOCCURRENCE, r.id))

    def name(self, entity_id: str) -> str:
        return str(self.graph.nodes[entity_id]["name"])

    def lookup(self, name: str) -> str | None:
        """Entity id for a name or alias (normalised match)."""
        return self.key_index.get(match_key(name))

    def entity_chunks(self, entity_id: str) -> list[str]:
        return list(self.graph.nodes[entity_id].get("chunk_ids", []))

    def undirected(self, include_cooccurrence: bool = True) -> nx.Graph:
        """Weighted simple graph: edge weight is the summed weight of all relations between the pair."""
        if include_cooccurrence not in self._projection:
            projection = nx.Graph()
            projection.add_nodes_from(self.graph.nodes)
            for u, v, data in self.graph.edges(data=True):
                if not include_cooccurrence and data["type"] == COOCCURRENCE:
                    continue
                weight = float(data["weight"])
                if projection.has_edge(u, v):
                    projection[u][v]["weight"] += weight
                else:
                    projection.add_edge(u, v, weight=weight)
            self._projection[include_cooccurrence] = projection
        return self._projection[include_cooccurrence]

    def stats(self) -> dict[str, Any]:
        types: dict[str, int] = defaultdict(int)
        for _, data in self.graph.nodes(data=True):
            types[data["type"]] += 1
        methods: dict[str, int] = defaultdict(int)
        for _, _, data in self.graph.edges(data=True):
            methods[data["method"]] += 1
        typed = sum(1 for _, _, d in self.graph.edges(data=True) if d["type"] != COOCCURRENCE)
        return {
            "documents": len(self.documents),
            "chunks": len(self.chunks),
            "entities": self.graph.number_of_nodes(),
            "relations": self.graph.number_of_edges(),
            "typed_relations": typed,
            "cooccurrence_relations": self.graph.number_of_edges() - typed,
            "entity_types": dict(sorted(types.items())),
            "relation_methods": dict(sorted(methods.items())),
        }

    # ------------------------------------------------------------------ persistence
    def to_dict(self) -> dict[str, Any]:
        return {
            "format": FORMAT,
            "documents": self.documents,
            "chunks": [c.model_dump() for c in self.chunks.values()],
            "entities": [e.model_dump() for e in self.entities()],
            "relations": [r.model_dump() for r in self.relations()],
            "chunk_entities": self.chunk_entities,
            "key_index": self.key_index,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> KnowledgeGraph:
        if data.get("format") != FORMAT:
            raise ValueError(f"Unsupported graph format: {data.get('format')!r}")
        kg = cls()
        kg.documents = data["documents"]
        kg.chunks = {c["id"]: Chunk(**c) for c in data["chunks"]}
        for entity in data["entities"]:
            kg.add_entity(Entity(**entity))
        for relation in data["relations"]:
            kg.add_relation(Relation(**relation))
        kg.chunk_entities = {k: list(v) for k, v in data["chunk_entities"].items()}
        kg.key_index = dict(data["key_index"])
        return kg

    def save_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=1), encoding="utf-8")
        return path

    @classmethod
    def load_json(cls, path: str | Path) -> KnowledgeGraph:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def save_graphml(self, path: str | Path) -> Path:
        """GraphML export; list attributes are stored as JSON strings because GraphML has no list type."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        export = nx.MultiDiGraph()
        for node, data in self.graph.nodes(data=True):
            attrs = {k: (json.dumps(v) if k in _LIST_ATTRS else v) for k, v in data.items()}
            export.add_node(node, **attrs)
        for u, v, key, data in self.graph.edges(keys=True, data=True):
            export.add_edge(u, v, key=key, **data)
        nx.write_graphml(export, path)
        return path


def load_graphml(path: str | Path) -> nx.MultiDiGraph:
    """Read a GraphML export back into a ``MultiDiGraph`` with list attributes decoded."""
    graph = nx.read_graphml(path, force_multigraph=True)
    for _, data in graph.nodes(data=True):
        for attr in _LIST_ATTRS:
            if isinstance(data.get(attr), str):
                data[attr] = json.loads(data[attr])
    return graph


def scan_mentions(chunks: Iterable[Chunk], entities: Iterable[Entity]) -> dict[str, set[str]]:
    """Find every occurrence of every known name or alias in every chunk (case-sensitive, whole words).

    This recovers mentions NER missed in a given sentence, so a chunk is linked to every entity it names.
    """
    surface_to_id: dict[str, str] = {}
    for entity in entities:
        for surface in (entity.name, *entity.aliases):
            if len(surface) >= 3 and any(ch.isupper() for ch in surface):
                surface_to_id.setdefault(surface, entity.id)
    if not surface_to_id:
        return {}
    alternatives = "|".join(re.escape(s) for s in sorted(surface_to_id, key=len, reverse=True))
    pattern = re.compile(rf"(?<![\w-])(?:{alternatives})(?![\w-])")
    found: dict[str, set[str]] = defaultdict(set)
    for chunk in chunks:
        for match in pattern.finditer(mask_markdown(chunk.text)):
            found[chunk.id].add(surface_to_id[match.group(0)])
    return found
