"""Extraction quality against a small hand-labelled gold set.

The pipeline output for each gold document is compared with the annotation:

* **entities**: the resolved entities linked to the document's chunks vs the gold entities (matched by name or
  alias), untyped (was it found?) and typed (was it found with the right type?);
* **relations**: typed relations whose evidence is in the document vs gold triples, scored *typed and directed*
  and *untyped and undirected* (are the right two entities connected at all?);
* **co-occurrence edges**: how many of them link a pair that the gold set says is related (their precision), and
  how many gold pairs are connected by a typed or co-occurrence edge (connectivity recall).

Scores are micro-averaged over documents. Per-pattern precision shows which dependency patterns are reliable.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from pydantic import BaseModel, Field

from ..data import load_json
from ..graph import KnowledgeGraph
from ..schemas import COOCCURRENCE

SYMMETRIC = frozenset({"PARTNERED_WITH"})


class GoldEntity(BaseModel):
    name: str
    type: str
    aliases: list[str] = Field(default_factory=list)


class GoldDocument(BaseModel):
    entities: list[GoldEntity]
    relations: list[tuple[str, str, str]]


def load_gold(name: str = "gold_extraction.json") -> dict[str, GoldDocument]:
    """A bundled extraction gold set, keyed by document id (``gold_extraction_heldout.json`` for the held-out one)."""
    return {doc: GoldDocument(**data) for doc, data in load_json(name)["documents"].items()}


def prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4), "tp": tp, "fp": fp,
            "fn": fn}


def _triple(source: str, rel_type: str, target: str) -> tuple[str, str, str]:
    if rel_type in SYMMETRIC and target < source:
        source, target = target, source
    return source, rel_type, target


def evaluate_extraction(kg: KnowledgeGraph, gold: dict[str, GoldDocument]) -> dict[str, Any]:
    """Score the built graph against the gold annotation (see the module docstring)."""
    counts: Counter[str] = Counter()
    by_method: dict[str, Counter[str]] = {}
    per_document: dict[str, dict[str, int]] = {}
    for doc_id, annotation in gold.items():
        chunk_ids = {c for c, chunk in kg.chunks.items() if chunk.doc_id == doc_id}
        gold_ids: dict[str, str] = {}  # entity id -> gold type
        missing = 0
        for entity in annotation.entities:
            entity_id = next((i for n in (entity.name, *entity.aliases) if (i := kg.lookup(n))), None)
            if entity_id is None:
                missing += 1
            else:
                gold_ids[entity_id] = entity.type
        predicted = {e for c in chunk_ids for e in kg.chunk_entities.get(c, [])}
        found = predicted & set(gold_ids)
        typed_found = {e for e in found if kg.graph.nodes[e]["type"] == gold_ids[e]}
        counts["ent_tp"] += len(found)
        counts["ent_fp"] += len(predicted - set(gold_ids))
        counts["ent_fn"] += len(set(gold_ids) - predicted) + missing
        counts["ent_typed_tp"] += len(typed_found)
        counts["ent_typed_fp"] += len(predicted) - len(typed_found)
        counts["ent_typed_fn"] += len(gold_ids) + missing - len(typed_found)

        gold_triples = set()
        for source, rel_type, target in annotation.relations:
            s, t = kg.lookup(source), kg.lookup(target)
            gold_triples.add(_triple(s or f"?{source}", rel_type, t or f"?{target}"))
        gold_pairs = {frozenset((s, t)) for s, _, t in gold_triples}
        edges = [r for r in kg.relations() if r.chunk_id in chunk_ids]
        typed = [r for r in edges if r.type != COOCCURRENCE]
        pred_triples = {_triple(r.source, r.type, r.target) for r in typed}
        pred_pairs = {frozenset((r.source, r.target)) for r in typed}
        co_pairs = {frozenset((r.source, r.target)) for r in edges if r.type == COOCCURRENCE}
        counts["rel_tp"] += len(pred_triples & gold_triples)
        counts["rel_fp"] += len(pred_triples - gold_triples)
        counts["rel_fn"] += len(gold_triples - pred_triples)
        counts["pair_tp"] += len(pred_pairs & gold_pairs)
        counts["pair_fp"] += len(pred_pairs - gold_pairs)
        counts["pair_fn"] += len(gold_pairs - pred_pairs)
        counts["co_correct"] += len(co_pairs & gold_pairs)
        counts["co_total"] += len(co_pairs)
        counts["connected"] += len(gold_pairs & (pred_pairs | co_pairs))
        counts["gold_pairs"] += len(gold_pairs)
        for r in typed:
            bucket = by_method.setdefault(r.method, Counter())
            bucket["total"] += 1
            bucket["correct"] += int(_triple(r.source, r.type, r.target) in gold_triples)
        per_document[doc_id] = {
            "gold_relations": len(gold_triples),
            "predicted_relations": len(pred_triples),
            "correct_relations": len(pred_triples & gold_triples),
        }
    return {
        "documents": len(gold),
        "gold_entities": sum(len(d.entities) for d in gold.values()),
        "gold_relations": sum(len(d.relations) for d in gold.values()),
        "entities": prf(counts["ent_tp"], counts["ent_fp"], counts["ent_fn"]),
        "entities_typed": prf(counts["ent_typed_tp"], counts["ent_typed_fp"], counts["ent_typed_fn"]),
        "relations_typed": prf(counts["rel_tp"], counts["rel_fp"], counts["rel_fn"]),
        "relations_untyped": prf(counts["pair_tp"], counts["pair_fp"], counts["pair_fn"]),
        "cooccurrence_precision": round(counts["co_correct"] / counts["co_total"], 4) if counts["co_total"] else 0.0,
        "cooccurrence_edges": counts["co_total"],
        "connectivity_recall": round(counts["connected"] / counts["gold_pairs"], 4) if counts["gold_pairs"] else 0.0,
        "pattern_precision": {
            method: {"correct": c["correct"], "total": c["total"], "precision": round(c["correct"] / c["total"], 4)}
            for method, c in sorted(by_method.items())
        },
        "per_document": per_document,
    }
