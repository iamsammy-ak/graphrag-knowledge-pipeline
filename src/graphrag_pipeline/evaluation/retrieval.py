"""Multi-hop evidence retrieval benchmark: vector-only vs graph local search vs hybrid.

Each question lists its supporting chunks hop by hop (any chunk in a hop's list supports that hop). For a ranking
of chunk ids and a cut-off k:

* **support recall@k**: the fraction of hops with at least one supporting chunk in the top k;
* **full support@k**: 1 if every hop is covered in the top k, else 0 (the question is answerable from the context).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel

from ..data import load_json
from ..embeddings import Embedder
from ..graph import KnowledgeGraph
from ..search.local import LocalSearchConfig, LocalSearcher, Retriever

DEFAULT_KS: tuple[int, ...] = (3, 5, 10)


class BenchQuestion(BaseModel):
    id: str
    question: str
    answer: str
    hops: int
    split: str
    support: list[list[str]]


def load_questions() -> list[BenchQuestion]:
    """The bundled multi-hop questions."""
    return [BenchQuestion(**q) for q in load_json("questions.json")["questions"]]


def support_recall(ranked: Sequence[str], support: Sequence[Sequence[str]], k: int) -> float:
    """Fraction of hops with at least one supporting chunk among the first ``k`` ranked ids."""
    if not support:
        return 0.0
    top = set(ranked[:k])
    return sum(1 for hop in support if top.intersection(hop)) / len(support)


def full_support(ranked: Sequence[str], support: Sequence[Sequence[str]], k: int) -> bool:
    """True when every hop is covered among the first ``k`` ranked ids."""
    top = set(ranked[:k])
    return bool(support) and all(top.intersection(hop) for hop in support)


def _summary(rows: Sequence[dict[str, Any]], method: str, ks: Sequence[int]) -> dict[str, Any]:
    out: dict[str, Any] = {"questions": len(rows)}
    for k in ks:
        out[f"recall@{k}"] = round(sum(r[method][f"recall@{k}"] for r in rows) / max(len(rows), 1), 4)
        out[f"full@{k}"] = round(sum(r[method][f"full@{k}"] for r in rows) / max(len(rows), 1), 4)
    return out


def run_retrieval_benchmark(
    kg: KnowledgeGraph,
    questions: Sequence[BenchQuestion],
    embedder: Embedder,
    *,
    ks: Sequence[int] = DEFAULT_KS,
    variants: dict[str, tuple[Retriever, LocalSearchConfig]] | None = None,
) -> dict[str, Any]:
    """Rank chunks for every question with each variant and score the rankings against the gold support.

    ``variants`` maps a display name to (retriever, config); the default compares the three retrievers with the
    default configuration.
    """
    variants = variants or {name: (name, LocalSearchConfig()) for name in ("vector", "graph", "hybrid")}
    searchers: dict[LocalSearchConfig, LocalSearcher] = {}
    rows: list[dict[str, Any]] = []
    for q in questions:
        row: dict[str, Any] = {"id": q.id, "hops": q.hops, "split": q.split}
        for name, (retriever, config) in variants.items():
            searcher = searchers.get(config)
            if searcher is None:
                searcher = searchers[config] = LocalSearcher(kg, embedder, config)
            ranked = [h.chunk_id for h in searcher.rank(q.question, retriever).hits][: max(ks)]
            row[name] = {"ranked": ranked}
            for k in ks:
                row[name][f"recall@{k}"] = support_recall(ranked, q.support, k)
                row[name][f"full@{k}"] = float(full_support(ranked, q.support, k))
        rows.append(row)
    groups: dict[str, list[dict[str, Any]]] = {
        "all": rows,
        "test": [r for r in rows if r["split"] == "test"],
        "dev": [r for r in rows if r["split"] == "dev"],
        "2-hop": [r for r in rows if r["hops"] == 2],
        "3-hop": [r for r in rows if r["hops"] == 3],
    }
    return {
        "embedder": embedder.name,
        "ks": list(ks),
        "methods": list(variants),
        "summary": {g: {m: _summary(rs, m, ks) for m in variants} for g, rs in groups.items() if rs},
        "per_question": rows,
    }
