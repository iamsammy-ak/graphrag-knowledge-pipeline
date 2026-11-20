"""The full benchmark: build the sample index, score retrieval variants and extraction, write JSON + Markdown."""

from __future__ import annotations

import json
import platform
from datetime import date
from pathlib import Path
from typing import Any

from ..data import corpus_dir, heldout_dir
from ..embeddings import Embedder
from ..pipeline import BuildConfig, GraphIndex, build_index
from ..search.local import LocalSearchConfig, Retriever
from .extraction import evaluate_extraction, load_gold
from .retrieval import load_questions, run_retrieval_benchmark

#: Retrieval variants: the three methods compared, plus two ablations that show where the graph gain comes from.
VARIANTS: dict[str, tuple[Retriever, LocalSearchConfig]] = {
    "vector": ("vector", LocalSearchConfig()),
    "graph-ppr": ("graph", LocalSearchConfig(expansion="ppr")),
    "graph-unmasked": ("graph", LocalSearchConfig(mask_entities=False)),
    "graph": ("graph", LocalSearchConfig()),
    "hybrid": ("hybrid", LocalSearchConfig()),
}
DESCRIPTIONS = {
    "vector": "dense top-k over chunks (title-prefixed), cosine similarity",
    "graph-ppr": "ablation: structural personalised PageRank from the linked entities (question used only for linking)",
    "graph-unmasked": "ablation: relation-aware beam search, but edge relevance computed without masking entity names",
    "graph": "graph local search: relation-aware beam search over relation paths (entity names masked)",
    "hybrid": "weighted reciprocal rank fusion of graph (weight 3) and vector (weight 1), k = 10",
}


def slug(name: str) -> str:
    return name.lower().replace("/", "-").replace(" ", "-")


def run_benchmark(
    embedder: Embedder, *, index: GraphIndex | None = None, heldout: bool = True, command: str = ""
) -> dict[str, Any]:
    """Run retrieval (all variants) and extraction (development + held-out gold sets) benchmarks."""
    index = index or build_index(corpus_dir(), BuildConfig(extractor="spacy"))
    retrieval = run_retrieval_benchmark(index.kg, load_questions(), embedder, variants=VARIANTS)
    extraction = {"development": evaluate_extraction(index.kg, load_gold())}
    if heldout:
        held = build_index(heldout_dir(), BuildConfig(extractor="spacy"))
        extraction["heldout"] = evaluate_extraction(held.kg, load_gold("gold_extraction_heldout.json"))
    return {
        "date": date.today().isoformat(),
        "command": command,
        "python": platform.python_version(),
        "corpus": {
            "documents": index.manifest["stats"]["documents"],
            "chunks": index.manifest["stats"]["chunks"],
            "entities": index.manifest["stats"]["entities"],
            "typed_relations": index.manifest["stats"]["typed_relations"],
            "cooccurrence_relations": index.manifest["stats"]["cooccurrence_relations"],
            "extractor": index.manifest["extractor"],
        },
        "variants": DESCRIPTIONS,
        "retrieval": retrieval,
        "extraction": extraction,
    }


def _retrieval_table(result: dict[str, Any], group: str) -> list[str]:
    ks = result["ks"]
    summary = result["summary"][group]
    header = "| method | " + " | ".join(f"recall@{k}" for k in ks) + " | " + " | ".join(f"full@{k}" for k in ks) + " |"
    lines = [header, "|---|" + "---:|" * (2 * len(ks))]
    for method, scores in summary.items():
        cells = [f"{scores[f'recall@{k}']:.2f}" for k in ks] + [f"{scores[f'full@{k}']:.2f}" for k in ks]
        lines.append(f"| {method} | " + " | ".join(cells) + " |")
    return lines


def _extraction_table(result: dict[str, Any]) -> list[str]:
    lines = ["| measure | precision | recall | F1 |", "|---|---:|---:|---:|"]
    for key, label in (
        ("entities", "entities found"),
        ("entities_typed", "entities with the right type"),
        ("relations_typed", "relations, typed and directed"),
        ("relations_untyped", "relations, any type or direction"),
    ):
        row = result[key]
        lines.append(f"| {label} | {row['precision']:.2f} | {row['recall']:.2f} | {row['f1']:.2f} |")
    lines.append("")
    lines.append(
        f"Co-occurrence edges: {result['cooccurrence_edges']}, of which {result['cooccurrence_precision']:.0%} link a "
        f"gold-related pair. Gold pairs connected by any edge (typed or co-occurrence): "
        f"{result['connectivity_recall']:.0%}."
    )
    patterns = ", ".join(
        f"{m} {p['correct']}/{p['total']}" for m, p in result["pattern_precision"].items()
    )
    lines.append(f"Correct typed relations per pattern: {patterns}.")
    return lines


def to_markdown(data: dict[str, Any]) -> str:
    retrieval = data["retrieval"]
    corpus = data["corpus"]
    n_test = retrieval["summary"]["test"]["vector"]["questions"]
    n_all = retrieval["summary"]["all"]["vector"]["questions"]
    lines = [
        f"# Benchmark results ({retrieval['embedder']})",
        "",
        f"Reproduce: `{data['command']}`  ",
        f"Run on {data['date']} with Python {data['python']}.",
        "",
        "## Data",
        "",
        f"Synthetic, fictional corpus bundled with the package: {corpus['documents']} documents, "
        f"{corpus['chunks']} chunks. Graph built offline by `{corpus['extractor']}`: {corpus['entities']} entities, "
        f"{corpus['typed_relations']} typed relations, {corpus['cooccurrence_relations']} co-occurrence links.",
        f"{n_all} hand-written multi-hop questions (2-3 hops) with gold supporting chunks per hop. "
        f"Settings were chosen on the 10 `dev` questions; the {n_test} `test` questions were not used for tuning.",
        "",
        "Metrics: **recall@k** = share of hops with a supporting chunk in the top k; **full@k** = share of questions "
        "with every hop covered in the top k.",
        "",
        "Variants:",
        "",
        *[f"- `{name}`: {text}" for name, text in data["variants"].items()],
        "",
        f"## Retrieval: test questions ({n_test})",
        "",
        *_retrieval_table(retrieval, "test"),
        "",
        f"## Retrieval: all questions ({n_all})",
        "",
        *_retrieval_table(retrieval, "all"),
        "",
    ]
    for group in ("2-hop", "3-hop"):
        if group in retrieval["summary"]:
            count = retrieval["summary"][group]["vector"]["questions"]
            lines += [f"### {group} questions ({count})", "", *_retrieval_table(retrieval, group), ""]
    extraction = data["extraction"]
    dev = extraction["development"]
    lines += [
        "## Extraction quality (offline spaCy extractor)",
        "",
        f"### Development gold set ({dev['documents']} documents, {dev['gold_entities']} entities, "
        f"{dev['gold_relations']} relations)",
        "",
        "The extraction rules were written while reading this corpus, so these numbers are optimistic.",
        "",
        *_extraction_table(dev),
        "",
    ]
    if "heldout" in extraction:
        held = extraction["heldout"]
        lines += [
            f"### Held-out gold set ({held['documents']} documents, {held['gold_entities']} entities, "
            f"{held['gold_relations']} relations)",
            "",
            "New documents in a different style, written and labelled after the rules were frozen. This is the "
            "honest estimate of the extractor on unseen text.",
            "",
            *_extraction_table(held),
            "",
        ]
    return "\n".join(lines)


def write_results(data: dict[str, Any], out_dir: str | Path) -> tuple[Path, Path]:
    """Write ``benchmark-<embedder>.json`` and ``.md`` (per-question rankings are kept in the JSON)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    name = f"benchmark-{slug(data['retrieval']['embedder'].split('/')[-1])}"
    json_path = out / f"{name}.json"
    md_path = out / f"{name}.md"
    json_path.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    md_path.write_text(to_markdown(data) + "\n", encoding="utf-8")
    return json_path, md_path
