"""Regenerate the README figures from the bundled corpus (offline).

    python benchmarks/make_assets.py            # graph.png, demo-local.svg, demo-path.svg in assets/

The benchmark chart comes from `graphrag bench --embedder bge --chart assets/benchmark.png`.
"""

from __future__ import annotations

import io
from pathlib import Path

from rich.console import Console

from graphrag_pipeline import BuildConfig, LocalSearcher, build_index, corpus_dir, get_embedder, path_search
from graphrag_pipeline.cli import DEMO_PATH, DEMO_QUESTION
from graphrag_pipeline.render import render_local, render_path, save_svg, stats_panel
from graphrag_pipeline.viz import render_graph_png

ASSETS = Path(__file__).resolve().parent.parent / "assets"


def recorder() -> Console:
    """A 100-column console that records to memory (UTF-8, so box drawing is not downgraded to ASCII)."""
    return Console(record=True, width=100, file=io.StringIO(), force_terminal=True, color_system="truecolor")


def main() -> None:
    index = build_index(corpus_dir(), BuildConfig(extractor="spacy"))
    print("graph:", render_graph_png(index, ASSETS / "graph.png"))

    console = recorder()
    console.print(stats_panel(index, title="Offline build of the sample corpus"))
    result = LocalSearcher(index.kg, get_embedder("auto")).search(DEMO_QUESTION, k=3, retriever="graph")
    render_local(console, index.kg, result)
    save_svg(console, ASSETS / "demo-local.svg", "graphrag query --mode local")

    console = recorder()
    render_path(console, index.kg, path_search(index.kg, DEMO_PATH, max_paths=2))
    save_svg(console, ASSETS / "demo-path.svg", "graphrag query --mode path")
    print("terminal SVGs written")


if __name__ == "__main__":
    main()
