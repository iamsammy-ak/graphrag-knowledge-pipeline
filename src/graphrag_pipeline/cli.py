"""Command line: ``graphrag build | query | communities | bench | demo | serve``."""

from __future__ import annotations

import shlex
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from . import __version__
from .data import corpus_dir
from .embeddings import Embedder, get_embedder
from .llm import resolve_llm
from .pipeline import BuildConfig, GraphIndex, build_index
from .render import (
    render_communities,
    render_extraction,
    render_global,
    render_local,
    render_path,
    render_report,
    render_retrieval,
    save_svg,
    stats_panel,
)
from .search import LocalSearcher, global_search, path_search

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Unstructured text to knowledge graph, then local, path and global graph search.",
)

DEFAULT_INDEX = Path(".graphrag")
DEMO_QUESTION = "In which province is the company that makes the Halcyon Sensor headquartered?"
DEMO_PATH = "How is the Stackbot connected to Velltrix Systems?"
DEMO_GLOBAL = "Which companies work on batteries and energy storage, and where are they?"

IndexOption = Annotated[Path, typer.Option("--index", "-i", help="Index directory created by `graphrag build`.")]
OfflineOption = Annotated[bool, typer.Option("--offline", help="Never call an LLM, even if a key is configured.")]
EmbedderOption = Annotated[
    str, typer.Option("--embedder", "-e", help="auto (bge-small if cached, else hashing), bge or hashing.")
]
SvgOption = Annotated[Path | None, typer.Option("--svg", help="Also save the terminal output as an SVG.")]


def _console(svg: Path | None) -> Console:
    return Console(record=svg is not None, width=100 if svg is not None else None)


def _save_svg(console: Console, svg: Path | None, title: str) -> None:
    if svg is not None:
        save_svg(console, svg, title)


def _load(index: Path) -> GraphIndex:
    try:
        return GraphIndex.load(index)
    except FileNotFoundError as exc:
        raise typer.BadParameter(f"{exc}. Try: graphrag build --out {index}") from exc


def _embedder(kind: str) -> Embedder:
    try:
        return get_embedder(kind)
    except (ValueError, ImportError) as exc:
        raise typer.BadParameter(str(exc)) from exc


@app.command()
def build(
    source: Annotated[
        Path | None, typer.Argument(help="Folder or file of .md/.txt documents (default: the bundled corpus).")
    ] = None,
    out: Annotated[Path, typer.Option("--out", "-o", help="Where to write the index.")] = DEFAULT_INDEX,
    extractor: Annotated[str, typer.Option(help="auto, spacy or llm.")] = "auto",
    max_chars: Annotated[int, typer.Option(help="Longest chunk before a paragraph is split.")] = 800,
    cooccurrence: Annotated[bool, typer.Option(help="Add low-weight co-occurrence edges.")] = True,
    offline: OfflineOption = False,
) -> None:
    """Ingest documents, extract entities and relations, resolve, build communities and reports."""
    console = Console()
    llm = resolve_llm(offline)
    if extractor == "llm" and llm is None:
        raise typer.BadParameter("--extractor llm needs OPENROUTER_API_KEY (and no --offline)")
    config = BuildConfig(max_chars=max_chars, extractor=extractor, cooccurrence=cooccurrence)
    with console.status("Building the knowledge graph..."):
        index = build_index(source or corpus_dir(), config, llm=llm, progress=lambda m: console.log(m))
    index.save(out)
    console.print(stats_panel(index, title=f"Index saved to {out}"))


@app.command()
def query(
    question: Annotated[str, typer.Argument(help="The question (for path mode it should name two entities).")],
    index: IndexOption = DEFAULT_INDEX,
    mode: Annotated[str, typer.Option("--mode", "-m", help="local, global or path.")] = "local",
    retriever: Annotated[str, typer.Option(help="local mode: graph, vector or hybrid.")] = "hybrid",
    k: Annotated[int, typer.Option("--k", help="local mode: number of evidence chunks.")] = 5,
    source: Annotated[str | None, typer.Option("--from", help="path mode: start entity.")] = None,
    target: Annotated[str | None, typer.Option("--to", help="path mode: end entity.")] = None,
    level: Annotated[int | None, typer.Option(help="global mode: community level (default coarsest).")] = None,
    embedder: EmbedderOption = "auto",
    offline: OfflineOption = False,
    as_json: Annotated[bool, typer.Option("--json", help="Print the raw result as JSON.")] = False,
    svg: SvgOption = None,
) -> None:
    """Ask the graph a question in local, global or path mode."""
    if mode not in {"local", "global", "path"}:
        raise typer.BadParameter("--mode must be local, global or path")
    if retriever not in {"graph", "vector", "hybrid"}:
        raise typer.BadParameter("--retriever must be graph, vector or hybrid")
    graph_index = _load(index)
    llm = resolve_llm(offline)
    console = _console(svg)
    if mode == "path":
        result_path = path_search(graph_index.kg, question, source=source, target=target)
        if as_json:
            print(result_path.model_dump_json(indent=1))
            return
        render_path(console, graph_index.kg, result_path)
    elif mode == "global":
        result_global = global_search(graph_index.reports, question, level=level, embedder=_embedder(embedder), llm=llm)
        if as_json:
            print(result_global.model_dump_json(indent=1))
            return
        render_global(console, result_global)
    else:
        searcher = LocalSearcher(graph_index.kg, _embedder(embedder))
        result_local = searcher.search(question, k=k, retriever=retriever, llm=llm)  # type: ignore[arg-type]
        if as_json:
            print(result_local.model_dump_json(indent=1))
            return
        render_local(console, graph_index.kg, result_local)
    _save_svg(console, svg, f"graphrag query --mode {mode}")


@app.command()
def communities(
    index: IndexOption = DEFAULT_INDEX,
    level: Annotated[int, typer.Option(help="Community level (0 = coarsest).")] = 0,
    show: Annotated[str | None, typer.Option("--show", help="Print the report of one community, e.g. L0-C1.")] = None,
    png: Annotated[Path | None, typer.Option("--png", help="Render the graph, coloured by community.")] = None,
) -> None:
    """List communities at a level, show a community report, or render the graph as a PNG."""
    graph_index = _load(index)
    console = Console()
    if show:
        report = graph_index.report(show)
        if report is None:
            raise typer.BadParameter(f"No report for community {show}")
        render_report(console, report)
    else:
        console.print(stats_panel(graph_index))
        render_communities(console, graph_index, level)
    if png is not None:
        from .viz import render_graph_png

        console.print(f"Graph image written to {render_graph_png(graph_index, png, level=level)}")


@app.command()
def bench(
    embedder: EmbedderOption = "auto",
    out: Annotated[Path, typer.Option("--out", "-o", help="Folder for the JSON and Markdown results.")] = Path(
        "benchmarks/results"
    ),
    heldout: Annotated[bool, typer.Option(help="Also score extraction on the held-out gold set.")] = True,
    chart: Annotated[Path | None, typer.Option("--chart", help="Also render the benchmark chart (PNG).")] = None,
) -> None:
    """Run the retrieval and extraction benchmarks on the bundled synthetic data and save the results."""
    from .evaluation.bench import run_benchmark, write_results

    console = Console()
    model = _embedder(embedder)
    command = "graphrag " + " ".join(shlex.quote(a) for a in sys.argv[1:]) if sys.argv[1:2] == ["bench"] else ""
    with console.status(f"Benchmarking with {model.name}..."):
        data = run_benchmark(model, heldout=heldout, command=command or f"graphrag bench --embedder {embedder}")
    json_path, md_path = write_results(data, out)
    render_retrieval(console, data["retrieval"], "test")
    render_retrieval(console, data["retrieval"], "all")
    render_extraction(console, "Extraction, development gold set (optimistic)", data["extraction"]["development"])
    if "heldout" in data["extraction"]:
        render_extraction(console, "Extraction, held-out gold set", data["extraction"]["heldout"])
    console.print(f"Results: {json_path} and {md_path}")
    if chart is not None:
        from .viz import render_benchmark_chart

        console.print(f"Chart: {render_benchmark_chart(data, chart)}")


@app.command()
def demo(
    embedder: EmbedderOption = "auto",
    svg: SvgOption = None,
    sections: Annotated[str, typer.Option(help="Comma list of: local, path, global.")] = "local,path,global",
) -> None:
    """Build the bundled fictional corpus offline and walk through local, path and global search."""
    console = _console(svg)
    chosen = {s.strip() for s in sections.split(",") if s.strip()}
    with console.status("Building the knowledge graph from 54 fictional documents..."):
        index = build_index(corpus_dir(), BuildConfig(extractor="spacy"))
    console.print(stats_panel(index, title="Offline build of the sample corpus"))
    model = _embedder(embedder)
    if "local" in chosen:
        console.rule("[bold]Local search: a 3-hop question")
        result = LocalSearcher(index.kg, model).search(DEMO_QUESTION, k=3, retriever="graph")
        render_local(console, index.kg, result)
    if "path" in chosen:
        console.rule("[bold]Path search: how are two entities connected?")
        render_path(console, index.kg, path_search(index.kg, DEMO_PATH, max_paths=2))
    if "global" in chosen:
        console.rule("[bold]Global search over community reports")
        render_global(console, global_search(index.reports, DEMO_GLOBAL, embedder=model, max_points=4))
    _save_svg(console, svg, "graphrag demo")


@app.command()
def serve(
    index: IndexOption = DEFAULT_INDEX,
    host: Annotated[str, typer.Option(help="Interface to bind.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Port to listen on.")] = 8703,
    embedder: EmbedderOption = "auto",
) -> None:
    """Serve the query API (FastAPI) for an index."""
    import uvicorn

    from .api import create_app

    uvicorn.run(create_app(_load(index), _embedder(embedder)), host=host, port=port)


@app.command()
def version() -> None:
    """Print the package version."""
    print(__version__)


def main() -> None:
    """Console entry point: load ``.env`` (if present) once, then run the CLI."""
    from dotenv import load_dotenv

    load_dotenv(override=False)
    app()
