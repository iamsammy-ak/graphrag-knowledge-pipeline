"""Rich terminal rendering for the command line and the demo."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from rich.console import Console, Group
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

from .graph import KnowledgeGraph
from .pipeline import GraphIndex
from .reports import cooccurrence_share
from .schemas import (
    CommunityReport,
    GlobalSearchResult,
    LocalSearchResult,
    PathSearchResult,
    TraversalEdge,
)

ACCENT = "bold #0f766e"
TYPE_STYLE = {
    "PERSON": "#b45309",
    "ORGANIZATION": "#1d4ed8",
    "LOCATION": "#15803d",
    "PRODUCT": "#7c3aed",
    "EVENT": "#be123c",
    "CONCEPT": "#475569",
}


def save_svg(console: Console, path: str | Path, title: str) -> Path:
    """Save a recorded console as SVG, without the exporter's advertising comment."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    svg = console.export_svg(title=title)
    path.write_text(re.sub(r"\s*<!--.*?-->", "", svg, count=1), encoding="utf-8")
    return path


def _entity(kg: KnowledgeGraph, entity_id: str) -> str:
    data = kg.graph.nodes[entity_id]
    return f"[bold {TYPE_STYLE.get(data['type'], 'white')}]{escape(data['name'])}[/]"


def _triple(kg: KnowledgeGraph, edge: TraversalEdge) -> str:
    left, right = (edge.source, edge.target) if edge.forward else (edge.target, edge.source)
    return f"{_entity(kg, left)} [{ACCENT}]-[{edge.relation}]->[/] {_entity(kg, right)}  [dim]{edge.chunk_id}[/]"


def snippet(text: str, limit: int = 150) -> str:
    return text if len(text) <= limit else text[: limit - 1].rsplit(" ", 1)[0] + " ..."


def stats_panel(index: GraphIndex, title: str = "Knowledge graph") -> Panel:
    stats = index.manifest.get("stats") or index.kg.stats()
    types = ", ".join(f"{n} {t.lower()}" for t, n in stats["entity_types"].items())
    levels = index.manifest.get("levels", [])
    community_text = "Louvain communities: " + ", ".join(
        f"L{lv['level']} = {lv['communities']} (resolution {lv['resolution']}, modularity {lv['modularity']:.2f})"
        for lv in levels
    )
    lines = [
        f"[bold]{stats['documents']}[/] documents -> [bold]{stats['chunks']}[/] chunks -> "
        f"[bold]{stats['entities']}[/] entities ({types})",
        f"[bold]{stats['typed_relations']}[/] typed relations + [bold]{stats['cooccurrence_relations']}[/] "
        f"co-occurrence links, extracted by [bold]{index.manifest.get('extractor', '?')}[/]",
    ]
    if levels:
        lines.append(community_text)
    return Panel("\n".join(lines), title=title, border_style="#0f766e", expand=False)


def render_local(console: Console, kg: KnowledgeGraph, result: LocalSearchResult) -> None:
    seeds = ", ".join(f"{_entity(kg, s.entity_id)} [dim](matched '{escape(s.matched)}')[/]" for s in result.seeds)
    console.print(f"[bold]Question:[/] {escape(result.query)}")
    console.print(f"[bold]Linked entities:[/] {seeds or '[red]none[/]'}   [dim]retriever: {result.retriever}[/]")
    if result.traversal:
        children: dict[str, list[TraversalEdge]] = defaultdict(list)
        for edge in result.traversal:
            children[edge.source].append(edge)
        root = Tree("[bold]Reasoning paths[/] [dim](every edge cites the chunk it was read from)[/]")

        def grow(branch: Tree, node: str, depth: int, seen: set[str]) -> None:
            for edge in children.get(node, []):
                if edge.depth != depth + 1 or edge.target in seen:
                    continue
                child = branch.add(_triple(kg, edge))
                grow(child, edge.target, depth + 1, seen | {edge.target})

        for seed in result.seeds:
            if seed.entity_id in children:
                grow(root.add(_entity(kg, seed.entity_id)), seed.entity_id, 0, {seed.entity_id})
        console.print(root)
    table = Table(title="Cited evidence", title_justify="left", show_lines=False, expand=True)
    table.add_column("#", justify="right", width=2)
    table.add_column("chunk", style="dim", no_wrap=True)
    table.add_column("evidence")
    for rank, hit in enumerate(result.hits, start=1):
        table.add_row(str(rank), hit.chunk_id, escape(snippet(hit.text)))
    console.print(table)
    if result.answer:
        console.print(Panel(escape(result.answer), title="Answer", border_style="#0f766e"))
    else:
        console.print("[dim]Offline mode: evidence and reasoning paths only (answer synthesis needs an LLM key).[/]")


def render_path(console: Console, kg: KnowledgeGraph, result: PathSearchResult) -> None:
    console.print(f"[bold]Question:[/] {escape(result.query)}")
    if result.message and not result.paths:
        console.print(f"[yellow]{escape(result.message)}[/]")
        return
    assert result.source is not None and result.target is not None
    console.print(
        f"[bold]From[/] {_entity(kg, result.source.entity_id)} [bold]to[/] {_entity(kg, result.target.entity_id)}"
    )
    for number, path in enumerate(result.paths, start=1):
        chain = Text.from_markup(_entity(kg, path.hops[0].source))
        for hop in path.hops:
            arrow = f" -[{hop.relation}]-> " if hop.forward else f" <-[{hop.relation}]- "
            chain.append_text(Text(arrow, style=ACCENT))
            chain.append_text(Text.from_markup(_entity(kg, hop.target)))
        lines = [chain]
        shown: set[tuple[str, str]] = set()
        for hop in path.hops:
            if (hop.chunk_id, hop.evidence) not in shown:  # two hops can come from the same sentence
                shown.add((hop.chunk_id, hop.evidence))
                lines.append(Text.from_markup(f"  [dim]{hop.chunk_id}:[/] {escape(snippet(hop.evidence, 120))}"))
        console.print(
            Panel(Group(*lines), title=f"Path {number}  ({len(path.hops)} hops, cost {path.cost})", expand=True)
        )


def render_global(console: Console, result: GlobalSearchResult) -> None:
    console.print(f"[bold]Question:[/] {escape(result.query)}  [dim](community level {result.level})[/]")
    table = Table(title="Most relevant community reports", title_justify="left", expand=True)
    table.add_column("community", style="dim", no_wrap=True)
    table.add_column("title")
    table.add_column("score", justify="right")
    for report in result.reports:
        table.add_row(report.community_id, escape(report.title), f"{report.score:.3f}")
    console.print(table)
    if result.answer:
        console.print(Panel(escape(result.answer), title="Answer (map-reduce)", border_style="#0f766e"))
    for point in result.points:
        cites = ", ".join(point.chunk_ids) or point.community_id
        console.print(f" [{ACCENT}]*[/] {escape(point.text)} [dim]({point.community_id}; {cites})[/]")
    if not result.answer:
        console.print("[dim]Offline mode: key points extracted from community reports, not generated.[/]")


def render_communities(console: Console, index: GraphIndex, level: int, limit: int = 20) -> None:
    reports = {r.community_id: r for r in index.reports}
    table = Table(title=f"Communities at level {level}", title_justify="left", expand=True)
    table.add_column("id", style="dim", no_wrap=True)
    table.add_column("size", justify="right")
    table.add_column("title / top entities")
    table.add_column("parent", style="dim")
    table.add_column("co-occ.", justify="right")
    communities = [c for c in index.communities if c.level == level]
    for community in communities[:limit]:
        report = reports.get(community.id)
        title = report.title if report else ", ".join(index.kg.name(m) for m in community.members[:3])
        table.add_row(
            community.id,
            str(len(community.members)),
            escape(title),
            community.parent or "",
            f"{cooccurrence_share(index.kg, community):.0%}",
        )
    console.print(table)
    if len(communities) > limit:
        console.print(f"[dim]{len(communities) - limit} more not shown.[/]")


def render_report(console: Console, report: CommunityReport) -> None:
    lines = [f"[bold]{escape(report.summary)}[/]", ""]
    for finding in report.findings:
        lines.append(f" [{ACCENT}]*[/] {escape(finding.text)} [dim]({', '.join(finding.chunk_ids)})[/]")
    console.print(
        Panel("\n".join(lines), title=f"{report.community_id}: {escape(report.title)}", subtitle=report.method)
    )


def render_retrieval(console: Console, result: dict[str, Any], group: str = "test") -> None:
    summary = result["summary"][group]
    ks: Sequence[int] = result["ks"]
    table = Table(
        title=f"Multi-hop evidence retrieval ({group} questions: {next(iter(summary.values()))['questions']}, "
        f"embedder {result['embedder']})",
        title_justify="left",
    )
    table.add_column("method")
    for k in ks:
        table.add_column(f"recall@{k}", justify="right")
    for k in ks:
        table.add_column(f"full@{k}", justify="right")
    for method, scores in summary.items():
        recalls = [f"{scores[f'recall@{k}']:.2f}" for k in ks]
        fulls = [f"{scores[f'full@{k}']:.2f}" for k in ks]
        table.add_row(method, *recalls, *fulls)
    console.print(table)


def render_extraction(console: Console, title: str, result: dict[str, Any]) -> None:
    table = Table(title=title, title_justify="left")
    table.add_column("measure")
    table.add_column("precision", justify="right")
    table.add_column("recall", justify="right")
    table.add_column("F1", justify="right")
    for key, label in (
        ("entities", "entities (found)"),
        ("entities_typed", "entities (right type)"),
        ("relations_typed", "relations (typed, directed)"),
        ("relations_untyped", "relations (any type, either direction)"),
    ):
        row = result[key]
        table.add_row(label, f"{row['precision']:.2f}", f"{row['recall']:.2f}", f"{row['f1']:.2f}")
    console.print(table)
    console.print(
        f"[dim]co-occurrence edges: {result['cooccurrence_edges']} (precision {result['cooccurrence_precision']:.2f}); "
        f"gold pairs connected by any edge: {result['connectivity_recall']:.2f}[/]"
    )
