"""Static figures: the knowledge graph coloured by community, and the benchmark chart (matplotlib, optional)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import networkx as nx

from .pipeline import GraphIndex
from .reports import pagerank

# Categorical slots in fixed order (validated colour-blind-safe reference palette), then a neutral for "other".
CATEGORICAL = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
NEUTRAL = "#b5b3ad"
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#e4e2dc"


def _layout(index: GraphIndex, level: int, seed: int, nodes: set[str]) -> dict[str, Any]:
    """Two-level layout: community centres spaced on a circle, a spring layout inside each community."""
    import math

    graph = index.kg.undirected().subgraph(nodes)
    communities = sorted(
        (c for c in index.communities if c.level == level and set(c.members) & nodes), key=lambda c: -len(c.members)
    )
    centres = nx.circular_layout([c.id for c in communities], scale=1.7) if len(communities) > 1 else {}
    positions: dict[str, Any] = {}
    for community in communities:
        members = [m for m in community.members if m in nodes]
        centre = centres.get(community.id, (0.0, 0.0))
        local = nx.spring_layout(graph.subgraph(members), weight="weight", seed=seed, k=1.6 / math.sqrt(len(members)))
        radius = 0.2 + 0.13 * math.sqrt(len(members))
        for member in members:
            positions[member] = (centre[0] + radius * local[member][0], centre[1] + radius * local[member][1])
    return positions


def render_graph_png(
    index: GraphIndex,
    path: str | Path,
    *,
    level: int = 0,
    max_labels: int = 26,
    title: str = "Knowledge graph built from the sample corpus",
    seed: int = 7,
) -> Path:
    """Draw entities sized by PageRank and coloured by community; label the most central ones."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    kg = index.kg
    graph = kg.undirected()
    ranks = pagerank(kg)
    communities = sorted((c for c in index.communities if c.level == level), key=lambda c: -len(c.members))
    reports = {r.community_id: r for r in index.reports}
    colour: dict[str, str] = {}
    legend: list[tuple[str, str]] = []
    for slot, community in enumerate(communities):
        hue = CATEGORICAL[slot] if slot < len(CATEGORICAL) - 1 and len(community.members) > 1 else NEUTRAL
        for member in community.members:
            colour[member] = hue
        if hue != NEUTRAL:
            report = reports.get(community.id)
            name = report.top_entities[0] if report and report.top_entities else kg.name(community.members[0])
            legend.append((hue, f"{name} ({len(community.members)})"))
    if any(colour.get(n) == NEUTRAL for n in graph.nodes if graph.degree(n) > 0):
        legend.append((NEUTRAL, "other small communities"))
    drawn = {n for n in graph.nodes if graph.degree(n) > 0}
    graph = graph.subgraph(drawn)
    pos = _layout(index, level, seed, drawn)

    fig, ax = plt.subplots(figsize=(13, 8.6), dpi=100)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    membership = {m: c.id for c in communities for m in c.members}
    typed = {(u, v) for u, v in graph.edges if any(r.type != "RELATED_TO" for r in kg.relations_between(u, v))}
    loose = [e for e in graph.edges if e not in typed]
    inside = [e for e in typed if membership.get(e[0]) == membership.get(e[1])]
    across = [e for e in typed if membership.get(e[0]) != membership.get(e[1])]
    nx.draw_networkx_edges(graph, pos, edgelist=loose, ax=ax, edge_color=GRID, width=0.8, style="dashed")
    nx.draw_networkx_edges(graph, pos, edgelist=across, ax=ax, edge_color="#dedbd3", width=0.9)
    nx.draw_networkx_edges(graph, pos, edgelist=inside, ax=ax, edge_color="#a8a59c", width=1.4)
    nodes = list(graph.nodes)
    top = max(ranks.values()) if ranks else 1.0
    sizes = [60 + 900 * (ranks.get(n, 0.0) / top) for n in nodes]
    nx.draw_networkx_nodes(
        graph, pos, nodelist=nodes, node_size=sizes, node_color=[colour.get(n, NEUTRAL) for n in nodes],
        edgecolors="white", linewidths=1.5, ax=ax,
    )
    labelled = sorted(nodes, key=lambda n: -ranks.get(n, 0.0))[:max_labels]
    for node in labelled:
        x, y = pos[node]
        ax.annotate(
            kg.name(node), (x, y), xytext=(0, 7 + 9 * ranks.get(node, 0.0) / top), textcoords="offset points",
            ha="center", va="bottom", fontsize=8, color=INK,
            bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.8},
        )
    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=h, markersize=9, label=t) for h, t in legend]
    handles += [
        Line2D([0], [0], color="#a8a59c", lw=1.4, label="typed relation (within community)"),
        Line2D([0], [0], color="#dedbd3", lw=0.9, label="typed relation (across communities)"),
        Line2D([0], [0], color=GRID, lw=1.0, ls="--", label="co-occurrence only"),
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=9,
              title=f"Louvain communities (level {level})", title_fontsize=9.5)
    ax.set_title(title, fontsize=13, color=INK, loc="left")
    note = "Node size: PageRank. Labels: the most central entities."
    isolated = kg.graph.number_of_nodes() - len(drawn)
    if isolated:
        note += f" Not drawn: {isolated} {'entity' if isolated == 1 else 'entities'} with no relation."
    ax.text(0, -0.02, note, transform=ax.transAxes,
            fontsize=8.5, color=INK_MUTED)
    ax.axis("off")
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return path


def render_benchmark_chart(result: dict[str, Any], path: str | Path, *, group: str = "test") -> Path:
    """Two panels: full support@k for vector / graph / hybrid, and full support@5 for every variant."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    retrieval = result["retrieval"]
    summary = retrieval["summary"][group]
    ks: Sequence[int] = retrieval["ks"]
    main = [m for m in ("vector", "graph", "hybrid") if m in summary]
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.6), dpi=110, gridspec_kw={"width_ratios": [1.25, 1]})
    fig.patch.set_facecolor("white")
    width = 0.8 / len(main)
    for i, method in enumerate(main):
        xs = [j + (i - (len(main) - 1) / 2) * width for j in range(len(ks))]
        values = [summary[method][f"full@{k}"] for k in ks]
        bars = left.bar(xs, values, width=width - 0.03, color=CATEGORICAL[i], label=method, zorder=3)
        for bar, value in zip(bars, values, strict=True):
            left.text(bar.get_x() + bar.get_width() / 2, value + 0.015, f"{value:.2f}", ha="center", va="bottom",
                      fontsize=8.5, color=INK_MUTED)
    left.set_xticks(range(len(ks)), [f"top {k}" for k in ks])
    left.set_ylim(0, 1.08)
    left.set_ylabel("share of questions with every hop retrieved", color=INK_MUTED, fontsize=9)
    left.set_title("Full support@k: vector vs graph vs hybrid", loc="left", fontsize=11, color=INK)
    left.legend(frameon=False, fontsize=9, loc="upper left")
    order = [m for m in ("vector", "graph-ppr", "graph-unmasked", "graph", "hybrid") if m in summary]
    labels = {
        "vector": "vector only",
        "graph-ppr": "graph: structural PageRank",
        "graph-unmasked": "graph: relation-aware, names kept",
        "graph": "graph: relation-aware, names masked",
        "hybrid": "hybrid (weighted RRF)",
    }
    values = [summary[m]["full@5"] for m in order]
    ys = list(range(len(order)))[::-1]
    right.barh(ys, values, color=CATEGORICAL[0], height=0.6, zorder=3)
    for y, value in zip(ys, values, strict=True):
        right.text(value + 0.015, y, f"{value:.2f}", va="center", fontsize=9, color=INK_MUTED)
    right.set_yticks(ys, [labels[m] for m in order], fontsize=9)
    right.set_xlim(0, 1.1)
    right.set_title("Where the gain comes from (full support@5)", loc="left", fontsize=11, color=INK)
    for ax in (left, right):
        ax.set_facecolor("white")
        ax.grid(axis="y" if ax is left else "x", color=GRID, zorder=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#c9c6be")
        ax.tick_params(colors=INK_MUTED, labelsize=9)
    n = summary["vector"]["questions"]
    fig.suptitle(
        f"Multi-hop evidence retrieval on {n} held-out test questions (synthetic corpus, {retrieval['embedder']})",
        x=0.01, ha="left", fontsize=11.5, color=INK,
    )
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return path
