"""Community detection with Louvain at several resolutions, linked into a coarse-to-fine hierarchy.

Louvain maximises modularity; its ``resolution`` parameter trades community size for cohesion (lower = fewer,
larger communities). Running it at several resolutions gives levels that global search can choose between. Each
community at a finer level points to the coarser community it overlaps most (Louvain partitions at different
resolutions are not guaranteed to nest, so the parent is a best match, not a strict containment).
"""

from __future__ import annotations

from collections.abc import Sequence

import networkx as nx

from .graph import KnowledgeGraph
from .schemas import Community

DEFAULT_RESOLUTIONS: tuple[float, ...] = (0.5, 1.0, 2.0)


def detect_communities(
    kg: KnowledgeGraph,
    resolutions: Sequence[float] = DEFAULT_RESOLUTIONS,
    *,
    seed: int = 42,
    include_cooccurrence: bool = True,
) -> list[Community]:
    """Partition the entity graph at each resolution (sorted ascending: level 0 is the coarsest)."""
    graph = kg.undirected(include_cooccurrence)
    communities: list[Community] = []
    previous: list[Community] = []
    for level, resolution in enumerate(sorted(resolutions)):
        if graph.number_of_edges() == 0:
            parts = [{node} for node in graph.nodes]
        else:
            parts = nx.community.louvain_communities(graph, weight="weight", resolution=resolution, seed=seed)
        ordered = sorted((sorted(p) for p in parts), key=lambda members: (-len(members), members[0]))
        current: list[Community] = []
        for index, members in enumerate(ordered):
            parent = None
            if previous:
                overlap = max(previous, key=lambda c: (len(set(c.members) & set(members)), -len(c.members)))
                parent = overlap.id
            current.append(
                Community(id=f"L{level}-C{index}", level=level, resolution=resolution, members=members, parent=parent)
            )
        communities.extend(current)
        previous = current
    return communities


def levels(communities: Sequence[Community]) -> list[int]:
    return sorted({c.level for c in communities})


def at_level(communities: Sequence[Community], level: int) -> list[Community]:
    return [c for c in communities if c.level == level]


def modularity(kg: KnowledgeGraph, communities: Sequence[Community], level: int) -> float:
    """Modularity of one level's partition on the weighted entity graph."""
    graph = kg.undirected()
    parts = [set(c.members) for c in at_level(communities, level)]
    if graph.number_of_edges() == 0 or not parts:
        return 0.0
    return float(nx.community.modularity(graph, parts, weight="weight"))
