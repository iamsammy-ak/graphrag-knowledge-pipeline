"""Path search: explain how entity A connects to entity B.

Edges cost ``1 / weight`` on the undirected entity graph, so a typed relation (weight 1, more with repeated
evidence) is cheaper than a co-occurrence link (weight 0.2, cost 5). ``networkx.shortest_simple_paths`` (Yen's
algorithm) yields loop-free paths in order of total cost; each hop is then explained with its strongest relation,
its direction and the evidence sentence it came from.
"""

from __future__ import annotations

from itertools import islice, pairwise

import networkx as nx

from ..graph import KnowledgeGraph
from ..schemas import EntityPath, LinkedEntity, PathHop, PathSearchResult
from .linking import link_entities


def find_paths(
    kg: KnowledgeGraph,
    source: str,
    target: str,
    *,
    max_paths: int = 3,
    max_hops: int = 5,
    include_cooccurrence: bool = True,
) -> list[EntityPath]:
    """Up to ``max_paths`` cheapest loop-free paths with at most ``max_hops`` hops."""
    if source not in kg.graph or target not in kg.graph or source == target:
        return []
    weighted = kg.undirected(include_cooccurrence)
    costed = nx.Graph()
    costed.add_nodes_from(weighted.nodes)
    costed.add_edges_from((u, v, {"cost": 1.0 / max(d["weight"], 1e-6)}) for u, v, d in weighted.edges(data=True))
    try:
        generator = nx.shortest_simple_paths(costed, source, target, weight="cost")
        candidates = list(islice(generator, max_paths * 10))
    except nx.NetworkXNoPath:
        return []
    paths: list[EntityPath] = []
    for nodes in candidates:
        if len(nodes) - 1 > max_hops:
            continue
        hops: list[PathHop] = []
        for u, v in pairwise(nodes):
            relation = kg.relations_between(u, v)[0]
            hops.append(
                PathHop(
                    source=u,
                    source_name=kg.name(u),
                    target=v,
                    target_name=kg.name(v),
                    relation=relation.type,
                    forward=relation.source == u,
                    chunk_id=relation.chunk_id,
                    evidence=relation.evidence,
                    weight=relation.weight,
                )
            )
        cost = sum(costed[u][v]["cost"] for u, v in pairwise(nodes))
        paths.append(EntityPath(hops=hops, cost=round(cost, 4)))
        if len(paths) >= max_paths:
            break
    return paths


def path_search(
    kg: KnowledgeGraph,
    query: str,
    *,
    source: str | None = None,
    target: str | None = None,
    max_paths: int = 3,
    max_hops: int = 5,
) -> PathSearchResult:
    """Link the two entities (explicit names win over those found in the query) and explain the paths."""
    linked = link_entities(kg, query)
    ends: list[LinkedEntity | None] = []
    for explicit, fallback_index in ((source, 0), (target, 1)):
        if explicit:
            found = link_entities(kg, explicit)
            ends.append(found[0] if found else None)
        else:
            ends.append(linked[fallback_index] if len(linked) > fallback_index else None)
    start, end = ends
    if start is None or end is None:
        return PathSearchResult(
            query=query, source=start, target=end, message="Path search needs two entities that exist in the graph."
        )
    paths = find_paths(kg, start.entity_id, end.entity_id, max_paths=max_paths, max_hops=max_hops)
    message = "" if paths else f"No path of at most {max_hops} hops connects {start.name} and {end.name}."
    return PathSearchResult(query=query, source=start, target=end, paths=paths, message=message)
