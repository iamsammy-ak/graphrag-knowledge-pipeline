"""Local search: from the entities a question names, walk the graph to the evidence chunks.

1. **Link** the query to seed entities.
2. **Score every relation for this question.** Each edge carries the sentence it was extracted from. Entity names
   are masked in both the question and the sentence ("In which city is the company that makes X headquartered?" vs
   "X is a robotics company headquartered in X"), so the score reflects the *relation wording* rather than shared
   names; the graph structure already takes care of which entities are involved.
3. **Expand** from the seeds, by default with a **relation-aware beam search**: extend the best partial paths one
   hop at a time, multiplying edge relevances, keeping the best path per end entity and the top ``beam_width``.
   Every chunk is credited with the best path that passes through its evidence. Two structural alternatives are
   kept for comparison: personalised PageRank and a weighted k-hop expansion, both of which ignore the question
   beyond its seeds.
4. **Fuse** (hybrid mode) with dense vector retrieval by weighted reciprocal rank fusion.
5. **Explain**: the paths that reached the retrieved chunks are returned as a traversal, so every edge shown is
   backed by a cited chunk.
"""

from __future__ import annotations

import re
from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Literal

import networkx as nx
import numpy as np

from ..embeddings import Embedder
from ..graph import KnowledgeGraph
from ..ingest import clean_for_display
from ..llm import LLM, clean_text
from ..schemas import COOCCURRENCE, EvidenceHit, LinkedEntity, LocalSearchResult, TraversalEdge
from .hybrid import reciprocal_rank_fusion
from .lexical import lexical_scores
from .linking import link_entities
from .vector import VectorIndex

Retriever = Literal["graph", "vector", "hybrid"]
MASK = "X"


@dataclass(frozen=True)
class LocalSearchConfig:
    """Retrieval settings. Defaults were chosen on the 10 dev questions of the benchmark, never on the test set."""

    expansion: Literal["beam", "ppr", "khop"] = "beam"
    mask_entities: bool = True  # beam: judge relation wording, not shared names
    beam_width: int = 10
    max_hops: int = 3
    relevance_floor: float = 0.05  # keeps weakly matching edges reachable
    cooccurrence_factor: float = 0.2  # co-occurrence edges count for a fifth of a typed relation
    alpha: float = 0.5  # ppr: probability of following an edge rather than restarting at a seed
    hops: int = 2  # khop: expansion depth
    hop_decay: float = 0.5  # khop: score multiplier per hop
    size_damping: float = 0.5  # ppr/khop: chunk score / (entities in chunk) ** size_damping
    fusion_weights: tuple[float, float] = (3.0, 1.0)  # hybrid: (graph, vector)
    fusion_k: int = 10  # hybrid: reciprocal rank fusion constant
    depth: int = 50  # candidates each retriever contributes


@dataclass(frozen=True)
class _Edge:
    source: str
    target: str
    type: str
    chunk_id: str
    evidence: str


class EdgeRelevance:
    """Question-to-relation relevance in [0, 1] for every edge, from its evidence sentence."""

    def __init__(self, kg: KnowledgeGraph, embedder: Embedder | None = None, *, mask_entities: bool = True) -> None:
        self.embedder = embedder
        self.mask_entities = mask_entities
        surfaces = sorted(
            {s for _, d in kg.graph.nodes(data=True) for s in (d["name"], *d.get("aliases", [])) if len(s) >= 3},
            key=len,
            reverse=True,
        )
        alternatives = "|".join(re.escape(s) for s in surfaces)
        self._names = re.compile(rf"(?<![\w-])(?:{alternatives})(?![\w-])") if surfaces else None
        self.texts = sorted({str(d["evidence"]) for _, _, d in kg.graph.edges(data=True)})
        self._prepared = [self.prepare(t) for t in self.texts]
        self._matrix = embedder.embed_documents(self._prepared) if embedder is not None and self.texts else None

    def prepare(self, text: str, extra_names: Sequence[str] = ()) -> str:
        """Replace entity names with a placeholder (when masking is on)."""
        if not self.mask_entities:
            return text
        for name in sorted(extra_names, key=len, reverse=True):
            text = re.sub(rf"(?<![\w-]){re.escape(name)}(?![\w-])", MASK, text, flags=re.I)
        return self._names.sub(MASK, text) if self._names is not None else text

    def scores(self, query: str, seeds: Sequence[LinkedEntity] = ()) -> dict[str, float]:
        """Min-max normalised similarity between the (masked) query and every evidence sentence."""
        if not self.texts:
            return {}
        prepared = self.prepare(query, [s.matched for s in seeds])
        if self._matrix is not None and self.embedder is not None:
            raw = self._matrix @ self.embedder.embed_query(prepared)
        else:
            raw = np.asarray(lexical_scores(prepared, self._prepared), dtype=np.float64)
        low, high = float(raw.min()), float(raw.max())
        span = high - low
        values = (raw - low) / span if span > 1e-12 else np.zeros_like(raw)
        return {text: float(v) for text, v in zip(self.texts, values, strict=True)}


def entity_relevance(kg: KnowledgeGraph, seeds: Mapping[str, float], config: LocalSearchConfig) -> dict[str, float]:
    """Structural relevance of every reachable entity (personalised PageRank or weighted k-hop expansion)."""
    seeds = {s: w for s, w in seeds.items() if s in kg.graph and w > 0}
    if not seeds:
        return {}
    graph = kg.undirected()
    if config.expansion == "ppr":
        if graph.number_of_edges() == 0:
            total = sum(seeds.values())
            return {s: w / total for s, w in seeds.items()}
        ranks = nx.pagerank(graph, alpha=config.alpha, personalization=dict(seeds), weight="weight")
        return {node: float(score) for node, score in ranks.items() if score > 1e-9}
    scores = dict(seeds)
    frontier = deque((s, 0) for s in seeds)
    while frontier:
        node, depth = frontier.popleft()
        if depth >= config.hops:
            continue
        for neighbour, data in graph[node].items():
            candidate = scores[node] * config.hop_decay * min(1.0, float(data["weight"]))
            if candidate > scores.get(neighbour, 0.0):
                scores[neighbour] = candidate
                frontier.append((neighbour, depth + 1))
    return scores


class LocalSearcher:
    """Graph, vector and hybrid evidence retrieval over one knowledge graph.

    Build it once and reuse it: it embeds the chunks (when an embedder is given) and every relation's evidence.
    Without an embedder, graph search scores relations lexically and the vector/hybrid retrievers are unavailable.
    """

    def __init__(
        self, kg: KnowledgeGraph, embedder: Embedder | None = None, config: LocalSearchConfig | None = None
    ) -> None:
        self.kg = kg
        self.config = config or LocalSearchConfig()
        self.vector = VectorIndex(list(kg.chunks.values()), embedder) if embedder is not None else None
        self.relevance = EdgeRelevance(kg, embedder, mask_entities=self.config.mask_entities)
        self._edges: list[_Edge] = []
        self._incident: dict[str, list[tuple[str, int]]] = defaultdict(list)
        for u, v, data in kg.graph.edges(data=True):
            self._incident[u].append((v, len(self._edges)))
            self._incident[v].append((u, len(self._edges)))
            self._edges.append(_Edge(u, v, str(data["type"]), str(data["chunk_id"]), str(data["evidence"])))

    # ------------------------------------------------------------------ graph retrieval
    def _beam(
        self, query: str, seeds: Sequence[LinkedEntity]
    ) -> tuple[dict[str, float], dict[str, tuple[tuple[str, ...], tuple[int, ...]]]]:
        """Best path score per chunk, and the path (nodes, edge indices) that achieved it."""
        cfg = self.config
        relevance = self.relevance.scores(query, seeds)
        Path = tuple[tuple[str, ...], tuple[int, ...], float]
        frontier: list[Path] = [((s.entity_id,), (), s.score) for s in seeds if s.entity_id in self.kg.graph]
        best: dict[str, float] = {}
        best_path: dict[str, tuple[tuple[str, ...], tuple[int, ...]]] = {}
        for _ in range(cfg.max_hops):
            expanded: list[Path] = []
            for nodes, edge_ids, score in frontier:
                for other, index in self._incident.get(nodes[-1], []):
                    if other in nodes:
                        continue
                    edge = self._edges[index]
                    strength = cfg.relevance_floor + relevance.get(edge.evidence, 0.0)
                    if edge.type == COOCCURRENCE:
                        strength *= cfg.cooccurrence_factor
                    path: Path = (nodes + (other,), edge_ids + (index,), score * strength)
                    expanded.append(path)
                    if path[2] > best.get(edge.chunk_id, 0.0):
                        best[edge.chunk_id] = path[2]
                        best_path[edge.chunk_id] = (path[0], path[1])
            expanded.sort(key=lambda p: (-p[2], p[0]))
            per_end: dict[str, Path] = {}
            for candidate in expanded:
                per_end.setdefault(candidate[0][-1], candidate)
            frontier = list(per_end.values())[: cfg.beam_width]
        return best, best_path

    def _structural(self, seeds: Sequence[LinkedEntity]) -> dict[str, float]:
        relevance = entity_relevance(self.kg, {s.entity_id: s.score for s in seeds}, self.config)
        scores: dict[str, float] = {}
        for chunk_id in {c for entity in relevance for c in self.kg.entity_chunks(entity)}:
            entities = self.kg.chunk_entities.get(chunk_id, [])
            if entities:
                total = sum(relevance.get(e, 0.0) for e in entities)
                scores[chunk_id] = total / (len(entities) ** self.config.size_damping)
        return scores

    def graph_rank(
        self, query: str, seeds: Sequence[LinkedEntity]
    ) -> tuple[list[tuple[str, float]], list[TraversalEdge]]:
        """Chunks ranked by graph relevance, plus the paths behind them (beam expansion only)."""
        paths: dict[str, tuple[tuple[str, ...], tuple[int, ...]]] = {}
        if self.config.expansion == "beam":
            scores, paths = self._beam(query, seeds)
        else:
            scores = self._structural(seeds)
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[: self.config.depth]
        return ranked, self._path_edges([paths[c] for c, _ in ranked if c in paths])

    def _path_edges(self, paths: Sequence[tuple[tuple[str, ...], tuple[int, ...]]]) -> list[TraversalEdge]:
        edges: dict[tuple[str, str, str], TraversalEdge] = {}
        for nodes, edge_ids in paths:
            for depth, ((u, v), index) in enumerate(zip(pairwise(nodes), edge_ids, strict=True), start=1):
                edge = self._edges[index]
                key = (u, v, edge.chunk_id)
                if key not in edges:
                    edges[key] = TraversalEdge(
                        source=u,
                        source_name=self.kg.name(u),
                        target=v,
                        target_name=self.kg.name(v),
                        relation=edge.type,
                        forward=edge.source == u,
                        chunk_id=edge.chunk_id,
                        depth=depth,
                    )
        return list(edges.values())

    # ------------------------------------------------------------------ public API
    def rank(self, query: str, retriever: Retriever = "hybrid") -> LocalSearchResult:
        """All candidates (up to ``config.depth`` per retriever), best first, without answer synthesis."""
        if retriever in ("vector", "hybrid") and self.vector is None:
            raise ValueError(f"the '{retriever}' retriever needs an embedding model")
        seeds = link_entities(self.kg, query)
        graph_ranked: list[tuple[str, float]] = []
        paths: list[TraversalEdge] = []
        if retriever in ("graph", "hybrid"):
            graph_ranked, paths = self.graph_rank(query, seeds)
        vector_ranked: list[tuple[str, float]] = []
        if retriever in ("vector", "hybrid") and self.vector is not None:
            vector_ranked = self.vector.search(query, self.config.depth)
        if retriever == "graph":
            fused = graph_ranked
        elif retriever == "vector":
            fused = vector_ranked
        else:
            fused = reciprocal_rank_fusion(
                [[c for c, _ in graph_ranked], [c for c, _ in vector_ranked]],
                k=self.config.fusion_k,
                weights=self.config.fusion_weights,
            )
        hits = [
            EvidenceHit(
                chunk_id=chunk_id,
                doc_id=self.kg.chunks[chunk_id].doc_id,
                score=round(score, 6),
                text=clean_for_display(self.kg.chunks[chunk_id].text, drop_headings=True),
                entities=[self.kg.name(e) for e in self.kg.chunk_entities.get(chunk_id, [])],
            )
            for chunk_id, score in fused
        ]
        return LocalSearchResult(query=query, retriever=retriever, seeds=seeds, hits=hits, traversal=paths)

    def search(
        self, query: str, *, k: int = 5, retriever: Retriever = "hybrid", llm: LLM | None = None
    ) -> LocalSearchResult:
        """Top ``k`` evidence chunks, the traversal that explains them and (with an LLM) a cited answer."""
        result = self.rank(query, retriever)
        hits = result.hits[:k]
        kept = {h.chunk_id for h in hits}
        traversal = [e for e in result.traversal if e.chunk_id in kept and self._reachable(e, result.traversal, kept)]
        if not traversal:
            traversal = traversal_tree(self.kg, result.seeds, kept)
        answer = answer_with_llm(llm, query, hits, traversal) if llm is not None and hits else None
        return result.model_copy(update={"hits": hits, "traversal": traversal, "answer": answer})

    @staticmethod
    def _reachable(edge: TraversalEdge, edges: Sequence[TraversalEdge], kept: set[str]) -> bool:
        """An edge is shown only if the edges leading to it are backed by retrieved chunks too."""
        if edge.depth == 1:
            return True
        parents = [e for e in edges if e.target == edge.source and e.depth == edge.depth - 1 and e.chunk_id in kept]
        return any(LocalSearcher._reachable(p, edges, kept) for p in parents)


def traversal_tree(
    kg: KnowledgeGraph, seeds: Sequence[LinkedEntity], chunk_ids: set[str], max_depth: int = 3
) -> list[TraversalEdge]:
    """Breadth-first tree from the seeds over relations whose evidence chunk was retrieved (any retriever)."""

    def neighbours(node: str, typed: bool) -> list[tuple[str, str, bool, str, float]]:
        out = []
        for u, v, data in [*kg.graph.out_edges(node, data=True), *kg.graph.in_edges(node, data=True)]:
            if data["chunk_id"] not in chunk_ids or (typed and data["type"] == COOCCURRENCE):
                continue
            other, forward = (v, True) if u == node else (u, False)
            out.append((other, data["type"], forward, data["chunk_id"], float(data["weight"])))
        out.sort(key=lambda item: (-item[4], item[1], item[0]))
        return out

    edges: list[TraversalEdge] = []
    for typed in (True, False):  # fall back to co-occurrence links only when no typed relation was retrieved
        visited = {s.entity_id for s in seeds}
        queue = deque((s.entity_id, 0) for s in seeds)
        while queue:
            node, depth = queue.popleft()
            if depth >= max_depth:
                continue
            for other, relation, forward, chunk_id, _ in neighbours(node, typed):
                if other in visited:
                    continue
                visited.add(other)
                edges.append(
                    TraversalEdge(
                        source=node,
                        source_name=kg.name(node),
                        target=other,
                        target_name=kg.name(other),
                        relation=relation,
                        forward=forward,
                        chunk_id=chunk_id,
                        depth=depth + 1,
                    )
                )
                queue.append((other, depth + 1))
        if edges:
            break
    return edges


def answer_with_llm(llm: LLM, query: str, hits: Sequence[EvidenceHit], traversal: Sequence[TraversalEdge]) -> str:
    """Grounded answer: the model sees only the retrieved chunks and the traversal, and must cite chunk ids."""
    evidence = "\n".join(f"[{h.chunk_id}] {h.text}" for h in hits)
    paths = "\n".join(
        f"- {e.source_name} {e.relation if e.forward else e.relation + ' (reverse)'} {e.target_name} [{e.chunk_id}]"
        for e in traversal
    )
    messages = [
        {
            "role": "system",
            "content": "Answer the question using only the evidence. Follow the relation paths across chunks when "
            "the answer needs several hops. Cite chunk ids in square brackets. If the evidence is insufficient, "
            "say so.",
        },
        {"role": "user", "content": f"Question: {query}\n\nRelation paths:\n{paths}\n\nEvidence:\n{evidence}"},
    ]
    return clean_text(llm.complete(messages, temperature=0.1, max_tokens=600))
