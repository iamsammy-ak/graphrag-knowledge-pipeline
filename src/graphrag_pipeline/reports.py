"""Community reports: a readable summary of each community, used by global search.

* ``ExtractiveReporter`` (offline stand-in): ranks members by PageRank, lists the strongest relations inside the
  community and quotes the evidence sentences that support them plus sentences naming several members. Nothing is
  generated; every sentence in the report is copied from the corpus and cites its chunk.
* ``LLMReporter``: gives the same material to a model and asks for a titled summary with cited findings; falls back
  to the extractive report for a community if the model fails.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Protocol

import networkx as nx
from pydantic import BaseModel, Field

from .graph import KnowledgeGraph
from .ingest import clean_for_display, split_sentences
from .llm import LLM, LLMError, complete_structured
from .schemas import COOCCURRENCE, Community, CommunityReport, Finding

_TYPE_WORDS = {
    "PERSON": ("person", "people"),
    "ORGANIZATION": ("organisation", "organisations"),
    "LOCATION": ("place", "places"),
    "PRODUCT": ("product", "products"),
    "EVENT": ("event", "events"),
    "CONCEPT": ("other entity", "other entities"),
}


def pagerank(kg: KnowledgeGraph) -> dict[str, float]:
    """Global weighted PageRank on the undirected entity graph (empty graph -> empty dict)."""
    graph = kg.undirected()
    if graph.number_of_nodes() == 0:
        return {}
    return {k: float(v) for k, v in nx.pagerank(graph, weight="weight").items()}


def join_names(names: Sequence[str]) -> str:
    if len(names) <= 1:
        return "".join(names)
    return f"{', '.join(names[:-1])} and {names[-1]}"


class Reporter(Protocol):
    name: str

    def report(self, kg: KnowledgeGraph, community: Community, ranks: dict[str, float]) -> CommunityReport: ...


class ExtractiveReporter:
    """Deterministic report built only from graph structure and sentences copied from the corpus."""

    name = "extractive"

    def __init__(self, max_findings: int = 6, top_entities: int = 5) -> None:
        self.max_findings = max_findings
        self.top_entities = top_entities

    def material(
        self, kg: KnowledgeGraph, community: Community, ranks: dict[str, float]
    ) -> tuple[list[str], list[tuple[str, str, str, str, str]], list[Finding]]:
        """Ranked member ids, key relations (source, type, target, evidence, chunk) and evidence findings."""
        members = sorted(community.members, key=lambda m: (-ranks.get(m, 0.0), kg.name(m)))
        member_set = set(members)
        relations = [
            r
            for r in kg.relations(include_cooccurrence=False)
            if r.source in member_set and r.target in member_set
        ]
        relations.sort(key=lambda r: (-(ranks.get(r.source, 0) + ranks.get(r.target, 0)), -r.weight, r.id))
        key_relations: list[tuple[str, str, str, str, str]] = []
        seen: set[tuple[str, str, str]] = set()
        for r in relations:
            triple = (r.source, r.type, r.target)
            if triple not in seen:
                seen.add(triple)
                key_relations.append((kg.name(r.source), r.type, kg.name(r.target), r.evidence, r.chunk_id))
        findings: list[Finding] = []
        used: set[str] = set()

        def finding(sentence: str, chunk_id: str) -> Finding:
            """Keep the sentence verbatim; name its page when the sentence alone lacks context ("Its CEO is...")."""
            title = kg.chunks[chunk_id].title
            text = sentence if not title or title in sentence else f"{title}: {sentence}"
            return Finding(text=text, chunk_ids=[chunk_id])

        for _, _, _, evidence, chunk_id in key_relations:
            if evidence not in used and len(findings) < self.max_findings:
                used.add(evidence)
                findings.append(finding(evidence, chunk_id))
        if len(findings) < self.max_findings:  # sentences naming several members, most central first
            scored: list[tuple[float, str, str]] = []
            chunk_ids = sorted({c for m in members for c in kg.entity_chunks(m)})
            names = {m: kg.graph.nodes[m] for m in members}
            for chunk_id in chunk_ids:
                text = kg.chunks[chunk_id].text
                for start, end in split_sentences(text):
                    sentence = clean_for_display(text[start:end])
                    hits = [m for m, d in names.items() if any(s in sentence for s in (d["name"], *d["aliases"]))]
                    if len(hits) >= 2 and sentence not in used:
                        scored.append((len(hits) + sum(ranks.get(m, 0) for m in hits), sentence, chunk_id))
            for _, sentence, chunk_id in sorted(scored, key=lambda x: (-x[0], x[1])):
                if len(findings) >= self.max_findings:
                    break
                if sentence not in used:
                    used.add(sentence)
                    findings.append(finding(sentence, chunk_id))
        return members, key_relations, findings

    def report(self, kg: KnowledgeGraph, community: Community, ranks: dict[str, float]) -> CommunityReport:
        members, key_relations, findings = self.material(kg, community, ranks)
        top = [kg.name(m) for m in members[: self.top_entities]]
        counts = Counter(kg.graph.nodes[m]["type"] for m in members)
        mix = ", ".join(
            f"{n} {_TYPE_WORDS[t][0] if n == 1 else _TYPE_WORDS[t][1]}" for t, n in counts.most_common()
        )
        lead = members[0]
        lead_desc = kg.graph.nodes[lead].get("description") or kg.graph.nodes[lead]["type"].lower()
        summary = f"{len(members)} entities ({mix}), centred on {kg.name(lead)} ({lead_desc})."
        if key_relations:
            triples = "; ".join(f"{s} {t} {o}" for s, t, o, _, _ in key_relations[:4])
            summary += f" Key relations: {triples}."
        return CommunityReport(
            community_id=community.id,
            level=community.level,
            title=join_names(top[:3]),
            summary=summary,
            findings=findings,
            top_entities=top,
            importance=round(sum(ranks.get(m, 0.0) for m in members), 6),
            method=self.name,
        )


class _LLMFinding(BaseModel):
    text: str
    chunk_ids: list[str] = Field(default_factory=list)


class _LLMReportPayload(BaseModel):
    title: str
    summary: str
    findings: list[_LLMFinding] = Field(default_factory=list)


class LLMReporter:
    """Asks a model to write the report from the extractive material; keeps only citations it was given."""

    name = "llm"

    def __init__(self, llm: LLM, fallback: ExtractiveReporter | None = None) -> None:
        self.llm = llm
        self.fallback = fallback or ExtractiveReporter()

    def report(self, kg: KnowledgeGraph, community: Community, ranks: dict[str, float]) -> CommunityReport:
        members, key_relations, findings = self.fallback.material(kg, community, ranks)
        entity_lines = "\n".join(
            f"- {kg.name(m)} ({kg.graph.nodes[m]['type']}): {kg.graph.nodes[m].get('description', '')}"
            for m in members[:20]
        )
        relation_lines = "\n".join(f"- {s} {t} {o} [{c}]" for s, t, o, _, c in key_relations[:20])
        evidence_lines = "\n".join(f"- [{f.chunk_ids[0]}] {f.text}" for f in findings)
        messages = [
            {
                "role": "system",
                "content": "You write short analyst reports about one community of a knowledge graph. Use only "
                "the material given. Every finding must cite the chunk ids in square brackets that support it.",
            },
            {
                "role": "user",
                "content": f"Entities:\n{entity_lines}\n\nRelations:\n{relation_lines}\n\nEvidence:\n{evidence_lines}"
                "\n\nWrite a title (max 8 words), a 2-3 sentence summary and 3-5 findings.",
            },
        ]
        try:
            payload = complete_structured(self.llm, messages, _LLMReportPayload)
        except LLMError:
            return self.fallback.report(kg, community, ranks)
        allowed = {f.chunk_ids[0] for f in findings} | {c for *_, c in key_relations}
        base = self.fallback.report(kg, community, ranks)
        return base.model_copy(
            update={
                "title": payload.title.strip() or base.title,
                "summary": payload.summary.strip() or base.summary,
                "findings": [
                    Finding(text=f.text.strip(), chunk_ids=[c for c in f.chunk_ids if c in allowed])
                    for f in payload.findings
                    if f.text.strip()
                ]
                or base.findings,
                "method": self.name,
            }
        )


def build_reports(
    kg: KnowledgeGraph,
    communities: Sequence[Community],
    reporter: Reporter | None = None,
    *,
    min_size: int = 2,
) -> list[CommunityReport]:
    """One report per community with at least ``min_size`` members, ordered by level then importance."""
    reporter = reporter or ExtractiveReporter()
    ranks = pagerank(kg)
    reports = [reporter.report(kg, c, ranks) for c in communities if len(c.members) >= min_size]
    return sorted(reports, key=lambda r: (r.level, -r.importance, r.community_id))


def cooccurrence_share(kg: KnowledgeGraph, community: Community) -> float:
    """Share of the community's internal edges that are only co-occurrence (a rough noise indicator)."""
    members = set(community.members)
    edges = [d for u, v, d in kg.graph.edges(data=True) if u in members and v in members]
    if not edges:
        return 0.0
    return sum(1 for d in edges if d["type"] == COOCCURRENCE) / len(edges)
