"""Global search: answer corpus-wide questions from community reports (map-reduce).

* **Offline** (extractive stand-in): rank the reports of one level by similarity to the query (embeddings if
  available, otherwise a lexical score), then pick the findings of the best reports that are most similar to the
  query. The "answer" is those cited findings; nothing is generated.
* **With an LLM**: *map* asks the model to extract scored key points from each relevant report, *reduce* writes
  one answer from the highest-scoring points, citing communities.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, Field

from ..embeddings import Embedder
from ..llm import LLM, LLMError, clean_text, complete_structured
from ..schemas import CommunityReport, GlobalPoint, GlobalSearchResult, RankedReport
from .lexical import lexical_scores


def _similarities(query: str, texts: Sequence[str], embedder: Embedder | None) -> list[float]:
    if not texts:
        return []
    if embedder is None:
        return lexical_scores(query, texts)
    matrix = embedder.embed_documents(list(texts))
    return [float(x) for x in matrix @ embedder.embed_query(query)]


class _MapPoint(BaseModel):
    text: str
    score: int = Field(ge=0, le=100)


class _MapResponse(BaseModel):
    points: list[_MapPoint] = Field(default_factory=list)


def global_search(
    reports: Sequence[CommunityReport],
    query: str,
    *,
    level: int | None = None,
    top_reports: int = 3,
    max_points: int = 6,
    embedder: Embedder | None = None,
    llm: LLM | None = None,
) -> GlobalSearchResult:
    """Rank community reports at ``level`` (default: the coarsest available) and extract or synthesise points."""
    if not reports:
        return GlobalSearchResult(query=query, level=level or 0)
    level = min(r.level for r in reports) if level is None else level
    pool = [r for r in reports if r.level == level]
    report_scores = _similarities(query, [r.as_text() for r in pool], embedder)
    ranked = sorted(zip(pool, report_scores, strict=True), key=lambda item: (-item[1], item[0].community_id))
    chosen = ranked[: max(top_reports, 1)]
    summary = [RankedReport(community_id=r.community_id, title=r.title, score=round(s, 4)) for r, s in ranked]
    if llm is not None:
        try:
            mapped = [r for r, _ in ranked[: max(top_reports * 2, 4)]]
            return _map_reduce(llm, query, level, mapped, summary, max_points)
        except LLMError:
            pass  # fall back to the extractive answer below
    candidates = [(report, finding) for report, _ in chosen for finding in report.findings]
    finding_scores = _similarities(query, [f.text for _, f in candidates], embedder)
    report_weight = {r.community_id: s for r, s in chosen}
    scored = sorted(
        zip(candidates, finding_scores, strict=True),
        key=lambda item: (-(item[1] + 0.5 * report_weight[item[0][0].community_id]), item[0][1].text),
    )
    points: list[GlobalPoint] = []
    seen: set[str] = set()
    for (report, finding), score in scored:
        if finding.text in seen:
            continue
        seen.add(finding.text)
        points.append(
            GlobalPoint(text=finding.text, community_id=report.community_id, score=round(score, 4),
                        chunk_ids=finding.chunk_ids)
        )
        if len(points) >= max_points:
            break
    return GlobalSearchResult(query=query, level=level, reports=summary[:top_reports], points=points)


def _map_reduce(
    llm: LLM,
    query: str,
    level: int,
    reports: Sequence[CommunityReport],
    summary: list[RankedReport],
    max_points: int,
) -> GlobalSearchResult:
    points: list[GlobalPoint] = []
    for report in reports:
        messages = [
            {
                "role": "system",
                "content": "Extract the key points from this community report that help answer the question. "
                "Score each point 0-100 for how useful it is. Return no points if the report is irrelevant.",
            },
            {"role": "user", "content": f"Question: {query}\n\nReport {report.community_id}:\n{report.as_text()}"},
        ]
        mapped = complete_structured(llm, messages, _MapResponse)
        chunk_ids = sorted({c for f in report.findings for c in f.chunk_ids})
        points += [
            GlobalPoint(text=p.text, community_id=report.community_id, score=p.score / 100, chunk_ids=chunk_ids)
            for p in mapped.points
            if p.score > 0 and p.text.strip()
        ]
    points.sort(key=lambda p: -p.score)
    points = points[: max_points * 2]
    if not points:
        return GlobalSearchResult(query=query, level=level, reports=summary[: len(reports)], method="llm-map-reduce",
                                  answer="None of the community reports is relevant to this question.")
    context = "\n".join(f"- ({p.community_id}, score {p.score:.2f}) {p.text}" for p in points)
    messages = [
        {
            "role": "system",
            "content": "Write a concise answer to the question from the key points. Cite community ids in "
            "parentheses. Do not add facts that are not in the points.",
        },
        {"role": "user", "content": f"Question: {query}\n\nKey points:\n{context}"},
    ]
    answer = clean_text(llm.complete(messages, temperature=0.2, max_tokens=700))
    return GlobalSearchResult(
        query=query,
        level=level,
        reports=summary[: len(reports)],
        points=points[:max_points],
        answer=answer,
        method="llm-map-reduce",
    )
