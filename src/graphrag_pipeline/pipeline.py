"""End-to-end build: documents -> chunks -> extraction -> resolution -> graph -> communities -> reports.

An index directory holds ``graph.json`` (graph + chunks, reloadable), ``graph.graphml`` (for other graph tools),
``communities.json`` (communities and their reports) and ``manifest.json`` (settings and counts).
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from . import __version__
from .communities import DEFAULT_RESOLUTIONS, detect_communities, modularity
from .extraction import get_extractor
from .graph import KnowledgeGraph
from .ingest import chunk_documents, load_documents
from .llm import LLM
from .reports import ExtractiveReporter, LLMReporter, build_reports
from .resolution import EntityResolver
from .schemas import Community, CommunityReport, Document

ProgressFn = Callable[[str], None]


class BuildConfig(BaseModel):
    """Settings for one index build."""

    max_chars: int = 800
    extractor: str = "auto"
    cooccurrence: bool = True
    resolutions: list[float] = Field(default_factory=lambda: list(DEFAULT_RESOLUTIONS))
    seed: int = 42
    min_report_size: int = 2


@dataclass
class GraphIndex:
    """A built (or loaded) index: the graph plus its communities, reports and build manifest."""

    kg: KnowledgeGraph
    communities: list[Community]
    reports: list[CommunityReport]
    manifest: dict[str, Any] = field(default_factory=dict)

    def report(self, community_id: str) -> CommunityReport | None:
        return next((r for r in self.reports if r.community_id == community_id), None)

    def community(self, community_id: str) -> Community | None:
        return next((c for c in self.communities if c.id == community_id), None)

    def save(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.kg.save_json(directory / "graph.json")
        self.kg.save_graphml(directory / "graph.graphml")
        payload = {
            "communities": [c.model_dump() for c in self.communities],
            "reports": [r.model_dump() for r in self.reports],
        }
        (directory / "communities.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")
        (directory / "manifest.json").write_text(json.dumps(self.manifest, indent=1), encoding="utf-8")
        return directory

    @classmethod
    def load(cls, directory: str | Path) -> GraphIndex:
        directory = Path(directory)
        if not (directory / "graph.json").exists():
            raise FileNotFoundError(f"No index in {directory} (run `graphrag build` first)")
        kg = KnowledgeGraph.load_json(directory / "graph.json")
        payload = json.loads((directory / "communities.json").read_text(encoding="utf-8"))
        manifest_path = directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        return cls(
            kg=kg,
            communities=[Community(**c) for c in payload["communities"]],
            reports=[CommunityReport(**r) for r in payload["reports"]],
            manifest=manifest,
        )


def build_index(
    source: str | Path | Sequence[Document],
    config: BuildConfig | None = None,
    *,
    llm: LLM | None = None,
    progress: ProgressFn | None = None,
) -> GraphIndex:
    """Build an index from a folder, a file or in-memory documents.

    The extractor and reporter use the LLM when ``llm`` is given (and ``config.extractor`` allows it); otherwise
    the offline stand-ins (spaCy rules, extractive reports) run.
    """
    config = config or BuildConfig()
    say = progress or (lambda _message: None)
    started = time.perf_counter()
    documents = load_documents(source) if isinstance(source, (str, Path)) else list(source)
    chunks = chunk_documents(documents, config.max_chars)
    say(f"{len(documents)} documents -> {len(chunks)} chunks")
    extractor = get_extractor(config.extractor, llm, cooccurrence=config.cooccurrence)
    extractions = extractor.extract(chunks)
    say(f"extracted with {extractor.name}")
    resolution = EntityResolver().resolve(extractions, chunks)
    kg = KnowledgeGraph.build(documents, chunks, extractions, resolution)
    stats = kg.stats()
    say(f"{stats['entities']} entities, {stats['relations']} relations")
    communities = detect_communities(kg, config.resolutions, seed=config.seed)
    use_llm_reports = llm is not None and extractor.name == "llm"
    reporter = LLMReporter(llm) if use_llm_reports and llm is not None else ExtractiveReporter()
    reports = build_reports(kg, communities, reporter, min_size=config.min_report_size)
    say(f"{len(communities)} communities, {len(reports)} reports ({reporter.name})")
    levels = sorted({c.level for c in communities})
    manifest = {
        "version": __version__,
        "config": config.model_dump(),
        "extractor": extractor.name,
        "reporter": reporter.name,
        "stats": stats,
        "levels": [
            {
                "level": level,
                "resolution": sorted(config.resolutions)[level],
                "communities": sum(1 for c in communities if c.level == level),
                "modularity": round(modularity(kg, communities, level), 4),
            }
            for level in levels
        ],
        "build_seconds": round(time.perf_counter() - started, 2),
    }
    return GraphIndex(kg=kg, communities=communities, reports=reports, manifest=manifest)
