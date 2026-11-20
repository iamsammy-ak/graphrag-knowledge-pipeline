"""HTTP API over a built index (FastAPI). Start it with ``graphrag serve`` (default port 8703)."""

from __future__ import annotations

from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import __version__
from .embeddings import Embedder
from .llm import LLM
from .pipeline import GraphIndex
from .schemas import CommunityReport, Entity, GlobalSearchResult, LocalSearchResult, PathSearchResult
from .search import LocalSearcher, global_search, path_search


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    mode: Literal["local", "global", "path"] = "local"
    retriever: Literal["graph", "vector", "hybrid"] = "hybrid"
    k: int = Field(default=5, ge=1, le=20)
    source: str | None = None
    target: str | None = None
    level: int | None = None


class EntityDetail(BaseModel):
    entity: Entity
    relations: list[dict[str, str]]


def create_app(index: GraphIndex, embedder: Embedder | None = None, llm: LLM | None = None) -> FastAPI:
    """Build the FastAPI app; the searcher (and its embeddings) is created once, here."""
    app = FastAPI(title="GraphRAG Knowledge Pipeline", version=__version__)
    searcher = LocalSearcher(index.kg, embedder)

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "version": __version__, "llm": llm is not None}

    @app.get("/stats")
    def stats() -> dict[str, Any]:
        return {"stats": index.kg.stats(), "levels": index.manifest.get("levels", [])}

    @app.post("/query", response_model=None)
    def query(request: QueryRequest) -> LocalSearchResult | GlobalSearchResult | PathSearchResult:
        if request.mode == "path":
            return path_search(index.kg, request.question, source=request.source, target=request.target)
        if request.mode == "global":
            return global_search(index.reports, request.question, level=request.level, embedder=embedder, llm=llm)
        if request.retriever != "graph" and searcher.vector is None:
            raise HTTPException(status_code=400, detail="this server has no embedding model; use retriever=graph")
        return searcher.search(request.question, k=request.k, retriever=request.retriever, llm=llm)

    @app.get("/communities")
    def communities(level: int = 0) -> list[CommunityReport]:
        return [r for r in index.reports if r.level == level]

    @app.get("/entities/{entity_id}")
    def entity(entity_id: str) -> EntityDetail:
        if entity_id not in index.kg.graph:
            raise HTTPException(status_code=404, detail=f"unknown entity {entity_id}")
        relations = [
            {"source": index.kg.name(r.source), "type": r.type, "target": index.kg.name(r.target), "chunk": r.chunk_id}
            for r in index.kg.relations(include_cooccurrence=False)
            if entity_id in (r.source, r.target)
        ]
        return EntityDetail(entity=index.kg.entity(entity_id), relations=relations)

    return app
