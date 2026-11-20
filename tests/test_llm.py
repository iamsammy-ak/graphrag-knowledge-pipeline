from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from pydantic import BaseModel

from graphrag_pipeline.config import Settings, load_settings
from graphrag_pipeline.extraction.llm_extractor import LLMExtractor, coerce_entity_type, locate_evidence
from graphrag_pipeline.ingest import chunk_document, document_from_text
from graphrag_pipeline.llm import LLMError, OpenRouterLLM, complete_structured, extract_json, resolve_llm


class ScriptedLLM:
    """Returns canned replies in order and records the requests (stands in for a real model in tests)."""

    name = "scripted"

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.requests: list[list[dict[str, str]]] = []

    def complete(self, messages: list[dict[str, str]], **_: Any) -> str:
        self.requests.append(messages)
        if not self.replies:
            raise LLMError("no more scripted replies")
        return self.replies.pop(0)


class Answer(BaseModel):
    city: str
    score: int


def test_extract_json_handles_fences_prose_and_reasoning() -> None:
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('<think>hmm</think> Sure: {"a": 2} hope this helps') == {"a": 2}
    with pytest.raises(ValueError):
        extract_json("no json here")
    with pytest.raises(ValueError):
        extract_json("[1, 2]")


def test_complete_structured_repairs_once() -> None:
    llm = ScriptedLLM(['{"city": "Quorvale"}', '{"city": "Quorvale", "score": 3}'])
    assert complete_structured(llm, [{"role": "user", "content": "q"}], Answer) == Answer(city="Quorvale", score=3)
    assert len(llm.requests) == 2 and "could not be used" in llm.requests[1][-1]["content"]
    failing = ScriptedLLM(["{}", "{}"])
    with pytest.raises(LLMError):
        complete_structured(failing, [{"role": "user", "content": "q"}], Answer)


def _status_error(cls: type, status: int) -> Exception:
    response = httpx.Response(status, request=httpx.Request("POST", "https://example.invalid"))
    return cls("error", response=response, body=None)


class FakeCompletions:
    def __init__(self, outcomes: list[Any]) -> None:
        self.outcomes = outcomes
        self.models: list[str] = []

    def create(self, **kwargs: Any) -> Any:
        self.models.append(kwargs["model"])
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        message = SimpleNamespace(content=outcome)
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")])


def _client(outcomes: list[Any]) -> tuple[Any, FakeCompletions]:
    completions = FakeCompletions(outcomes)
    return SimpleNamespace(chat=SimpleNamespace(completions=completions)), completions


def test_openrouter_falls_back_across_models_and_caches(tmp_path: Path) -> None:
    from openai import RateLimitError

    client, completions = _client([_status_error(RateLimitError, 429), "hello"])
    llm = OpenRouterLLM(Settings(api_key="test", models=("m1", "m2")), cache_dir=tmp_path, client=client)
    assert llm.complete([{"role": "user", "content": "hi"}]) == "hello"
    assert completions.models == ["m1", "m2"] and llm.last_model == "m2"
    assert llm.complete([{"role": "user", "content": "hi"}]) == "hello"  # served from the disk cache
    assert completions.models == ["m1", "m2"]
    assert len(list(tmp_path.glob("*.json"))) == 1


def test_openrouter_rejected_key_raises() -> None:
    from openai import AuthenticationError

    client, _ = _client([_status_error(AuthenticationError, 401)])
    llm = OpenRouterLLM(Settings(api_key="bad", models=("m1",)), cache_dir=None, client=client)
    with pytest.raises(LLMError, match="401"):
        llm.complete([{"role": "user", "content": "hi"}])


def test_offline_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    assert resolve_llm() is None  # the autouse fixture removed the key and forced offline mode
    monkeypatch.delenv("GRAPHRAG_OFFLINE")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    monkeypatch.setenv("OPENROUTER_MODELS", "a/b, c/d")
    assert load_settings().models == ("a/b", "c/d")
    assert resolve_llm(offline=True) is None
    with pytest.raises(LLMError):
        OpenRouterLLM(Settings(api_key=None))


def test_llm_extractor_grounds_relations_in_the_text() -> None:
    chunk = chunk_document(document_from_text("d", "Velltrix Systems acquired Nimbrel Labs in 2019."))[0]
    payload = {
        "entities": [{"name": "Velltrix Systems", "type": "Company", "description": "a sensor maker"}],
        "relations": [
            {"source": "Velltrix Systems", "target": "Nimbrel Labs", "type": "acquired",
             "evidence": "velltrix systems acquired nimbrel labs"},
            {"source": "Nimbrel Labs", "target": "Mars", "type": "LOCATED_IN", "evidence": "Nimbrel is on Mars"},
        ],
    }
    extractor = LLMExtractor(ScriptedLLM([json.dumps(payload)]))
    result = extractor.extract([chunk])[0]
    assert [(r.source, r.type, r.target) for r in result.relations] == [
        ("Velltrix Systems", "ACQUIRED", "Nimbrel Labs")
    ]
    relation = result.relations[0]
    assert chunk.text[relation.start : relation.end].lower() == "velltrix systems acquired nimbrel labs"
    assert extractor.dropped_relations == 1
    assert {m.name: m.type for m in result.entities} == {"Velltrix Systems": "ORGANIZATION", "Nimbrel Labs": "CONCEPT"}


def test_llm_extractor_falls_back_per_chunk() -> None:
    chunk = chunk_document(document_from_text("d", "Some text."))[0]

    class Fallback:
        name = "fallback"

        def extract(self, chunks: Any) -> Any:
            from graphrag_pipeline.schemas import ChunkExtraction

            return [ChunkExtraction(chunk_id=c.id) for c in chunks]

    extractor = LLMExtractor(ScriptedLLM([]), fallback=Fallback())
    assert extractor.extract([chunk])[0].chunk_id == chunk.id and extractor.failed_chunks == 1


def test_locate_evidence_and_type_coercion() -> None:
    text = "Corvane Motors  builds\nthe Wayfarer truck."
    assert locate_evidence(text, "builds the Wayfarer") is not None
    assert locate_evidence(text, "") is None
    assert locate_evidence(text, "completely unrelated statement") is None
    assert coerce_entity_type("org") == "ORGANIZATION"
    assert coerce_entity_type("City") == "LOCATION"
    assert coerce_entity_type("gizmo") == "CONCEPT"
