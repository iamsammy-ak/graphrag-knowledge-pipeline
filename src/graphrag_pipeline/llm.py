"""A small LLM provider abstraction with an OpenRouter implementation.

* ``LLM`` is the protocol every component depends on: ``complete(messages) -> str``.
* ``complete_structured`` asks for JSON, validates it into a Pydantic model and makes one repair round-trip.
* ``OpenRouterLLM`` uses the OpenAI SDK against OpenRouter, falls back across models on rate limits, server errors
  and timeouts, and caches every successful reply on disk keyed by a hash of the request.
* ``resolve_llm`` picks OpenRouter when a key is configured and offline mode is not forced, otherwise ``None``;
  every component then uses its deterministic offline stand-in.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from .config import Settings, load_settings, offline_forced

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_CACHE_DIR = Path(".cache") / "llm"
T = TypeVar("T", bound=BaseModel)
Message = dict[str, str]

_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)
_FENCE = re.compile(r"```(?:json|markdown|md)?\s*(.*?)```", re.S | re.I)


class LLMError(RuntimeError):
    """No configured model produced a usable answer."""


class LLM(Protocol):
    """Anything that can turn chat messages into text."""

    name: str

    def complete(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.2,
        max_tokens: int = 2000,
        json_mode: bool = False,
    ) -> str: ...


def clean_text(text: str) -> str:
    """Drop reasoning tags and a wrapping code fence from a free-text reply."""
    text = _THINK.sub("", text or "").strip()
    fenced = _FENCE.fullmatch(text)
    return fenced.group(1).strip() if fenced else text


def extract_json(text: str) -> dict[str, Any]:
    """Pull the first JSON object out of a model reply (handles fences, prose and reasoning tags)."""
    text = _THINK.sub("", text or "").strip()
    fenced = _FENCE.search(text)
    if fenced:
        text = fenced.group(1).strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("no JSON object in the model reply") from None
        value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("the model reply is JSON but not an object")
    return value


def complete_structured(
    llm: LLM,
    messages: list[Message],
    schema: type[T],
    *,
    temperature: float = 0.1,
    max_tokens: int = 4000,
) -> T:
    """Ask for JSON matching ``schema``; on an invalid reply, send the error back once and retry."""
    schema_hint = json.dumps(schema.model_json_schema(), separators=(",", ":"))
    instruction = (
        "Reply with one JSON object only: no prose and no code fences. "
        f"It must validate against this JSON Schema:\n{schema_hint}"
    )
    request = [*messages]
    if request and request[0]["role"] == "system":
        request[0] = {"role": "system", "content": f"{request[0]['content']}\n\n{instruction}"}
    else:
        request.insert(0, {"role": "system", "content": instruction})
    raw = llm.complete(request, temperature=temperature, max_tokens=max_tokens, json_mode=True)
    try:
        return schema.model_validate(extract_json(raw))
    except (ValueError, ValidationError) as first_error:
        repair = [
            *request,
            {"role": "assistant", "content": raw[:4000]},
            {
                "role": "user",
                "content": f"That reply could not be used ({str(first_error)[:500]}). "
                "Return the corrected JSON object only.",
            },
        ]
        raw = llm.complete(repair, temperature=0.0, max_tokens=max_tokens, json_mode=True)
        try:
            return schema.model_validate(extract_json(raw))
        except (ValueError, ValidationError) as exc:
            raise LLMError(f"The model reply did not match the expected format: {exc}") from exc


class OpenRouterLLM:
    """OpenRouter chat client with model fallback and an on-disk reply cache."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        cache_dir: Path | None = DEFAULT_CACHE_DIR,
        timeout: float = 90.0,
        client: Any | None = None,
        retry_pause: float = 5.0,
    ) -> None:
        self.settings = settings or load_settings()
        if not self.settings.api_key:
            raise LLMError("OPENROUTER_API_KEY is not set.")
        if client is None:
            from openai import OpenAI

            client = OpenAI(
                base_url=OPENROUTER_BASE_URL,
                api_key=self.settings.api_key,
                default_headers={"X-Title": "GraphRAG Knowledge Pipeline"},
                timeout=timeout,
                max_retries=0,
            )
        self._client = client
        self.cache_dir = cache_dir
        self.retry_pause = retry_pause
        self.name = f"openrouter:{self.settings.models[0]}"
        self.last_model: str | None = None
        self.last_errors: list[str] = []
        self.calls_made = 0

    def _cache_file(self, payload: dict[str, Any]) -> Path | None:
        if self.cache_dir is None:
            return None
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:40]
        return self.cache_dir / f"{digest}.json"

    def complete(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.2,
        max_tokens: int = 2000,
        json_mode: bool = False,
    ) -> str:
        payload = {"models": list(self.settings.models), "messages": messages, "json": json_mode, "t": temperature}
        path = self._cache_file(payload)
        if path is not None and path.exists():
            cached = json.loads(path.read_text(encoding="utf-8"))
            self.last_model = cached.get("model")
            return str(cached["content"])
        content, model = self._call_with_fallback(messages, json_mode, temperature, max_tokens)
        self.last_model = model
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"model": model, "content": content}), encoding="utf-8")
        return content

    def _call_with_fallback(
        self, messages: list[Message], json_mode: bool, temperature: float, max_tokens: int
    ) -> tuple[str, str]:
        """Try each model once; a busy model is skipped (saving free-tier quota) and retried in a second pass."""
        from openai import APIConnectionError, APIStatusError, RateLimitError

        self.last_errors = []
        retryable: list[str] = []
        for round_ in range(2):
            candidates = list(self.settings.models) if round_ == 0 else retryable
            if round_ == 1:
                if not retryable:
                    break
                time.sleep(self.retry_pause)
            retryable = []
            for model in candidates:
                use_json_format = json_mode
                for _ in range(2):  # the second pass only happens when JSON mode itself was rejected
                    kwargs: dict[str, Any] = {
                        "model": model,
                        "messages": messages,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                    }
                    if use_json_format:
                        kwargs["response_format"] = {"type": "json_object"}
                    try:
                        self.calls_made += 1
                        response = self._client.chat.completions.create(**kwargs)
                    except RateLimitError as exc:
                        if "per-day" in str(exc) or "per day" in str(exc):
                            raise LLMError("The daily free-model limit is used up; try again tomorrow.") from exc
                        self.last_errors.append(f"{model}: busy (HTTP 429)")
                        retryable.append(model)
                        break
                    except APIConnectionError as exc:  # includes timeouts
                        self.last_errors.append(f"{model}: {type(exc).__name__}")
                        retryable.append(model)
                        break
                    except APIStatusError as exc:
                        self.last_errors.append(f"{model}: HTTP {exc.status_code}")
                        if exc.status_code == 401:
                            raise LLMError("OpenRouter rejected the API key (HTTP 401).") from exc
                        if exc.status_code == 400 and use_json_format:
                            use_json_format = False
                            continue
                        if exc.status_code >= 500:
                            retryable.append(model)
                        break
                    choices = getattr(response, "choices", None) or []
                    choice = choices[0] if choices else None
                    content = ((choice.message.content if choice else None) or "").strip()
                    if json_mode and getattr(choice, "finish_reason", None) == "length":
                        self.last_errors.append(f"{model}: reply cut off at the token limit")
                        break
                    if content:
                        return clean_text(content) if not json_mode else content, model
                    self.last_errors.append(f"{model}: empty reply")
                    break
        raise LLMError("No model returned an answer. " + "; ".join(self.last_errors[-6:]))


def resolve_llm(offline: bool = False, settings: Settings | None = None) -> LLM | None:
    """Return an OpenRouter client when a key exists and offline mode is not forced; otherwise ``None``."""
    if offline or offline_forced():
        return None
    settings = settings or load_settings()
    if not settings.has_key:
        return None
    return OpenRouterLLM(settings)
