"""Entity and relation extraction behind one interface.

* ``SpacyExtractor``: offline classical NLP (NER + dependency patterns + co-occurrence fallback).
* ``LLMExtractor``: structured LLM extraction with evidence grounding, used only when an API key is configured.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from ..llm import LLM
from ..schemas import Chunk, ChunkExtraction
from .llm_extractor import LLMExtractor
from .spacy_extractor import SpacyExtractor
from .vocab import CANONICAL_RELATIONS, canonical_relation

__all__ = [
    "CANONICAL_RELATIONS",
    "Extractor",
    "LLMExtractor",
    "SpacyExtractor",
    "canonical_relation",
    "get_extractor",
]


class Extractor(Protocol):
    """Turns chunks into entity mentions and relations (one ``ChunkExtraction`` per chunk, same order)."""

    name: str

    def extract(self, chunks: Sequence[Chunk]) -> list[ChunkExtraction]: ...


def get_extractor(kind: str = "auto", llm: LLM | None = None, **spacy_options: Any) -> Extractor:
    """``auto`` uses the LLM when one is available and spaCy otherwise; ``llm`` without a client is an error."""
    if kind == "auto":
        kind = "llm" if llm is not None else "spacy"
    if kind == "spacy":
        return SpacyExtractor(**spacy_options)
    if kind == "llm":
        if llm is None:
            raise ValueError("LLM extraction needs OPENROUTER_API_KEY; use the spaCy extractor offline.")
        return LLMExtractor(llm, fallback=SpacyExtractor(**spacy_options))
    raise ValueError(f"Unknown extractor '{kind}' (expected auto, spacy or llm)")
