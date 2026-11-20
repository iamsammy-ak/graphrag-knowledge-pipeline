"""Runtime settings read from environment variables.

The library never loads ``.env`` by itself: the command-line entry point does that once at start-up, so tests and
library users stay in control of their environment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_MODELS: tuple[str, ...] = (
    "nvidia/nemotron-3.5-super-vl:free",
    "google/gemma-4-26b-a4b-it:free",
    "inclusionai/ling-3.2-flash:free",
    "google/gemma-4-31b-it:free",
)

#: Set to any non-empty value to force offline mode even when an API key exists.
OFFLINE_ENV = "GRAPHRAG_OFFLINE"


@dataclass(frozen=True)
class Settings:
    """OpenRouter credentials and the ordered list of models to try."""

    api_key: str | None
    models: tuple[str, ...] = DEFAULT_MODELS

    @property
    def has_key(self) -> bool:
        return bool(self.api_key)


def load_settings() -> Settings:
    """Read ``OPENROUTER_API_KEY`` and ``OPENROUTER_MODELS`` (comma-separated) from the environment."""
    key = os.getenv("OPENROUTER_API_KEY", "").strip() or None
    raw_models = os.getenv("OPENROUTER_MODELS", "")
    models = tuple(m.strip() for m in raw_models.split(",") if m.strip()) or DEFAULT_MODELS
    return Settings(api_key=key, models=models)


def offline_forced() -> bool:
    """True when ``GRAPHRAG_OFFLINE`` is set to a non-empty value."""
    return bool(os.getenv(OFFLINE_ENV, "").strip())
