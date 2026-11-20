"""Bundled sample data: a fictional corpus, multi-hop benchmark questions and a hand-labelled extraction gold set.

Every company, person, product and place in the corpus is invented. Any resemblance to real entities is
coincidental.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).resolve().parent
CORPUS_DIR = DATA_DIR / "corpus"
HELDOUT_DIR = DATA_DIR / "heldout"


def corpus_dir() -> Path:
    """Folder with the 54 Markdown documents of the sample corpus."""
    return CORPUS_DIR


def heldout_dir() -> Path:
    """Folder with the 8 held-out documents used only to measure extraction quality."""
    return HELDOUT_DIR


def load_json(name: str) -> Any:
    return json.loads((DATA_DIR / name).read_text(encoding="utf-8"))
