"""A small dependency-free lexical relevance score (used when no embedding model is available)."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence

_WORD = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    "a an the of in on at for to and or is are was were which who what where how does do did with by x".split()
)
_SUFFIXES = ("ing", "ed", "es", "s")


def _stem(word: str) -> str:
    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[: -len(suffix)]
    return word


def terms(text: str) -> list[str]:
    """Lower-cased, stop-word-free, crudely stemmed words ("studied" -> "studi", "makes" -> "make")."""
    return [_stem(w) for w in _WORD.findall(text.lower()) if w not in _STOP]


def lexical_scores(query: str, texts: Sequence[str]) -> list[float]:
    """IDF-weighted, saturating term overlap between the query and each text (a light BM25 variant)."""
    docs = [terms(t) for t in texts]
    query_terms = set(terms(query))
    df = Counter(w for doc in docs for w in set(doc))
    n = len(docs) or 1
    scores = []
    for doc in docs:
        counts = Counter(doc)
        length = 1 + math.log(1 + len(doc))
        score = sum(math.log(1 + n / (1 + df[w])) * counts[w] / (counts[w] + 1.2) for w in query_terms if w in counts)
        scores.append(score / length)
    return scores
