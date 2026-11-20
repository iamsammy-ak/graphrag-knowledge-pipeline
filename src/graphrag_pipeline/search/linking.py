"""Entity linking: find which graph entities a free-text query mentions.

Longest-match first over the query's word n-grams against every normalised name and alias (so "Velltrix's" and
"the Quorvale Institute of Technology" both link), then a fuzzy pass (``difflib``) over capitalised words that did
not match exactly, which tolerates small typos such as "Halcyon Sensr".
"""

from __future__ import annotations

import difflib
import re

from ..graph import KnowledgeGraph
from ..resolution import match_key
from ..schemas import LinkedEntity

_TOKEN = re.compile(r"[A-Za-z0-9][\w'’-]*")
_STOP = frozenset(
    """a an the of in on at for to and or is are was were be by with from which who whom what where when how why
    does did do company city person product""".split()
)


def _is_name_like(word: str) -> bool:
    return word[:1].isupper() or any(ch.isdigit() for ch in word)


def link_entities(
    kg: KnowledgeGraph, text: str, *, max_ngram: int = 6, fuzzy_cutoff: float = 0.86
) -> list[LinkedEntity]:
    """Entities mentioned in ``text``, in order of appearance (exact matches score 1.0, fuzzy ones their ratio)."""
    tokens = list(_TOKEN.finditer(text))
    consumed = [False] * len(tokens)
    found: list[tuple[int, LinkedEntity]] = []

    def add(i: int, j: int, entity_id: str, score: float) -> None:
        for position in range(i, j):
            consumed[position] = True
        node = kg.graph.nodes[entity_id]
        found.append(
            (
                i,
                LinkedEntity(
                    entity_id=entity_id,
                    name=node["name"],
                    type=node["type"],
                    matched=text[tokens[i].start() : tokens[j - 1].end()],
                    score=round(score, 3),
                ),
            )
        )

    for size in range(min(max_ngram, len(tokens)), 0, -1):
        for i in range(len(tokens) - size + 1):
            if any(consumed[i : i + size]):
                continue
            words = [t.group(0) for t in tokens[i : i + size]]
            if size == 1 and (words[0].lower() in _STOP or not _is_name_like(words[0])):
                continue
            entity_id = kg.key_index.get(match_key(text[tokens[i].start() : tokens[i + size - 1].end()]))
            if entity_id is not None:
                add(i, i + size, entity_id, 1.0)

    def candidate(position: int) -> bool:
        word = tokens[position].group(0)
        return not consumed[position] and _is_name_like(word) and word.lower() not in _STOP

    keys = list(kg.key_index)
    i = 0
    while i < len(tokens):  # fuzzy pass over runs of unmatched, name-like words
        if not candidate(i):
            i += 1
            continue
        j = i
        while j < len(tokens) and candidate(j):
            j += 1
        for size in range(j - i, 0, -1):
            for start in range(i, j - size + 1):
                if any(consumed[start : start + size]):
                    continue
                phrase = match_key(text[tokens[start].start() : tokens[start + size - 1].end()])
                close = difflib.get_close_matches(phrase, keys, n=1, cutoff=fuzzy_cutoff) if len(phrase) >= 4 else []
                if close:
                    ratio = difflib.SequenceMatcher(None, phrase, close[0]).ratio()
                    add(start, start + size, kg.key_index[close[0]], ratio)
        i = j
    best: dict[str, tuple[int, LinkedEntity]] = {}
    for position, linked in found:
        current = best.get(linked.entity_id)
        if current is None or linked.score > current[1].score:
            best[linked.entity_id] = (position, linked)
    return [linked for _, linked in sorted(best.values(), key=lambda item: item[0])]
