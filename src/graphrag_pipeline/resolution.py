"""Entity resolution: decide which surface mentions name the same thing and give each a canonical id.

Steps (all deterministic):

1. **Normalise** every mention to a match key (ASCII, lower case, no punctuation or possessive, no leading "the",
   no legal suffix such as "Inc").
2. **Merge** keys with a union-find:
   * acronyms, explicit ("Quorvale Institute of Technology (QIT)") or by initials ("QIT");
   * short forms: an organisation or product name's leading words ("Velltrix" -> "Velltrix Systems") and a
     person's surname ("Okafor" -> "Mara Okafor"), only when exactly one compatible long form exists;
   * near-identical strings of the same type (``difflib`` ratio >= 0.92: "Halcyon Sensors" -> "Halcyon Sensor").
3. **Type** each cluster by weighted evidence: NER labels, descriptions ("a battery manufacturer"), surface cues
   ("... Labs") and the roles it plays in relations (the subject of BORN_IN is a person).
"""

from __future__ import annotations

import difflib
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from .extraction.vocab import GENERIC_NOUNS, refine_type
from .ingest import slugify
from .schemas import COOCCURRENCE, ENTITY_TYPES, Chunk, ChunkExtraction, Entity, EntityType

LEGAL_SUFFIXES = frozenset(
    "inc ltd llc plc gmbh corp corporation co sa ag spa srl bv nv limited incorporated".split()
)
_ACRONYM_STOPWORDS = frozenset({"of", "and", "for", "the", "&", "de", "la", "du"})
_EXPLICIT_ACRONYM = re.compile(r"((?:[A-Z][\w&'-]*\s+)(?:(?:of|and|for|the|&|[A-Z][\w&'-]*)\s+)*[A-Z][\w&'-]*)"
                               r"\s*\(([A-Z][A-Z0-9&]{1,7})\)")

# (relation type) -> (type implied for the source, type implied for the target, weight)
_ROLE_EVIDENCE: dict[str, tuple[str | None, str | None, float]] = {
    "BORN_IN": ("PERSON", "LOCATION", 2.0),
    "STUDIED_AT": ("PERSON", "ORGANIZATION", 2.0),
    "FOUNDED": ("PERSON", "ORGANIZATION", 1.0),
    "LEADS": ("PERSON", "ORGANIZATION", 1.0),
    "WORKS_FOR": ("PERSON", "ORGANIZATION", 1.0),
    "LOCATED_IN": (None, "LOCATION", 1.0),
    "MAKES": ("ORGANIZATION", "PRODUCT", 1.0),
    "ACQUIRED": ("ORGANIZATION", "ORGANIZATION", 1.0),
    "INVESTED_IN": ("ORGANIZATION", "ORGANIZATION", 1.0),
    "FINANCED": ("ORGANIZATION", "ORGANIZATION", 1.0),
    "PARTNERED_WITH": ("ORGANIZATION", "ORGANIZATION", 1.0),
    "SUBSIDIARY_OF": ("ORGANIZATION", "ORGANIZATION", 1.0),
}


def normalise_name(name: str) -> str:
    """ASCII, lower case, hyphens to spaces, punctuation and possessives removed, no leading 'the'."""
    text = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii").lower()
    text = re.sub(r"['’]s\b", "", text)
    text = re.sub(r"[-_/]", " ", text)
    text = re.sub(r"[^a-z0-9& ]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return re.sub(r"^the ", "", text)


def match_key(name: str) -> str:
    """The key mentions are grouped by: the normalised name without trailing legal suffixes."""
    words = normalise_name(name).split()
    while len(words) > 1 and words[-1] in LEGAL_SUFFIXES:
        words.pop()
    return " ".join(words)


def initials(name: str) -> str:
    """'Quorvale Institute of Technology' -> 'qit' (stop words skipped)."""
    return "".join(w[0] for w in normalise_name(name).split() if w not in _ACRONYM_STOPWORDS)


class _UnionFind:
    def __init__(self, items: Iterable[str]) -> None:
        self.parent = {item: item for item in items}

    def find(self, item: str) -> str:
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != root:
            self.parent[item], item = root, self.parent[item]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


@dataclass
class _KeyStats:
    surfaces: Counter[str] = field(default_factory=Counter)
    types: defaultdict[str, float] = field(default_factory=lambda: defaultdict(float))
    descriptions: Counter[str] = field(default_factory=Counter)
    chunk_ids: set[str] = field(default_factory=set)
    mentions: int = 0

    def best_type(self) -> str:
        scored = [(score, t) for t, score in self.types.items() if t != "CONCEPT" and score > 0]
        return max(scored)[1] if scored else "CONCEPT"


@dataclass
class Resolution:
    """Canonical entities plus the map from every match key to its entity id."""

    entities: dict[str, Entity]
    key_to_id: dict[str, str]

    def entity_id(self, name: str) -> str | None:
        return self.key_to_id.get(match_key(name))


class EntityResolver:
    """Merges mentions into canonical entities (see the module docstring for the rules)."""

    def __init__(
        self,
        similarity_threshold: float = 0.92,
        *,
        merge_acronyms: bool = True,
        merge_short_forms: bool = True,
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self.merge_acronyms = merge_acronyms
        self.merge_short_forms = merge_short_forms

    # ------------------------------------------------------------------ evidence
    @staticmethod
    def _collect(extractions: Sequence[ChunkExtraction]) -> dict[str, _KeyStats]:
        stats: dict[str, _KeyStats] = defaultdict(_KeyStats)
        for extraction in extractions:
            for mention in extraction.entities:
                key = match_key(mention.name)
                if not key:
                    continue
                item = stats[key]
                item.surfaces[mention.name.strip()] += 1
                item.types[mention.type] += 0.25 if mention.type == "CONCEPT" else 1.0
                item.chunk_ids.add(mention.chunk_id)
                item.mentions += 1
                if mention.description:
                    item.descriptions[mention.description] += 1
                    head = mention.description.lower().split()[-1]
                    if head in GENERIC_NOUNS:
                        item.types[GENERIC_NOUNS[head]] += 3.0
        for extraction in extractions:
            for relation in extraction.relations:
                if relation.type == COOCCURRENCE or relation.type not in _ROLE_EVIDENCE:
                    continue
                source_type, target_type, weight = _ROLE_EVIDENCE[relation.type]
                for name, implied in ((relation.source, source_type), (relation.target, target_type)):
                    key = match_key(name)
                    if implied and key in stats:
                        stats[key].types[implied] += weight
        return stats

    # ------------------------------------------------------------------ merging rules
    def _acronyms(self, uf: _UnionFind, stats: Mapping[str, _KeyStats], texts: Iterable[str]) -> None:
        for text in texts:
            for long_form, short_form in _EXPLICIT_ACRONYM.findall(text):
                long_key, short_key = match_key(long_form), match_key(short_form)
                if long_key in stats and short_key in stats and initials(long_form) == short_key:
                    uf.union(long_key, short_key)
        by_initials: dict[str, list[str]] = defaultdict(list)
        for key in stats:
            if len(key.split()) >= 2:
                by_initials[initials(key)].append(key)
        for key, item in stats.items():
            surface = item.surfaces.most_common(1)[0][0]
            if surface.isupper() and 2 <= len(surface) <= 8 and len(by_initials.get(key, [])) == 1:
                uf.union(by_initials[key][0], key)

    @staticmethod
    def _compatible(short_type: str, long_type: str) -> bool:
        """Place names never merge with non-place names ("Quorvale" vs "Quorvale Institute of Technology")."""
        return (short_type == "LOCATION") == (long_type == "LOCATION")

    def _short_forms(self, uf: _UnionFind, stats: Mapping[str, _KeyStats]) -> None:
        keys = sorted(stats, key=lambda k: len(k.split()))
        for key in keys:
            words = key.split()
            short_type = stats[key].best_type()
            candidates: set[str] = set()
            for other in keys:
                other_words = other.split()
                if len(other_words) <= len(words):
                    continue
                long_type = stats[other].best_type()
                prefix = other_words[: len(words)] == words and long_type != "PERSON"
                surname = len(words) == 1 and other_words[-1] == words[0] and long_type == "PERSON"
                if (prefix or surname) and self._compatible(short_type, long_type):
                    candidates.add(uf.find(other))
            if len(candidates) > 1:  # "Greyfen" -> prefer the long form with the same type
                same = {c for c in candidates if stats[c].best_type() == short_type}
                candidates = same if len(same) == 1 else candidates
            if len(candidates) == 1:
                uf.union(next(iter(candidates)), key)

    def _similar(self, uf: _UnionFind, stats: Mapping[str, _KeyStats]) -> None:
        keys = sorted(k for k in stats if len(k) >= 5)
        for i, a in enumerate(keys):
            for b in keys[i + 1 :]:
                if stats[a].best_type() != stats[b].best_type() or uf.find(a) == uf.find(b):
                    continue
                if difflib.SequenceMatcher(None, a, b).ratio() >= self.similarity_threshold:
                    uf.union(a, b)

    # ------------------------------------------------------------------ public API
    def resolve(self, extractions: Sequence[ChunkExtraction], chunks: Iterable[Chunk] = ()) -> Resolution:
        """Merge mentions into entities; ``chunks`` (optional) enables explicit acronym detection."""
        stats = self._collect(extractions)
        uf = _UnionFind(stats)
        if self.merge_acronyms:
            self._acronyms(uf, stats, (chunk.text for chunk in chunks))
        if self.merge_short_forms:
            self._short_forms(uf, stats)
        self._similar(uf, stats)

        clusters: dict[str, list[str]] = defaultdict(list)
        for key in stats:
            clusters[uf.find(key)].append(key)
        entities: dict[str, Entity] = {}
        key_to_id: dict[str, str] = {}
        used: set[str] = set()
        ordered = sorted(clusters.values(), key=lambda ks: (-sum(stats[k].mentions for k in ks), min(ks)))
        for keys in ordered:
            surfaces: Counter[str] = Counter()
            types: defaultdict[str, float] = defaultdict(float)
            descriptions: Counter[str] = Counter()
            chunk_ids: set[str] = set()
            for key in keys:
                surfaces.update(stats[key].surfaces)
                for etype_name, score in stats[key].types.items():
                    types[etype_name] += score
                descriptions.update(stats[key].descriptions)
                chunk_ids |= stats[key].chunk_ids
            name = max(surfaces, key=lambda s: (len(s.split()), surfaces[s], s))
            scored = [(score, t) for t, score in types.items() if t != "CONCEPT" and score > 0]
            etype = refine_type(name, _as_type(max(scored)[1] if scored else "CONCEPT"))
            base = slugify(name)
            entity_id, n = base, 2
            while entity_id in used:
                entity_id, n = f"{base}-{n}", n + 1
            used.add(entity_id)
            description = max(descriptions, key=lambda d: (descriptions[d], len(d))) if descriptions else ""
            entities[entity_id] = Entity(
                id=entity_id,
                name=name,
                type=etype,
                aliases=sorted(s for s in surfaces if s != name),
                description=description,
                chunk_ids=sorted(chunk_ids),
                mention_count=sum(stats[k].mentions for k in keys),
            )
            for key in keys:
                key_to_id[key] = entity_id
        return Resolution(entities=entities, key_to_id=key_to_id)


def _as_type(value: str) -> EntityType:
    return value if value in ENTITY_TYPES else "CONCEPT"  # type: ignore[return-value]
