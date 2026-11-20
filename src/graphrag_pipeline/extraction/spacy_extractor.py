"""Offline extractor: spaCy named-entity recognition plus dependency-based relation patterns.

This is the deterministic stand-in for LLM extraction, and it is plain classical NLP: precise on simple declarative
sentences and blind to anything that needs world knowledge. Relation patterns:

* subject-verb-object, including passives with a "by" agent ("Nimbrel Labs was acquired by Velltrix Systems");
* verb + preposition ("is headquartered in", "invested in", "studied at"), with "as <role>" refinement
  ("joined Brightwell Cells as chief executive");
* copula + role noun ("X is the chief executive of Y"), including possessives ("its chief executive is X");
* apposition ("Felix Ambrose, the chief executive of Greyfen Materials,");
* participial and relative clauses ("a lidar unit made by Velltrix", "Velltrix Systems, which bought it");
* a low-weight ``RELATED_TO`` co-occurrence edge for entity pairs that share a sentence but match no pattern.

Two corpus-level aids make it usable on unfamiliar names: a gazetteer pass (a name recognised anywhere is recognised
everywhere, including short forms such as "Corvane" for "Corvane Motors"), and a small salience model that resolves
"he/she/it/its" and definite descriptions ("the company", "the sensor") to the most recent compatible entity in the
chunk, falling back to the document title (the topic of the page).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from typing import TYPE_CHECKING

from ..ingest import mask_markdown, split_sentences
from ..schemas import COOCCURRENCE, Chunk, ChunkExtraction, EntityMention, EntityType, RelationMention
from .vocab import (
    CANONICAL_RELATIONS,
    GENERIC_NOUNS,
    LOCATION_HEAD_WORDS,
    NER_LABELS,
    ORG_HEAD_WORDS,
    PRODUCT_HEAD_WORDS,
    canonical_relation,
    refine_type,
)

if TYPE_CHECKING:
    from spacy.language import Language
    from spacy.tokens import Doc, Span, Token

DEFAULT_MODEL = "en_core_web_sm"

_PERSON_PRONOUNS = frozenset({"he", "she", "him", "her", "his", "hers"})
_THING_PRONOUNS = frozenset({"it", "its"})
_RELATIVE_PRONOUNS = frozenset({"which", "that", "who", "whom", "whose"})
_DEFINITE = frozenset({"the", "that", "this"})
_COPULAS = frozenset({"be", "become", "remain"})
_QUANTITY_NOUNS = frozenset({"fleet", "number", "range", "set", "group", "pair", "line", "series", "family", "batch"})
_NOUN_PREPS = frozenset({"of", "at", "for", "in", "with"})
_OBJECT_PREPS = frozenset({"at", "in", "from", "with", "to"})  # "studied economics at X", "lent 40 million to X"
_NOUN_METHODS = frozenset({"copula", "appos"})  # predicates built from role nouns
_MODEL_NUMBER = re.compile(r"^[A-Z][A-Za-z]*-?\d+[A-Za-z]*$")
_CONJUNCTIONS = frozenset({"and", "&"})


def load_pipeline(model: str = DEFAULT_MODEL) -> Language:
    """Load (once per process) a spaCy pipeline, with a clear message when the model is missing."""
    return _load_pipeline(model)


@lru_cache(maxsize=4)
def _load_pipeline(model: str) -> Language:
    import spacy

    try:
        return spacy.load(model)
    except OSError as exc:
        raise RuntimeError(f"spaCy model '{model}' is not installed: run `python -m spacy download {model}`") from exc


@dataclass(eq=False)
class _Ent:
    name: str
    type: EntityType
    start: int | None
    end: int | None
    description: str = ""


@dataclass
class _ChunkContext:
    """Entities seen so far in one chunk, for pronoun and definite-description resolution."""

    topic: _Ent | None
    memory: dict[str, _Ent]  # entities seen earlier in the same document, by name
    history: list[_Ent] = field(default_factory=list)
    subjects: list[_Ent] = field(default_factory=list)


@dataclass
class Lexicon:
    """Corpus-level knowledge for the second pass: known names and words that occur in lower case."""

    names: dict[tuple[str, ...], EntityType] = field(default_factory=dict)
    lower_words: frozenset[str] = frozenset()


def looks_like_person(name: str) -> bool:
    """Two or three capitalised words with no organisation, place or product cue ("Ilse Varga")."""
    words = name.split()
    cues = ORG_HEAD_WORDS | LOCATION_HEAD_WORDS | PRODUCT_HEAD_WORDS
    return 2 <= len(words) <= 3 and all(w[:1].isupper() and w.isalpha() and w.lower() not in cues for w in words)


def _is_namey(token: Token) -> bool:
    if _MODEL_NUMBER.match(token.text):
        return True
    return token.pos_ == "PROPN" and token.text[:1].isupper()


def _trim_span(span: Span) -> Span | None:
    """Drop leading determiners/punctuation and trailing lower-case words, possessives and punctuation."""
    start, end = span.start, span.end
    doc = span.doc
    while start < end and (doc[start].pos_ in {"DET", "PRON", "PUNCT"} or doc[start].text[:1].islower()):
        start += 1
    while end > start and (
        doc[end - 1].pos_ in {"PUNCT", "PART"} or (doc[end - 1].text[:1].islower() and doc[end - 1].is_alpha)
    ):
        end -= 1
    if start >= end:
        return None
    trimmed = doc[start:end]
    if not any(ch.isupper() for ch in trimmed.text):
        return None
    return trimmed


def _key(span: Span) -> tuple[str, ...]:
    return tuple(t.text for t in span)


def _candidate_spans(doc: Doc, lexicon: Lexicon) -> list[tuple[Span, EntityType]]:
    """NER spans mapped to our types, then known names, then proper-noun / model-number runs."""
    from spacy.util import filter_spans

    found: list[tuple[Span, EntityType]] = []
    covered: set[int] = set()  # tokens claimed by a kept span
    ner_tokens: set[int] = set()  # tokens inside any NER span (dates and numbers included)

    def sentence_initial_common_word(span: Span) -> bool:
        return len(span) == 1 and span.start == 0 and span[0].lower_ in lexicon.lower_words

    for ent in doc.ents:
        ner_tokens.update(range(ent.start, ent.end))
        label = NER_LABELS.get(ent.label_)
        trimmed = _trim_span(ent) if label else None
        if label is None or trimmed is None or sentence_initial_common_word(trimmed):
            continue
        covered.update(range(ent.start, ent.end))
        joiners = [t.i for t in trimmed if t.lower_ in _CONJUNCTIONS]
        if joiners:  # "Coastreach and Ashlenmark": keep the parts only when both are known names
            left, right = _trim_span(doc[trimmed.start : joiners[0]]), _trim_span(doc[joiners[0] + 1 : trimmed.end])
            if left is not None and right is not None and _key(left) in lexicon.names and _key(right) in lexicon.names:
                found += [(left, lexicon.names[_key(left)]), (right, lexicon.names[_key(right)])]
            continue
        found.append((trimmed, lexicon.names.get(_key(trimmed)) or refine_type(trimmed.text, label)))
    if lexicon.names:
        longest = max(len(key) for key in lexicon.names)
        i = 0
        while i < len(doc):
            for size in range(min(longest, len(doc) - i), 0, -1):
                key = tuple(t.text for t in doc[i : i + size])
                if key in lexicon.names and not covered.intersection(range(i, i + size)):
                    found.append((doc[i : i + size], lexicon.names[key]))
                    covered.update(range(i, i + size))
                    i += size - 1
                    break
            i += 1
    run: list[int] = []
    for token in [*doc, None]:
        if token is not None and token.i not in covered | ner_tokens and _is_namey(token):
            run.append(token.i)
            continue
        if run:
            trimmed = _trim_span(doc[run[0] : run[-1] + 1])
            # a lone capitalised word opening a sentence is only a name if it is known from elsewhere,
            # unless the "sentence" is a verbless heading such as "# Stackbot"
            heading = not any(t.pos_ in {"VERB", "AUX"} for t in doc)
            if trimmed is not None and (heading or not (len(trimmed) == 1 and trimmed.start == 0)):
                found.append((trimmed, refine_type(trimmed.text, "CONCEPT")))
            run = []
    kept = {(s.start, s.end) for s in filter_spans([span for span, _ in found])}
    return [(span, etype) for span, etype in found if (span.start, span.end) in kept]


def _described_type(span: Span) -> EntityType | None:
    """Type implied by a copula or appositive description: "X is a battery manufacturer" -> ORGANIZATION."""
    root = span.root
    nouns = [c for c in root.children if c.dep_ == "appos"]
    if root.dep_ == "nsubj" and root.head.lemma_ in _COPULAS:
        nouns += [c for c in root.head.children if c.dep_ == "attr"]
    for noun in nouns:
        implied = GENERIC_NOUNS.get(noun.lemma_.lower())
        if implied is not None:
            return implied
    return None


def build_lexicon(docs: Sequence[Doc]) -> Lexicon:
    """First pass over the corpus: every recognised name, plus safe short forms.

    Short forms are the first word of a multi-word organisation or product ("Corvane" for "Corvane Motors") and the
    surname of a person ("Okafor"), added only when the word never appears in lower case in the corpus. Ambiguous
    short forms prefer the organisation reading ("Greyfen" for "Greyfen Materials" rather than "Greyfen E7").
    """
    lower_words = frozenset(t.lower_ for doc in docs for t in doc if t.text[:1].islower())
    first_pass = Lexicon(lower_words=lower_words)
    votes: dict[tuple[str, ...], dict[str, int]] = {}
    described: set[tuple[str, ...]] = set()
    for doc in docs:
        for span, etype in _candidate_spans(doc, first_pass):
            bucket = votes.setdefault(_key(span), {})
            bucket[etype] = bucket.get(etype, 0) + 1
            implied = _described_type(span)
            if implied is not None:  # "Quorvale is the capital of ..." outweighs a few NER guesses
                bucket[implied] = bucket.get(implied, 0) + 3
                described.add(_key(span))
    names: dict[tuple[str, ...], EntityType] = {}
    for key, bucket in votes.items():
        ranked = sorted(bucket.items(), key=lambda kv: (kv[0] == "CONCEPT", -kv[1], kv[0]))
        names[key] = ranked[0][0]  # type: ignore[assignment]
    weak = {k for k, bucket in votes.items() if len(k) == 1 and sum(bucket.values()) <= 3 and k not in described}
    short: dict[tuple[str, ...], set[EntityType]] = {}
    for key, etype in names.items():
        short_type: EntityType
        if len(key) < 2:
            continue
        if etype in {"ORGANIZATION", "PRODUCT"}:
            word, short_type = key[0], etype
        elif etype == "PERSON" or looks_like_person(" ".join(key)):
            word, short_type = key[-1], "PERSON"
        else:
            continue
        if word.isalpha() and len(word) >= 4 and word.lower() not in lower_words and word.lower() not in GENERIC_NOUNS:
            short.setdefault((word,), set()).add(short_type)
    for key, types in short.items():
        if (key in names and key not in weak) or (len(types) > 1 and "ORGANIZATION" not in types):
            continue
        names[key] = "ORGANIZATION" if "ORGANIZATION" in types else next(iter(types))
    return Lexicon(names=names, lower_words=lower_words)


class _SentenceParser:
    """Applies the relation patterns to one parsed sentence whose entity spans have been merged into tokens."""

    def __init__(self, doc: Doc, ents: dict[int, _Ent], ctx: _ChunkContext) -> None:
        self.doc = doc
        self.ents = ents
        self.ctx = ctx
        self.relations: list[tuple[_Ent, str, _Ent, str]] = []
        self._seen: set[tuple[str, str, str]] = set()

    # ------------------------------------------------------------------ reference resolution
    def _before(self, index: int) -> list[_Ent]:
        local = [ent for i, ent in sorted(self.ents.items()) if i < index]
        return [*reversed(local), *reversed(self.ctx.history)]

    def _topic_if(self, types: tuple[str, ...]) -> _Ent | None:
        topic = self.ctx.topic
        return topic if topic is not None and topic.type in types else None

    def subjects_before(self, index: int) -> list[_Ent]:
        """Entities in subject position before ``index`` (most recent first), including resolved references."""
        local = [
            ent
            for t in self.doc
            if t.i < index and t.dep_.startswith("nsubj") and (ent := self.resolve(t)) is not None
        ]
        return [*reversed(local), *reversed(self.ctx.subjects)]

    def _person(self, index: int) -> _Ent | None:
        before = self._before(index)
        for ent in before:
            if ent.type == "PERSON":
                return ent
        for ent in before:  # NER often mislabels unfamiliar person names as organisations
            if ent.type in {"ORGANIZATION", "CONCEPT"} and looks_like_person(ent.name):
                return ent
        return self._topic_if(("PERSON",))

    def _thing(self, index: int, exclude: _Ent | None) -> _Ent | None:
        types = ("ORGANIZATION", "PRODUCT")
        for ent in [*self.subjects_before(index), *self._before(index)]:
            if ent.type in types and ent is not exclude and ent.name != getattr(exclude, "name", None):
                return ent
        return self._topic_if(types)

    def _definite(self, token: Token) -> _Ent | None:
        """Resolve "the company"/"the sensor": a name or description match, then the latest subject of the
        implied type, then the page topic, then the latest mention of that type."""
        noun = token.lemma_.lower()
        wanted = GENERIC_NOUNS[noun]
        topic = [self.ctx.topic] if self.ctx.topic is not None else []
        candidates = [*self._before(token.i), *topic, *reversed(list(self.ctx.memory.values()))]
        pattern = re.compile(rf"\b{re.escape(noun)}", re.I)
        for ent in candidates:
            if pattern.search(ent.name) or pattern.search(ent.description):
                return ent
        for ent in [*self.subjects_before(token.i), *topic, *candidates]:
            if ent.type == wanted:
                return ent
        return None

    def resolve(self, token: Token, exclude: _Ent | None = None) -> _Ent | None:
        """The entity a token refers to: itself, a compound/appositive name, a pronoun or a definite description.

        ``exclude`` keeps a possessive from resolving to the clause's own subject ("Velltrix is its customer").
        """
        if token.i in self.ents:
            return self.ents[token.i]
        lower = token.lower_
        if lower in _RELATIVE_PRONOUNS and token.head.dep_ == "relcl":
            return self.resolve(token.head.head)
        if lower in _PERSON_PRONOUNS:
            return self._person(token.i)
        if lower in _THING_PRONOUNS:
            return self._thing(token.i, exclude)
        for child in token.children:
            if child.dep_ in {"compound", "nmod", "amod", "appos"} and child.i in self.ents:
                return self.ents[child.i]
        noun = token.lemma_.lower()
        if noun in GENERIC_NOUNS and token.dep_ not in {"attr", "acomp", "appos"}:  # predicate nominals describe
            for child in token.children:  # "the province of Ashlenmark" is Ashlenmark, and a location
                if child.dep_ == "prep" and child.lower_ == "of":
                    named = [self.ents[p.i] for p in child.children if p.dep_ == "pobj" and p.i in self.ents]
                    if named and (GENERIC_NOUNS[noun] == "LOCATION" or named[0].type != "PERSON"):
                        named[0].type = GENERIC_NOUNS[noun]
                        return named[0]
            if any(c.dep_ == "det" and c.lower_ in _DEFINITE for c in token.children):
                return self._definite(token)
        return None

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _expand(token: Token) -> list[Token]:
        return [token, *token.conjuncts]

    def _role(self, noun: Token) -> str:
        """Role key of a noun with its modifiers, e.g. 'chief technology officer' -> 'chief_technology_officer'."""
        modifiers = [
            c.lower_ for c in noun.children if c.dep_ in {"amod", "compound"} and c.i < noun.i and c.i not in self.ents
        ]
        return "_".join([*modifiers, noun.lemma_.lower()])

    def _description(self, noun: Token) -> str:
        words = [t.text for t in self.doc[noun.left_edge.i : noun.i + 1] if t.pos_ != "PUNCT" or t.text == "-"]
        text = " ".join(words).replace(" - ", "-")
        return re.sub(r"^(a|an|the|its|his|her|their)\s+", "", text, flags=re.I)[:120]

    def _describe(self, ent: _Ent, noun: Token) -> None:
        """Attach a description ("a heavy electric truck"); its head noun also settles the entity type."""
        if not ent.description:
            ent.description = self._description(noun)
            implied = GENERIC_NOUNS.get(noun.lemma_.lower())
            if implied is not None:
                ent.type = implied

    def _noun_targets(
        self, noun: Token, *, possessive: bool = True, preps: frozenset[str] = _NOUN_PREPS, anchor: _Ent | None = None
    ) -> list[tuple[str, _Ent]]:
        """Entities attached to a noun: 'of/at/for Y' prepositions and possessives ('its', "Y's")."""
        targets: list[tuple[str, _Ent]] = []
        for child in noun.children:
            if child.dep_ == "prep" and child.lower_ in preps:
                for pobj in (p for p in child.children if p.dep_ == "pobj"):
                    targets += [(child.lower_, e) for item in self._expand(pobj) if (e := self.resolve(item))]
            elif child.dep_ == "poss" and possessive:
                ent = self.resolve(child, exclude=anchor)
                if ent is not None:
                    targets.append(("of", ent))
        return targets

    def _head_entity(self, noun: Token) -> _Ent | None:
        """Entity a modified noun stands for: the noun itself, its appositive anchor or its copula subject."""
        ent = self.resolve(noun)
        if ent is not None:
            return ent
        if noun.dep_ == "appos":
            return self.resolve(noun.head)
        if noun.dep_ in {"attr", "acomp"} and noun.head.lemma_ in _COPULAS:
            for subj in noun.head.children:
                if subj.dep_ == "nsubj":
                    return self.resolve(subj)
        return None

    def emit(self, source: _Ent | None, predicate: str, target: _Ent | None, method: str) -> None:
        if source is None or target is None or source is target or source.name == target.name:
            return
        # "<noun> in <place>" means located-in only for role nouns ("a company in Dunmarrow"), not for any verb
        rel_type, reverse = canonical_relation(predicate, target.type if method in _NOUN_METHODS else None)
        if reverse:
            source, target = target, source
        key = (source.name, rel_type, target.name)
        if key in self._seen:
            return
        self._seen.add(key)
        self.relations.append((source, rel_type, target, method))

    # ------------------------------------------------------------------ patterns
    def _subjects(self, verb: Token) -> tuple[list[_Ent], list[_Ent]]:
        """(active subjects, passive subjects) of a verb, inherited through coordination and participles."""
        active = [e for c in verb.children if c.dep_ == "nsubj" for t in self._expand(c) if (e := self.resolve(t))]
        passive = [e for c in verb.children if c.dep_ == "nsubjpass" for t in self._expand(c) if (e := self.resolve(t))]
        if not active and not passive:
            if verb.dep_ == "acl":
                anchor = self._head_entity(verb.head)
                if anchor is not None:
                    (passive if verb.tag_ == "VBN" else active).append(anchor)
            elif verb.dep_ in {"conj", "xcomp", "advcl"} and verb.head.pos_ in {"VERB", "AUX"}:
                return self._subjects(verb.head)
        return active, passive

    def _object_entities(self, token: Token) -> list[_Ent]:
        found: list[_Ent] = []
        for item in self._expand(token):
            ent = self.resolve(item)
            if ent is None and item.lemma_.lower() in _QUANTITY_NOUNS:  # "a fleet of 140 Wayfarer trucks"
                ent = next((e for _, e in self._noun_targets(item, possessive=False)), None)
            if ent is not None:
                found.append(ent)
        return found

    def _verb(self, verb: Token) -> None:
        lemma = verb.lemma_.lower()
        particle = next((c.lower_ for c in verb.children if c.dep_ == "prt"), None)
        if particle:
            lemma = f"{lemma}_{particle}"
        active, passive = self._subjects(verb)
        if lemma in _COPULAS:
            self._copula(verb, active)
            return
        objects: list[_Ent] = []
        preps: list[tuple[str, _Ent]] = []
        for dobj in (c for c in verb.children if c.dep_ == "dobj"):
            found = self._object_entities(dobj)
            objects += found
            if not found:  # "studied economics at X" -> study_at
                preps += self._noun_targets(dobj, possessive=False, preps=_OBJECT_PREPS)
        agents: list[_Ent] = []
        role: str | None = None
        for child in verb.children:
            if child.dep_ == "agent":
                agents += [e for p in child.children if p.dep_ == "pobj" for e in self._object_entities(p)]
            elif child.dep_ in {"prep", "dative"}:  # spaCy tags "lent X to Y" as a dative
                for pobj in (p for p in child.children if p.dep_ == "pobj"):
                    if child.lower_ == "as":  # "as chief executive", "as a division of Velltrix Systems"
                        role = self._role(pobj)
                        preps += self._noun_targets(pobj, possessive=False)
                        continue
                    found = self._object_entities(pobj)
                    preps += [(child.lower_, e) for e in found]
                    if not found:  # "sits on the board of Corvane Motors" -> board_of
                        nested = self._noun_targets(pobj, possessive=False)
                        preps += [(f"{pobj.lemma_.lower()}_{p}", e) for p, e in nested]

        def with_role(default: str, prep: str) -> str:
            """Prefer the 'as <role>' reading when it maps onto the vocabulary ("joined X as chief executive")."""
            if role is not None and canonical_relation(f"{role}_{prep}")[0] in CANONICAL_RELATIONS:
                return f"{role}_{prep}"
            return default

        for subject in active:
            for obj in objects:
                self.emit(subject, with_role(lemma, "of"), obj, "svo")
            for prep, target in preps:
                key = prep if "_" in prep else f"{lemma}_{prep}"
                self.emit(subject, with_role(key, prep.split("_")[-1]), target, "prep")
        for agent in agents:
            for patient in passive:
                self.emit(agent, lemma, patient, "passive")
        for patient in passive:
            for prep, target in preps:
                key = prep if "_" in prep else f"{lemma}_{prep}"
                self.emit(patient, key, target, "passive")

    def _copula(self, verb: Token, subjects: list[_Ent]) -> None:
        for attr in (c for c in verb.children if c.dep_ in {"attr", "acomp"}):
            attr_ent = self.resolve(attr)
            if attr_ent is not None:
                # "Its chief executive is Ravi Castellan": the role sits on the subject noun.
                for subj in (c for c in verb.children if c.dep_ == "nsubj" and c.i not in self.ents):
                    for prep, target in self._noun_targets(subj, anchor=attr_ent):
                        self.emit(attr_ent, f"{self._role(subj)}_{prep}", target, "copula")
                continue
            if attr.pos_ in {"NOUN", "PROPN"}:
                for subject in subjects:
                    self._describe(subject, attr)
            for subject in subjects:
                for prep, target in self._noun_targets(attr, anchor=subject):
                    self.emit(subject, f"{self._role(attr)}_{prep}", target, "copula")

    def _apposition(self, token: Token) -> None:
        head = token.head
        if token.i in self.ents and head.i not in self.ents and head.pos_ == "NOUN":
            # "its founder, Greta Solberg" / "the chief executive of Velltrix, Mara Okafor"
            person = self.ents[token.i]
            for prep, target in self._noun_targets(head, anchor=person):
                self.emit(person, f"{self._role(head)}_{prep}", target, "appos")
            return
        anchor = self.resolve(head)
        if anchor is None or token.i in self.ents or token.pos_ != "NOUN":
            return
        self._describe(anchor, token)
        for prep, target in self._noun_targets(token, anchor=anchor):
            self.emit(anchor, f"{self._role(token)}_{prep}", target, "appos")

    def run(self) -> list[tuple[_Ent, str, _Ent, str]]:
        for token in self.doc:
            if token.pos_ in {"VERB", "AUX"} and token.dep_ not in {"aux", "auxpass"}:
                self._verb(token)
            if token.dep_ == "appos":
                self._apposition(token)
        return self.relations


class SpacyExtractor:
    """Classical-NLP extractor (spaCy NER + dependency patterns). Deterministic and offline."""

    name = "spacy-rules"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        cooccurrence: bool = True,
        cooccurrence_weight: float = 0.2,
        max_cooccurrence_entities: int = 6,
    ) -> None:
        self.model = model
        self.cooccurrence = cooccurrence
        self.cooccurrence_weight = cooccurrence_weight
        self.max_cooccurrence_entities = max_cooccurrence_entities

    def extract(self, chunks: Sequence[Chunk]) -> list[ChunkExtraction]:
        """Extract entity mentions and relations from each chunk (chunks of a document should be in order)."""
        nlp = load_pipeline(self.model)
        jobs: list[tuple[int, int, int]] = []  # (chunk position, start, end)
        texts: list[str] = []
        for position, chunk in enumerate(chunks):
            masked = mask_markdown(chunk.text)
            for start, end in split_sentences(chunk.text):
                segment = masked[start:end]
                start += len(segment) - len(segment.lstrip())
                end = start + len(masked[start:end].rstrip())
                if end > start:
                    jobs.append((position, start, end))
                    texts.append(masked[start:end])
        docs = list(nlp.pipe(texts, batch_size=64))
        lexicon = build_lexicon(docs)
        results = [ChunkExtraction(chunk_id=chunk.id) for chunk in chunks]
        memories: dict[str, dict[str, _Ent]] = {}
        contexts: dict[int, _ChunkContext] = {}
        for (position, start, end), doc in zip(jobs, docs, strict=True):
            chunk = chunks[position]
            memory = memories.setdefault(chunk.doc_id, {})
            ctx = contexts.get(position)
            if ctx is None:
                ctx = contexts[position] = _ChunkContext(topic=memory.get(chunk.title), memory=memory)
            self._sentence(doc, chunk, start, end, ctx, results[position], lexicon)
        for position, ctx in contexts.items():
            results[position].entities = [
                EntityMention(
                    name=ent.name,
                    type=ent.type,
                    chunk_id=chunks[position].id,
                    start=ent.start,
                    end=ent.end,
                    description=ent.description,
                    method="ner",
                )
                for ent in ctx.history
            ]
        return results

    def _sentence(
        self,
        doc: Doc,
        chunk: Chunk,
        start: int,
        end: int,
        ctx: _ChunkContext,
        out: ChunkExtraction,
        lexicon: Lexicon,
    ) -> None:
        candidates = _candidate_spans(doc, lexicon)
        offsets = [(span.start_char, span.end_char, etype) for span, etype in candidates]
        doc.ents = ()
        with doc.retokenize() as retokenizer:  # one token per entity keeps the dependency patterns simple
            for span, _ in candidates:
                retokenizer.merge(span, attrs={"LEMMA": span.text})
        by_start = {token.idx: token.i for token in doc}
        ents = {
            by_start[s]: _Ent(name=doc.text[s:e], type=etype, start=start + s, end=start + e) for s, e, etype in offsets
        }
        parser = _SentenceParser(doc, ents, ctx)
        relations = parser.run()
        evidence = " ".join(doc.text.split())
        linked: set[frozenset[str]] = set()
        for source, rel_type, target, method in relations:
            linked.add(frozenset((source.name, target.name)))
            out.relations.append(
                RelationMention(
                    source=source.name,
                    target=target.name,
                    type=rel_type,
                    chunk_id=chunk.id,
                    evidence=evidence,
                    start=start,
                    end=end,
                    weight=1.0,
                    method=method,
                )
            )
        ordered = [ents[i] for i in sorted(ents)]
        if self.cooccurrence and 2 <= len(ordered) <= self.max_cooccurrence_entities:
            for i, left in enumerate(ordered):
                for right in ordered[i + 1 :]:
                    pair = frozenset((left.name, right.name))
                    if left.name == right.name or pair in linked:
                        continue
                    linked.add(pair)
                    out.relations.append(
                        RelationMention(
                            source=left.name,
                            target=right.name,
                            type=COOCCURRENCE,
                            chunk_id=chunk.id,
                            evidence=evidence,
                            start=start,
                            end=end,
                            weight=self.cooccurrence_weight,
                            method="cooccurrence",
                        )
                    )
        subjects = [ent for t in doc if t.dep_.startswith("nsubj") and (ent := parser.resolve(t)) is not None]
        ctx.history.extend(ordered)
        ctx.subjects.extend(subjects)
        for ent in ordered:
            known = ctx.memory.get(ent.name)
            if known is None or (ent.description and not known.description):
                ctx.memory[ent.name] = ent
