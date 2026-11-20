"""Load text and Markdown documents and split them into chunks with exact provenance.

Chunking is paragraph-aware: a paragraph is a natural unit of meaning, so each paragraph becomes one chunk and a
heading is attached to the paragraph that follows it. Paragraphs longer than ``max_chars`` are split at sentence
boundaries. Every chunk satisfies ``document.text[chunk.start:chunk.end] == chunk.text``.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

from .schemas import Chunk, Document

TEXT_SUFFIXES = (".md", ".markdown", ".txt")

_HEADING = re.compile(r"^[ \t]{0,3}(#{1,6})[ \t]+(.+?)[ \t#]*$")
_BLOCK_SPLIT = re.compile(r"\n[ \t]*\n")
_ABBREVIATIONS = frozenset(
    {"dr", "mr", "mrs", "ms", "prof", "inc", "ltd", "co", "corp", "st", "vs", "no", "approx", "e.g", "i.e", "etc"}
)
_SENTENCE_END = re.compile(r"[.!?][\"')\]]*\s+(?=[\"'(\[]?[A-Z0-9])")


def slugify(text: str) -> str:
    """Lower-case, ASCII-only, hyphen-separated identifier."""
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "item"


def document_from_text(doc_id: str, text: str, title: str | None = None, source: str = "") -> Document:
    """Build a ``Document`` from raw text; the title defaults to the first Markdown heading."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if title is None:
        title = doc_id
        for line in text.split("\n"):
            match = _HEADING.match(line)
            if match:
                title = match.group(2).strip()
                break
    return Document(id=doc_id, title=title, text=text, source=source)


def load_documents(path: str | Path) -> list[Document]:
    """Load one file or every ``.md``/``.markdown``/``.txt`` file under a directory (sorted, recursive)."""
    root = Path(path)
    if root.is_file():
        files = [root]
        base = root.parent
    elif root.is_dir():
        files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in TEXT_SUFFIXES)
        base = root
    else:
        raise FileNotFoundError(f"No such file or directory: {root}")
    documents: list[Document] = []
    seen: set[str] = set()
    for file in files:
        relative = file.relative_to(base).with_suffix("").as_posix()
        doc_id = slugify(relative.replace("/", "-"))
        if doc_id in seen:
            raise ValueError(f"Two files map to the same document id: {doc_id}")
        seen.add(doc_id)
        documents.append(document_from_text(doc_id, file.read_text(encoding="utf-8"), source=file.name))
    return documents


def mask_markdown(text: str) -> str:
    """Replace Markdown syntax with spaces so NLP sees plain prose while every character offset is preserved."""
    chars = list(text)

    def blank(start: int, end: int) -> None:
        for i in range(start, end):
            if chars[i] != "\n":
                chars[i] = " "

    for match in re.finditer(r"!?\[([^\]\n]*)\]\(([^)\n]*)\)", text):  # [label](url) -> " label      "
        label_start = match.start(1)
        blank(match.start(), label_start)
        blank(match.end(1), match.end())
    for match in re.finditer(r"(?m)^[ \t]{0,3}(#{1,6}|>|[-*+]|\d{1,3}[.)])(?=[ \t])", text):
        blank(match.start(1), match.end(1))
    for match in re.finditer(r"`+|\*+|(?<!\w)_+|_+(?!\w)", text):
        blank(match.start(), match.end())
    return "".join(chars)


def _trim(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _blocks(text: str) -> list[tuple[int, int]]:
    """Paragraph spans separated by blank lines, trimmed of surrounding whitespace."""
    spans: list[tuple[int, int]] = []
    position = 0
    for match in _BLOCK_SPLIT.finditer(text):
        spans.append(_trim(text, position, match.start()))
        position = match.end()
    spans.append(_trim(text, position, len(text)))
    return [(s, e) for s, e in spans if e > s]


def _is_heading(text: str, span: tuple[int, int]) -> bool:
    block = text[span[0] : span[1]]
    return "\n" not in block and bool(_HEADING.match(block))


def split_sentences(text: str) -> list[tuple[int, int]]:
    """Sentence spans (start, end) over ``text``. Blank lines and heading lines are hard boundaries."""
    spans: list[tuple[int, int]] = []
    for block_start, block_end in _blocks(text):
        block = text[block_start:block_end]
        lines: list[tuple[int, int]] = []
        line_start = 0
        prose_start: int | None = None
        for line in block.split("\n"):
            line_end = line_start + len(line)
            if _HEADING.match(line):
                if prose_start is not None:
                    lines.append((prose_start, line_start - 1))
                    prose_start = None
                lines.append((line_start, line_end))
            elif prose_start is None:
                prose_start = line_start
            line_start = line_end + 1
        if prose_start is not None:
            lines.append((prose_start, len(block)))
        for seg_start, seg_end in lines:
            segment = block[seg_start:seg_end]
            if _HEADING.match(segment):
                s, e = _trim(text, block_start + seg_start, block_start + seg_end)
                if e > s:
                    spans.append((s, e))
                continue
            cursor = 0
            for match in _SENTENCE_END.finditer(segment):
                word = re.search(r"([A-Za-z.]+)[.!?][\"')\]]*$", segment[cursor : match.start() + 1])
                if word and word.group(1).lower().rstrip(".") in _ABBREVIATIONS:
                    continue
                s, e = _trim(text, block_start + seg_start + cursor, block_start + seg_start + match.end())
                if e > s:
                    spans.append((s, e))
                cursor = match.end()
            s, e = _trim(text, block_start + seg_start + cursor, block_start + seg_end)
            if e > s:
                spans.append((s, e))
    return spans


def _split_long(text: str, start: int, end: int, max_chars: int) -> list[tuple[int, int]]:
    """Pack the sentences of ``text[start:end]`` into windows of at most ``max_chars`` (hard-split if needed)."""
    pieces: list[tuple[int, int]] = []
    for s, e in split_sentences(text[start:end]):
        s, e = s + start, e + start
        while e - s > max_chars:  # a single overlong sentence: cut at the last space before the limit
            cut = text.rfind(" ", s + 1, s + max_chars)
            cut = cut if cut > s else s + max_chars
            pieces.append(_trim(text, s, cut))
            s = _trim(text, cut, e)[0]
        if e > s:
            pieces.append((s, e))
    windows: list[tuple[int, int]] = []
    for s, e in pieces:
        if windows and e - windows[-1][0] <= max_chars:
            windows[-1] = (windows[-1][0], e)
        else:
            windows.append((s, e))
    return windows


def chunk_document(document: Document, max_chars: int = 800) -> list[Chunk]:
    """Split a document into paragraph chunks; headings attach to the next paragraph."""
    if max_chars < 50:
        raise ValueError("max_chars must be at least 50")
    text = document.text
    spans: list[tuple[int, int]] = []
    heading_start: int | None = None
    for block in _blocks(text):
        if _is_heading(text, block):
            if heading_start is None:
                heading_start = block[0]
            continue
        start = heading_start if heading_start is not None else block[0]
        heading_start = None
        if block[1] - start <= max_chars:
            spans.append((start, block[1]))
            continue
        windows = _split_long(text, block[0], block[1], max_chars)
        if start < block[0] and windows and windows[0][1] - start <= max_chars:
            windows[0] = (start, windows[0][1])
        elif start < block[0]:
            spans.append(_trim(text, start, block[0]))
        spans.extend(windows)
    if heading_start is not None:  # the document ends with a heading
        spans.append(_trim(text, heading_start, len(text)))
    return [
        Chunk(
            id=f"{document.id}#{i}",
            doc_id=document.id,
            index=i,
            start=s,
            end=e,
            text=text[s:e],
            title=document.title,
        )
        for i, (s, e) in enumerate(spans)
    ]


def chunk_documents(documents: Iterable[Document], max_chars: int = 800) -> list[Chunk]:
    """Chunk every document, preserving document order."""
    return [chunk for document in documents for chunk in chunk_document(document, max_chars)]


def clean_for_display(text: str, *, drop_headings: bool = False) -> str:
    """Markdown masked and whitespace collapsed: what a reader should see in a citation."""
    if drop_headings:
        text = "\n".join(line for line in text.split("\n") if not _HEADING.match(line))
    return re.sub(r"\s+", " ", mask_markdown(text)).strip()
