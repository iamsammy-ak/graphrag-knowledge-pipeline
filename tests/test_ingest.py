from __future__ import annotations

from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from graphrag_pipeline.ingest import (
    chunk_document,
    clean_for_display,
    document_from_text,
    load_documents,
    mask_markdown,
    slugify,
    split_sentences,
)

markdown_text = st.text(
    alphabet=st.sampled_from(list("abcXYZ #*_`[]()-.!?\n 0123")), min_size=0, max_size=400
)


@given(markdown_text)
@settings(max_examples=150, deadline=None)
def test_mask_markdown_preserves_every_offset(text: str) -> None:
    masked = mask_markdown(text)
    assert len(masked) == len(text)
    assert [i for i, c in enumerate(text) if c == "\n"] == [i for i, c in enumerate(masked) if c == "\n"]


def test_mask_markdown_removes_syntax_but_keeps_words() -> None:
    text = "# Title\n\nSee **Velltrix** and [the docs](http://x.y) with `code`."
    masked = mask_markdown(text)
    assert "#" not in masked and "*" not in masked and "http" not in masked and "`" not in masked
    assert "Velltrix" in masked and "the docs" in masked
    assert len(masked) == len(text)


paragraphs = st.lists(
    st.text(alphabet=st.sampled_from(list("abcdefghij .,")), min_size=1, max_size=300), min_size=1, max_size=6
)


@given(paragraphs, st.integers(min_value=60, max_value=400))
@settings(max_examples=100, deadline=None)
def test_chunks_are_exact_ordered_slices_covering_all_text(parts: list[str], max_chars: int) -> None:
    doc = document_from_text("d", "# Heading\n\n" + "\n\n".join(parts))
    chunks = chunk_document(doc, max_chars=max_chars)
    previous_end = 0
    for chunk in chunks:
        assert doc.text[chunk.start : chunk.end] == chunk.text
        assert chunk.start >= previous_end
        previous_end = chunk.end
    covered = {i for c in chunks for i in range(c.start, c.end)}
    assert all(i in covered for i, ch in enumerate(doc.text) if not ch.isspace())


def test_heading_attaches_to_next_paragraph_and_titles_are_set() -> None:
    doc = document_from_text("acme", "# Acme\n\nFirst paragraph.\n\nSecond paragraph.")
    chunks = chunk_document(doc)
    assert [c.id for c in chunks] == ["acme#0", "acme#1"]
    assert chunks[0].text.startswith("# Acme") and chunks[1].text == "Second paragraph."
    assert doc.title == "Acme" and all(c.title == "Acme" for c in chunks)


def test_long_paragraph_is_split_at_sentence_boundaries() -> None:
    sentence = "Velltrix Systems builds sensors for trucks. "
    doc = document_from_text("long", sentence * 30)
    chunks = chunk_document(doc, max_chars=200)
    assert len(chunks) > 1
    assert all(len(c.text) <= 200 for c in chunks)
    assert all(c.text.endswith(".") for c in chunks)


def test_chunk_document_rejects_tiny_limits() -> None:
    with pytest.raises(ValueError):
        chunk_document(document_from_text("x", "text"), max_chars=10)


def test_split_sentences_respects_abbreviations_and_headings() -> None:
    text = "# Title\nDr. Ada Lin joined Acme Inc. in 2020. She left! Did she return?"
    sentences = [text[s:e] for s, e in split_sentences(text)]
    assert sentences == ["# Title", "Dr. Ada Lin joined Acme Inc. in 2020.", "She left!", "Did she return?"]


def test_load_documents_ids_and_errors(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "One Doc.md").write_text("# One\n\nHello.", encoding="utf-8")
    (tmp_path / "two.txt").write_text("Plain text.", encoding="utf-8")
    (tmp_path / "ignore.csv").write_text("a,b", encoding="utf-8")
    docs = load_documents(tmp_path)
    assert [d.id for d in docs] == ["sub-one-doc", "two"]
    assert docs[0].title == "One" and docs[1].title == "two"
    with pytest.raises(FileNotFoundError):
        load_documents(tmp_path / "missing")


def test_clean_for_display_can_drop_headings() -> None:
    assert clean_for_display("# Acme\n\nAcme  makes\n robots.", drop_headings=True) == "Acme makes robots."
    assert slugify("Hello, World / 2") == "hello-world-2"
