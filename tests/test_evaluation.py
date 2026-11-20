from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from graphrag_pipeline.data import corpus_dir, heldout_dir
from graphrag_pipeline.embeddings import HashingEmbedder
from graphrag_pipeline.evaluation.bench import VARIANTS, to_markdown
from graphrag_pipeline.evaluation.extraction import evaluate_extraction, load_gold, prf
from graphrag_pipeline.evaluation.retrieval import full_support, load_questions, run_retrieval_benchmark, support_recall
from graphrag_pipeline.ingest import chunk_documents, load_documents
from graphrag_pipeline.pipeline import GraphIndex


def test_support_metrics_on_a_worked_example() -> None:
    support = [["a", "b"], ["c"], ["d"]]
    ranked = ["x", "b", "c", "y", "d"]
    assert support_recall(ranked, support, 3) == 2 / 3
    assert not full_support(ranked, support, 3)
    assert full_support(ranked, support, 5) and support_recall(ranked, support, 5) == 1.0
    assert support_recall(ranked, [], 5) == 0.0 and not full_support(ranked, [], 5)


ids = st.text(alphabet="abcdefgh", min_size=1, max_size=2)


@given(st.lists(ids, unique=True, max_size=10), st.lists(st.lists(ids, min_size=1, max_size=3), max_size=4),
       st.integers(min_value=1, max_value=10))
def test_support_metrics_are_bounded_and_monotone(ranked: list[str], support: list[list[str]], k: int) -> None:
    recall = support_recall(ranked, support, k)
    assert 0.0 <= recall <= 1.0
    assert recall <= support_recall(ranked, support, k + 1)
    assert full_support(ranked, support, k) == (bool(support) and recall == 1.0)


def test_benchmark_questions_are_consistent_with_the_corpus() -> None:
    chunks = {c.id: c for c in chunk_documents(load_documents(corpus_dir()))}
    questions = load_questions()
    assert len(questions) >= 40 and len({q.id for q in questions}) == len(questions)
    assert sum(q.split == "dev" for q in questions) == 10
    assert {q.hops for q in questions} == {2, 3}
    for q in questions:
        assert q.hops == len(q.support)
        flat = [c for hop in q.support for c in hop]
        assert all(c in chunks for c in flat) and len(flat) == len(set(flat))
        last = " ".join(chunks[c].text for c in q.support[-1])
        assert q.answer in last or q.answer.split()[-1] in last


def test_gold_sets_refer_to_real_documents() -> None:
    corpus = {d.id for d in load_documents(corpus_dir())}
    heldout = {d.id for d in load_documents(heldout_dir())}
    development = load_gold()
    held = load_gold("gold_extraction_heldout.json")
    assert set(development) <= corpus and len(development) == 14
    assert set(held) == heldout and not heldout & corpus
    for gold in (*development.values(), *held.values()):
        names = {e.name for e in gold.entities}
        assert all(s in names and t in names for s, _, t in gold.relations)


def test_extraction_scores_on_the_development_set(sample_index: GraphIndex) -> None:
    result = evaluate_extraction(sample_index.kg, load_gold())
    assert result["documents"] == 14
    for key in ("entities", "entities_typed", "relations_typed", "relations_untyped"):
        assert 0.0 <= result[key]["precision"] <= 1.0 and 0.0 <= result[key]["recall"] <= 1.0
    assert result["relations_untyped"]["recall"] >= result["relations_typed"]["recall"]
    assert prf(0, 0, 0) == {"precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "fp": 0, "fn": 0}


def test_retrieval_benchmark_and_markdown_report(sample_index: GraphIndex) -> None:
    questions = load_questions()[:4]
    variants = {k: VARIANTS[k] for k in ("vector", "graph")}
    result = run_retrieval_benchmark(sample_index.kg, questions, HashingEmbedder(), variants=variants, ks=(3, 5))
    assert result["methods"] == ["vector", "graph"]
    assert result["summary"]["all"]["graph"]["questions"] == 4
    row = result["per_question"][0]
    assert len(row["graph"]["ranked"]) <= 5 and "recall@3" in row["vector"]
    extraction = evaluate_extraction(sample_index.kg, load_gold())
    data = {
        "date": "2026-01-01", "command": "graphrag bench", "python": "3.x",
        "corpus": {"documents": 54, "chunks": 109, "entities": 59, "typed_relations": 1,
                   "cooccurrence_relations": 1, "extractor": "spacy-rules"},
        "variants": {"vector": "v", "graph": "g"},
        "retrieval": {**result, "summary": {**result["summary"], "test": result["summary"]["all"]}},
        "extraction": {"development": extraction},
    }
    markdown = to_markdown(data)
    assert "| graph |" in markdown and "Reproduce: `graphrag bench`" in markdown
