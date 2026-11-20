from __future__ import annotations

import json

import pytest
from hypothesis import given
from hypothesis import strategies as st

from graphrag_pipeline.embeddings import HashingEmbedder
from graphrag_pipeline.pipeline import GraphIndex
from graphrag_pipeline.search import (
    EdgeRelevance,
    LocalSearchConfig,
    LocalSearcher,
    find_paths,
    global_search,
    link_entities,
    path_search,
    reciprocal_rank_fusion,
)
from graphrag_pipeline.search.lexical import lexical_scores, terms

from .test_llm import ScriptedLLM

THREE_HOP = "In which province is the company that makes the Halcyon Sensor headquartered?"
GOLD_CHAIN = {"halcyon-sensor#0", "velltrix-systems#0", "quorvale#0"}


def test_linking_exact_possessive_fuzzy_and_stop_words(sample_index: GraphIndex) -> None:
    kg = sample_index.kg
    linked = link_entities(kg, "Who leads the bank that financed Orrin Freight's trucks?")
    assert [e.entity_id for e in linked] == ["orrin-freight"]
    assert link_entities(kg, "What does QIT teach?")[0].name == "Quorvale Institute of Technology"
    fuzzy = link_entities(kg, "Who makes the Halcyon Sensr?")
    assert fuzzy[0].entity_id == "halcyon-sensor" and fuzzy[0].score < 1.0
    assert link_entities(kg, "who is the company in the city of a product") == []


def test_graph_search_follows_a_three_hop_chain(sample_index: GraphIndex, hashing: HashingEmbedder) -> None:
    result = LocalSearcher(sample_index.kg, hashing).search(THREE_HOP, k=5, retriever="graph")
    assert GOLD_CHAIN <= {h.chunk_id for h in result.hits}
    kept = {h.chunk_id for h in result.hits}
    assert result.traversal and all(edge.chunk_id in kept for edge in result.traversal)
    assert result.answer is None  # offline: no answer synthesis


def test_graph_search_without_an_embedder_uses_lexical_relevance(sample_index: GraphIndex) -> None:
    searcher = LocalSearcher(sample_index.kg)
    hits = searcher.search(THREE_HOP, k=6, retriever="graph").hits
    assert GOLD_CHAIN <= {h.chunk_id for h in hits}
    with pytest.raises(ValueError):
        searcher.search(THREE_HOP, retriever="hybrid")


def test_structural_and_vector_retrievers_run(sample_index: GraphIndex, hashing: HashingEmbedder) -> None:
    for config in (LocalSearchConfig(expansion="ppr"), LocalSearchConfig(expansion="khop")):
        hits = LocalSearcher(sample_index.kg, hashing, config).search(THREE_HOP, k=10, retriever="graph").hits
        assert "halcyon-sensor#0" in {h.chunk_id for h in hits}
    searcher = LocalSearcher(sample_index.kg, hashing)
    vector = searcher.search(THREE_HOP, k=5, retriever="vector")
    hybrid = searcher.search(THREE_HOP, k=5, retriever="hybrid")
    assert len(vector.hits) == 5 and len(hybrid.hits) == 5
    scores = [h.score for h in hybrid.hits]
    assert scores == sorted(scores, reverse=True)


def test_local_search_answers_with_an_llm_when_given_one(sample_index: GraphIndex) -> None:
    llm = ScriptedLLM(["Ashlenmark [quorvale#0]"])
    result = LocalSearcher(sample_index.kg).search(THREE_HOP, k=6, retriever="graph", llm=llm)
    assert result.answer == "Ashlenmark [quorvale#0]"
    assert "quorvale#0" in llm.requests[0][-1]["content"]


def test_edge_relevance_masks_entity_names(sample_index: GraphIndex) -> None:
    relevance = EdgeRelevance(sample_index.kg)
    masked = relevance.prepare("Velltrix Systems is headquartered in Quorvale.")
    assert "Velltrix" not in masked and "Quorvale" not in masked and "headquartered" in masked
    scores = relevance.scores("Where is X headquartered?")
    assert min(scores.values()) == 0.0 and max(scores.values()) == 1.0
    unmasked = EdgeRelevance(sample_index.kg, mask_entities=False)
    assert unmasked.prepare("Velltrix Systems") == "Velltrix Systems"


@given(st.lists(st.text(alphabet="abcdef", min_size=1, max_size=3), unique=True, max_size=12))
def test_rrf_of_identical_rankings_preserves_order(items: list[str]) -> None:
    fused = reciprocal_rank_fusion([items, items])
    assert [item for item, _ in fused] == items


def test_rrf_rewards_agreement_and_checks_weights() -> None:
    fused = dict(reciprocal_rank_fusion([["a", "b", "c"], ["c", "b", "a"]], weights=[1, 3]))
    assert fused["c"] > fused["b"] > fused["a"]
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["a"]], weights=[1, 2])


def test_path_search_explains_each_hop(sample_index: GraphIndex) -> None:
    result = path_search(sample_index.kg, "How is the Stackbot connected to Velltrix Systems?", max_paths=2)
    assert result.paths and result.source and result.target
    path = result.paths[0]
    assert path.hops[0].source == "stackbot" and path.hops[-1].target == "velltrix-systems"
    assert all(h.evidence and h.chunk_id for h in path.hops)
    costs = [p.cost for p in result.paths]
    assert costs == sorted(costs)
    explicit = path_search(sample_index.kg, "", source="Mara Okafor", target="Calder Optics")
    assert explicit.paths and len(explicit.paths[0].hops) <= 3
    missing = path_search(sample_index.kg, "How is Atlantis connected to Velltrix Systems?")
    assert not missing.paths and missing.message
    assert find_paths(sample_index.kg, "stackbot", "stackbot") == []


def test_global_search_offline_extracts_cited_points(sample_index: GraphIndex, hashing: HashingEmbedder) -> None:
    question = "Which companies make batteries?"
    for embedder in (hashing, None):
        result = global_search(sample_index.reports, question, embedder=embedder, max_points=4)
        assert result.level == 0 and result.method == "extractive" and result.answer is None
        assert result.reports and 0 < len(result.points) <= 4
        assert all(point.chunk_ids for point in result.points)
    assert any("Brightwell" in p.text for p in result.points)
    assert global_search([], question).points == []


def test_global_search_map_reduce_with_an_llm(sample_index: GraphIndex) -> None:
    mapped = json.dumps({"points": [{"text": "Brightwell Cells makes the Ember-9 cell.", "score": 90}]})
    llm = ScriptedLLM([mapped, json.dumps({"points": []}), json.dumps({"points": []}),
                       json.dumps({"points": []}), "Brightwell Cells (L0-C1)."])
    result = global_search(sample_index.reports, "Who makes battery cells?", llm=llm, top_reports=2)
    assert result.method == "llm-map-reduce" and result.answer == "Brightwell Cells (L0-C1)."
    assert result.points[0].score == pytest.approx(0.9)


def test_lexical_scores_prefer_overlap() -> None:
    assert terms("Studied making cells") == ["studi", "making", "cell"]
    scores = lexical_scores("where is it headquartered", ["X is headquartered in X", "X makes sensors"])
    assert scores[0] > scores[1] == 0.0
