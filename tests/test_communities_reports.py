from __future__ import annotations

import json

from graphrag_pipeline.communities import at_level, detect_communities, levels, modularity
from graphrag_pipeline.pipeline import GraphIndex
from graphrag_pipeline.reports import ExtractiveReporter, LLMReporter, build_reports, join_names, pagerank

from .test_llm import ScriptedLLM


def test_each_level_partitions_every_entity(sample_index: GraphIndex) -> None:
    nodes = set(sample_index.kg.graph.nodes)
    assert levels(sample_index.communities) == [0, 1, 2]
    for level in (0, 1, 2):
        members = [m for c in at_level(sample_index.communities, level) for m in c.members]
        assert sorted(members) == sorted(nodes)  # every entity exactly once
        assert 0.0 < modularity(sample_index.kg, sample_index.communities, level) <= 1.0
    coarse = len(at_level(sample_index.communities, 0))
    fine = len(at_level(sample_index.communities, 2))
    assert coarse <= fine


def test_parents_point_to_the_previous_level(sample_index: GraphIndex) -> None:
    ids = {c.id: c for c in sample_index.communities}
    for community in sample_index.communities:
        if community.level == 0:
            assert community.parent is None
        else:
            assert ids[community.parent].level == community.level - 1  # type: ignore[index]


def test_detection_is_deterministic(sample_index: GraphIndex) -> None:
    again = detect_communities(sample_index.kg)
    assert [c.members for c in again] == [c.members for c in sample_index.communities]


def test_extractive_reports_quote_the_corpus(sample_index: GraphIndex) -> None:
    kg = sample_index.kg
    assert sample_index.reports and all(r.method == "extractive" for r in sample_index.reports)
    for report in sample_index.reports:
        community = sample_index.community(report.community_id)
        assert community is not None and len(community.members) >= 2
        assert report.top_entities[0] in report.title
        for finding in report.findings:
            assert finding.chunk_ids
            chunk_text = " ".join(kg.chunks[finding.chunk_ids[0]].text.split())
            sentence = finding.text.split(": ", 1)[-1]
            assert sentence[:40] in chunk_text  # copied, not generated
    importance = [r.importance for r in sample_index.reports if r.level == 0]
    assert importance == sorted(importance, reverse=True)
    assert abs(sum(pagerank(kg).values()) - 1.0) < 1e-6


def test_llm_reporter_keeps_only_known_citations_and_falls_back(sample_index: GraphIndex) -> None:
    kg = sample_index.kg
    community = max(sample_index.communities, key=lambda c: len(c.members))
    ranks = pagerank(kg)
    material = ExtractiveReporter().material(kg, community, ranks)
    real_chunk = material[2][0].chunk_ids[0]
    payload = {
        "title": "Sensor makers",
        "summary": "A cluster of companies.",
        "findings": [{"text": "They build sensors.", "chunk_ids": [real_chunk, "made-up#9"]}],
    }
    report = LLMReporter(ScriptedLLM([json.dumps(payload)])).report(kg, community, ranks)
    assert report.title == "Sensor makers" and report.method == "llm"
    assert report.findings[0].chunk_ids == [real_chunk]
    fallback = LLMReporter(ScriptedLLM([])).report(kg, community, ranks)
    assert fallback.method == "extractive"


def test_build_reports_respects_min_size(sample_index: GraphIndex) -> None:
    reports = build_reports(sample_index.kg, sample_index.communities, min_size=5)
    sizes = {c.id: len(c.members) for c in sample_index.communities}
    assert reports and all(sizes[r.community_id] >= 5 for r in reports)
    assert join_names(["A"]) == "A" and join_names(["A", "B", "C"]) == "A, B and C"
