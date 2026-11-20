from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from graphrag_pipeline.embeddings import HashingEmbedder, get_embedder
from graphrag_pipeline.pipeline import GraphIndex
from graphrag_pipeline.search.vector import VectorIndex, chunk_embedding_text
from graphrag_pipeline.viz import render_benchmark_chart, render_graph_png


def test_hashing_embedder_is_deterministic_normalised_and_meaningful() -> None:
    embedder = HashingEmbedder(dim=256)
    a = embedder.embed_documents(["battery cells for trucks", "battery cell for a truck", "a poem about the sea"])
    assert a.shape == (3, 256)
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0)
    assert np.array_equal(a, HashingEmbedder(dim=256).embed_documents(["battery cells for trucks",
                                                                     "battery cell for a truck",
                                                                     "a poem about the sea"]))
    assert a[0] @ a[1] > a[0] @ a[2]
    assert embedder.embed_documents([]).shape == (0, 256)


def test_get_embedder_options() -> None:
    assert get_embedder("hashing").name.startswith("hashing")
    assert get_embedder("auto").name  # bge when cached locally, otherwise the hashing stand-in
    with pytest.raises(ValueError):
        get_embedder("word2vec")


def test_vector_index_uses_contextual_titles(sample_index: GraphIndex) -> None:
    chunk = sample_index.kg.chunks["mara-okafor#1"]
    assert chunk_embedding_text(chunk).startswith("Mara Okafor. Okafor is")
    index = VectorIndex(list(sample_index.kg.chunks.values()), HashingEmbedder())
    top = index.search("Mara Okafor chief executive of Velltrix Systems", k=3)
    assert len(top) == 3 and "mara-okafor#1" in [c for c, _ in top]


@pytest.mark.models
@pytest.mark.skipif(not os.getenv("FASTEMBED_CACHE_PATH"), reason="bge-small model cache not configured")
def test_bge_small_embeddings() -> None:
    embedder = get_embedder("bge")
    vectors = embedder.embed_documents(["battery manufacturer", "lidar sensor"])
    assert vectors.shape == (2, 384)
    query = embedder.embed_query("who makes batteries?")
    assert vectors[0] @ query > vectors[1] @ query


def test_figures_are_written(sample_index: GraphIndex, tmp_path: Path) -> None:
    graph_png = render_graph_png(sample_index, tmp_path / "graph.png", max_labels=10)
    assert graph_png.stat().st_size > 10_000
    summary = {
        m: {"questions": 3, **{f"full@{k}": 0.5 for k in (3, 5, 10)}, **{f"recall@{k}": 0.6 for k in (3, 5, 10)}}
        for m in ("vector", "graph-ppr", "graph-unmasked", "graph", "hybrid")
    }
    chart = render_benchmark_chart(
        {"retrieval": {"embedder": "hashing-512", "ks": [3, 5, 10], "summary": {"test": summary}}},
        tmp_path / "chart.png",
    )
    assert chart.stat().st_size > 10_000
