"""Text embeddings for the vector baseline and for hybrid retrieval.

* ``FastEmbedEmbedder``: ``BAAI/bge-small-en-v1.5`` (384 dimensions) through fastembed/ONNX Runtime, on CPU.
  Queries get the model's recommended retrieval instruction prefix. The cache folder honours
  ``FASTEMBED_CACHE_PATH``.
* ``HashingEmbedder``: a deterministic, dependency-free stand-in (signed feature hashing of words, word bigrams and
  character n-grams, L2-normalised). It needs no download, so tests and CI use it.
"""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Sequence
from itertools import pairwise
from typing import Any, Protocol

import numpy as np

BGE_SMALL = "BAAI/bge-small-en-v1.5"
_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
_WORD = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")


class Embedder(Protocol):
    name: str

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


def _normalise_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class HashingEmbedder:
    """Signed feature hashing (words, word bigrams, character 3-5-grams); deterministic across processes."""

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim
        self.name = f"hashing-{dim}"

    def _features(self, text: str) -> list[tuple[str, float]]:
        words = _WORD.findall(text.lower())
        features = [(f"w:{w}", 1.0) for w in words]
        features += [(f"b:{a} {b}", 0.7) for a, b in pairwise(words)]
        for word in words:
            padded = f"<{word}>"
            for n in (3, 4, 5):
                features += [(f"c:{padded[i : i + n]}", 0.3) for i in range(max(1, len(padded) - n + 1))]
        return features

    def _vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for feature, weight in self._features(text):
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "little")
            vec[value % self.dim] += weight if (value >> 63) & 1 else -weight
        return vec

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return _normalise_rows(np.stack([self._vector(t) for t in texts]))

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]


class FastEmbedEmbedder:
    """bge-small-en-v1.5 via fastembed (downloads ~70 MB once unless ``local_files_only``)."""

    def __init__(self, model: str = BGE_SMALL, *, local_files_only: bool = False) -> None:
        from fastembed import TextEmbedding

        kwargs: dict[str, Any] = {"model_name": model, "local_files_only": local_files_only}
        cache = os.getenv("FASTEMBED_CACHE_PATH")
        if cache:
            kwargs["cache_dir"] = cache
        self._model = TextEmbedding(**kwargs)
        self.name = model
        self._prefix = _BGE_QUERY_PREFIX if "bge" in model.lower() else ""

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)
        return _normalise_rows(np.asarray(list(self._model.embed(list(texts))), dtype=np.float32))

    def embed_query(self, text: str) -> np.ndarray:
        vector = np.asarray(next(iter(self._model.embed([self._prefix + text]))), dtype=np.float32)
        return vector / (np.linalg.norm(vector) or 1.0)


def get_embedder(kind: str = "auto") -> Embedder:
    """``bge`` (download allowed), ``hashing``, or ``auto``: bge if fastembed and a cached model exist, else hashing."""
    if kind == "hashing":
        return HashingEmbedder()
    if kind == "bge":
        return FastEmbedEmbedder()
    if kind != "auto":
        raise ValueError(f"Unknown embedder '{kind}' (expected auto, bge or hashing)")
    try:
        return FastEmbedEmbedder(local_files_only=True)
    except Exception:  # fastembed missing, model not cached, or ONNX runtime unavailable
        return HashingEmbedder()
