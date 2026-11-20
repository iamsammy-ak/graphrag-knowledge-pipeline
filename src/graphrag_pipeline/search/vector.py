"""Dense vector retrieval over chunks: the baseline that graph search is compared against.

Each chunk is embedded with its document title prepended when the chunk does not already start with it (a
"contextual chunk header"), which gives the baseline a fair chance on chunks such as "Okafor is the chief executive
of Velltrix Systems." that never repeat the full name.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..embeddings import Embedder
from ..ingest import clean_for_display
from ..schemas import Chunk


def chunk_embedding_text(chunk: Chunk) -> str:
    text = clean_for_display(chunk.text)
    if chunk.title and not text.startswith(chunk.title):
        return f"{chunk.title}. {text}"
    return text


class VectorIndex:
    """Brute-force cosine search (exact; fine for thousands of chunks)."""

    def __init__(self, chunks: Sequence[Chunk], embedder: Embedder) -> None:
        self.embedder = embedder
        self.ids = [c.id for c in chunks]
        self.matrix = embedder.embed_documents([chunk_embedding_text(c) for c in chunks])

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        if not self.ids:
            return []
        scores = self.matrix @ self.embedder.embed_query(query)
        order = np.argsort(-scores, kind="stable")[:k]
        return [(self.ids[i], float(scores[i])) for i in order]
