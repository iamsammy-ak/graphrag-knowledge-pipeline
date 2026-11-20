"""GraphRAG knowledge pipeline: unstructured text -> knowledge graph -> local, path and global search.

Typical use::

    from graphrag_pipeline import BuildConfig, LocalSearcher, build_index, corpus_dir

    index = build_index(corpus_dir(), BuildConfig(extractor="spacy"))
    result = LocalSearcher(index.kg).search("Who founded the maker of the Wayfarer truck?", retriever="graph")
"""

__version__ = "0.1.0"

from .data import corpus_dir  # noqa: E402
from .embeddings import FastEmbedEmbedder, HashingEmbedder, get_embedder  # noqa: E402
from .graph import KnowledgeGraph  # noqa: E402
from .pipeline import BuildConfig, GraphIndex, build_index  # noqa: E402
from .search import LocalSearchConfig, LocalSearcher, find_paths, global_search, path_search  # noqa: E402

__all__ = [
    "BuildConfig",
    "FastEmbedEmbedder",
    "GraphIndex",
    "HashingEmbedder",
    "KnowledgeGraph",
    "LocalSearchConfig",
    "LocalSearcher",
    "__version__",
    "build_index",
    "corpus_dir",
    "find_paths",
    "get_embedder",
    "global_search",
    "path_search",
]
