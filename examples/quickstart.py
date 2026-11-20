"""Build the bundled fictional corpus offline and answer two questions with the graph.

    python examples/quickstart.py
"""

from __future__ import annotations

from graphrag_pipeline import BuildConfig, LocalSearcher, build_index, corpus_dir, path_search

index = build_index(corpus_dir(), BuildConfig(extractor="spacy"))
print(index.kg.stats()["entities"], "entities,", index.kg.stats()["typed_relations"], "typed relations")

# Local search: no embedding model here, so relation evidence is scored lexically.
searcher = LocalSearcher(index.kg)
result = searcher.search("Who founded the manufacturer of the Wayfarer truck?", k=3, retriever="graph")
for edge in result.traversal:
    left, right = (edge.source_name, edge.target_name) if edge.forward else (edge.target_name, edge.source_name)
    print(f"{'  ' * edge.depth}{left} -[{edge.relation}]-> {right}   ({edge.chunk_id})")
for hit in result.hits:
    print(f"[{hit.chunk_id}] {hit.text[:90]}")

# Path search: how are two entities connected?
paths = path_search(index.kg, "How is the Stackbot connected to Velltrix Systems?").paths
print(" -> ".join([paths[0].hops[0].source_name, *(h.target_name for h in paths[0].hops)]))
