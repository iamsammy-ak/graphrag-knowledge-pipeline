# Benchmark results (hashing-512)

Reproduce: `graphrag bench --embedder hashing`  
Run on 2026-11-20 with Python 3.13.2.

## Data

Synthetic, fictional corpus bundled with the package: 54 documents, 109 chunks. Graph built offline by `spacy-rules`: 59 entities, 161 typed relations, 35 co-occurrence links.
42 hand-written multi-hop questions (2-3 hops) with gold supporting chunks per hop. Settings were chosen on the 10 `dev` questions; the 32 `test` questions were not used for tuning.

Metrics: **recall@k** = share of hops with a supporting chunk in the top k; **full@k** = share of questions with every hop covered in the top k.

Variants:

- `vector`: dense top-k over chunks (title-prefixed), cosine similarity
- `graph-ppr`: ablation: structural personalised PageRank from the linked entities (question used only for linking)
- `graph-unmasked`: ablation: relation-aware beam search, but edge relevance computed without masking entity names
- `graph`: graph local search: relation-aware beam search over relation paths (entity names masked)
- `hybrid`: weighted reciprocal rank fusion of graph (weight 3) and vector (weight 1), k = 10

## Retrieval: test questions (32)

| method | recall@3 | recall@5 | recall@10 | full@3 | full@5 | full@10 |
|---|---:|---:|---:|---:|---:|---:|
| vector | 0.38 | 0.44 | 0.54 | 0.06 | 0.06 | 0.16 |
| graph-ppr | 0.38 | 0.52 | 0.77 | 0.00 | 0.12 | 0.53 |
| graph-unmasked | 0.51 | 0.66 | 0.85 | 0.16 | 0.41 | 0.69 |
| graph | 0.64 | 0.74 | 0.89 | 0.41 | 0.59 | 0.78 |
| hybrid | 0.58 | 0.71 | 0.89 | 0.28 | 0.47 | 0.75 |

## Retrieval: all questions (42)

| method | recall@3 | recall@5 | recall@10 | full@3 | full@5 | full@10 |
|---|---:|---:|---:|---:|---:|---:|
| vector | 0.38 | 0.43 | 0.55 | 0.07 | 0.07 | 0.14 |
| graph-ppr | 0.40 | 0.55 | 0.75 | 0.00 | 0.17 | 0.48 |
| graph-unmasked | 0.56 | 0.70 | 0.87 | 0.21 | 0.45 | 0.71 |
| graph | 0.67 | 0.79 | 0.91 | 0.40 | 0.67 | 0.83 |
| hybrid | 0.62 | 0.73 | 0.91 | 0.31 | 0.50 | 0.81 |

### 2-hop questions (30)

| method | recall@3 | recall@5 | recall@10 | full@3 | full@5 | full@10 |
|---|---:|---:|---:|---:|---:|---:|
| vector | 0.45 | 0.48 | 0.60 | 0.10 | 0.10 | 0.20 |
| graph-ppr | 0.43 | 0.60 | 0.83 | 0.00 | 0.23 | 0.67 |
| graph-unmasked | 0.63 | 0.78 | 0.90 | 0.30 | 0.57 | 0.80 |
| graph | 0.77 | 0.90 | 0.97 | 0.57 | 0.83 | 0.93 |
| hybrid | 0.70 | 0.82 | 0.97 | 0.43 | 0.63 | 0.93 |

### 3-hop questions (12)

| method | recall@3 | recall@5 | recall@10 | full@3 | full@5 | full@10 |
|---|---:|---:|---:|---:|---:|---:|
| vector | 0.19 | 0.31 | 0.42 | 0.00 | 0.00 | 0.00 |
| graph-ppr | 0.31 | 0.42 | 0.53 | 0.00 | 0.00 | 0.00 |
| graph-unmasked | 0.36 | 0.50 | 0.78 | 0.00 | 0.17 | 0.50 |
| graph | 0.42 | 0.53 | 0.78 | 0.00 | 0.25 | 0.58 |
| hybrid | 0.42 | 0.53 | 0.78 | 0.00 | 0.17 | 0.50 |

## Extraction quality (offline spaCy extractor)

### Development gold set (14 documents, 70 entities, 56 relations)

The extraction rules were written while reading this corpus, so these numbers are optimistic.

| measure | precision | recall | F1 |
|---|---:|---:|---:|
| entities found | 1.00 | 1.00 | 1.00 |
| entities with the right type | 1.00 | 1.00 | 1.00 |
| relations, typed and directed | 0.96 | 0.84 | 0.90 |
| relations, any type or direction | 1.00 | 0.90 | 0.95 |

Co-occurrence edges: 12, of which 17% link a gold-related pair. Gold pairs connected by any edge (typed or co-occurrence): 92%.
Correct typed relations per pattern: appos 2/2, copula 8/8, passive 14/14, prep 8/8, svo 15/17.

### Held-out gold set (8 documents, 39 entities, 31 relations)

New documents in a different style, written and labelled after the rules were frozen. This is the honest estimate of the extractor on unseen text.

| measure | precision | recall | F1 |
|---|---:|---:|---:|
| entities found | 0.81 | 1.00 | 0.90 |
| entities with the right type | 0.62 | 0.77 | 0.69 |
| relations, typed and directed | 0.67 | 0.52 | 0.58 |
| relations, any type or direction | 0.75 | 0.58 | 0.65 |

Co-occurrence edges: 20, of which 10% link a gold-related pair. Gold pairs connected by any edge (typed or co-occurrence): 65%.
Correct typed relations per pattern: appos 2/3, copula 4/4, passive 3/4, prep 5/9, svo 2/4.

