"""Reciprocal rank fusion (RRF): combine rankings without calibrating their scores against each other.

``score(d) = sum over rankings of weight / (k + rank(d))``, with ranks starting at 1. Graph scores (PageRank mass)
and cosine similarities live on unrelated scales, so fusing ranks is safer than adding scores, and it has a single
well-known constant (k = 60, from Cormack et al., 2009).
"""

from __future__ import annotations

from collections.abc import Sequence


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]], *, k: int = 60, weights: Sequence[float] | None = None
) -> list[tuple[str, float]]:
    """Fuse ranked id lists; ties break by first appearance across the input rankings."""
    weights = list(weights) if weights is not None else [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("one weight per ranking is required")
    scores: dict[str, float] = {}
    first_seen: dict[str, int] = {}
    for ranking, weight in zip(rankings, weights, strict=True):
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + weight / (k + rank)
            first_seen.setdefault(item, len(first_seen))
    return sorted(scores.items(), key=lambda kv: (-kv[1], first_seen[kv[0]]))
