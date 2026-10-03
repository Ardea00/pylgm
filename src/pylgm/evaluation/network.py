"""Scores for a reconstructed weighted network against the true one.

Every function takes aligned 1-D arrays over the same candidate edges.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

from pylgm.exceptions import DataContractError


def _aligned(*arrays) -> list[np.ndarray]:
    out = []
    for array in arrays:
        try:
            values = np.asarray(array, dtype=float)
        except (TypeError, ValueError) as cause:
            raise DataContractError("network arrays must be numeric") from cause
        if values.ndim != 1 or values.size == 0:
            raise DataContractError("network arrays must be non-empty and one-dimensional")
        if not np.isfinite(values).all():
            raise DataContractError("network arrays must be finite")
        out.append(values)
    if len({values.size for values in out}) != 1:
        raise DataContractError("network arrays must have the same length")
    return out


def link_auc(truth, score) -> float:
    """ROC AUC of ``score`` against binary ``truth`` (Mann-Whitney, average ranks for ties)."""
    truth, score = _aligned(truth, score)
    positive = truth > 0
    n_pos = int(positive.sum())
    n_neg = positive.size - n_pos
    if n_pos == 0 or n_neg == 0:
        raise DataContractError("link_auc needs both existing and absent links in truth")
    ranks = rankdata(score)
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def precision_at_k(truth, score, k: int) -> float:
    """Fraction of true links among the ``k`` highest scores.

    Ties are broken by a stable sort, i.e. the earlier edge ranks first.
    """
    truth, score = _aligned(truth, score)
    if not 1 <= k <= truth.size:
        raise DataContractError("k must lie in [1, number of candidate edges]")
    top = np.argsort(-score, kind="stable")[:k]
    return float((truth[top] > 0).mean())


def weighted_cosine(true_weights, predicted_weights) -> float:
    """Cosine similarity of the two weight vectors."""
    t, p = _aligned(true_weights, predicted_weights)
    norm = np.linalg.norm(t) * np.linalg.norm(p)
    if norm == 0:
        raise DataContractError("weighted_cosine is undefined for a zero weight vector")
    return float(t @ p / norm)


def weighted_jaccard(true_weights, predicted_weights) -> float:
    """``sum(min) / sum(max)`` for nonnegative weights."""
    t, p = _aligned(true_weights, predicted_weights)
    if (t < 0).any() or (p < 0).any():
        raise DataContractError("weighted_jaccard needs nonnegative weights")
    denominator = np.maximum(t, p).sum()
    if denominator == 0:
        raise DataContractError("weighted_jaccard is undefined when both vectors are zero")
    return float(np.minimum(t, p).sum() / denominator)


def reconstruction_scores(
    truth_weights, link_probability, predicted_weights, k: int | None = None
) -> dict[str, float]:
    """All network scores; true links are ``truth_weights > 0``, ``k`` defaults to their count."""
    truth_weights, link_probability, predicted_weights = _aligned(
        truth_weights, link_probability, predicted_weights
    )
    links = (truth_weights > 0).astype(float)
    return {
        "auc": link_auc(links, link_probability),
        "precision_at_k": precision_at_k(
            links, link_probability, int(links.sum()) if k is None else k
        ),
        "weighted_cosine": weighted_cosine(truth_weights, predicted_weights),
        "weighted_jaccard": weighted_jaccard(truth_weights, predicted_weights),
    }
