import numpy as np
import pytest

from pylgm.evaluation import (
    link_auc,
    precision_at_k,
    reconstruction_scores,
    weighted_cosine,
    weighted_jaccard,
)
from pylgm.exceptions import DataContractError


def test_auc_perfect_ties_and_brute_force():
    assert link_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert link_auc([0, 1, 0, 1], [0.5] * 4) == 0.5
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 60)
    s = np.round(rng.random(60), 1)  # rounding forces ties
    pos, neg = s[y == 1], s[y == 0]
    brute = ((pos[:, None] > neg).sum() + 0.5 * (pos[:, None] == neg).sum()) / (
        pos.size * neg.size
    )
    assert link_auc(y, s) == pytest.approx(brute)


def test_precision_at_k():
    assert precision_at_k([1, 0, 1, 0], [0.9, 0.8, 0.7, 0.1], 2) == 0.5
    assert precision_at_k([0, 1, 1], [0.5, 0.5, 0.5], 1) == 0.0  # stable ties: first edge wins


def test_cosine_and_jaccard():
    assert weighted_cosine([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0)
    assert weighted_jaccard([1, 2, 0], [2, 1, 1]) == pytest.approx(2 / 5)


def test_reconstruction_scores():
    out = reconstruction_scores([0, 2, 4, 0], [0.1, 0.9, 0.8, 0.2], [0, 2, 4, 0])
    assert out == {
        "auc": 1.0,
        "precision_at_k": 1.0,
        "weighted_cosine": pytest.approx(1.0),
        "weighted_jaccard": 1.0,
    }


def test_validation_errors():
    with pytest.raises(DataContractError):
        link_auc([1, 1], [0.1, 0.2])
    with pytest.raises(DataContractError):
        link_auc([0, 1, 1], [0.1, 0.2])
    with pytest.raises(DataContractError):
        link_auc([0, 1], [0.1, np.nan])
    with pytest.raises(DataContractError):
        weighted_jaccard([1, -1], [1, 1])
    with pytest.raises(DataContractError):
        weighted_jaccard([0, 0], [0, 0])
    with pytest.raises(DataContractError):
        precision_at_k([0, 1], [0.1, 0.2], 3)
