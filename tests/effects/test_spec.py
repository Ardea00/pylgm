from dataclasses import FrozenInstanceError

import pytest

from pylgm import Fixed, IID, RW1
from pylgm.effects import RW2
from pylgm.parameters import Hyperparameter


def test_effects_compose_in_declaration_order():
    predictor = Fixed("1 + x") + IID("region", "region", 2.0) + RW1(
        "trend", "time", 3.0
    )

    assert [effect.name for effect in predictor.effects] == [
        "fixed",
        "region",
        "trend",
    ]


def test_predictor_rejects_duplicate_effect_names():
    with pytest.raises(ValueError, match="unique"):
        Fixed("1") + IID("fixed", "region")


def test_effect_declarations_are_immutable():
    effect = RW2("trend", "time", Hyperparameter("trend_precision", 3.0))

    with pytest.raises(FrozenInstanceError):
        effect.index = "other"  # type: ignore[misc]
