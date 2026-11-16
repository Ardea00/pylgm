"""YAML ``weighted`` / ``replicated`` / ``copy`` == the Python declaration."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pylgm import (
    AR1Structure, Copy, Fixed, Gaussian, Grouped, IID, LGM, Replicated, Weighted,
)
from pylgm.compiler import _effect_block
from pylgm.config import load_model
from pylgm.config.schema import EffectConfig
from pylgm.exceptions import ConfigurationError


def _frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = [
        {"t": t, "region": r, "firm": f, "j": (r + 1) % 3, "z": rng.normal(),
         "y": rng.normal()}
        for t in range(3) for r in range(3) for f in ("f1", "f2")
    ]
    frame = pd.DataFrame(rows)
    frame["k"] = frame["t"] * 100 + frame["region"] * 10 + (frame["firm"] == "f2")
    return frame


def _doc(effects: str) -> str:
    return (
        "response: y\nlikelihood: {family: gaussian, sigma: 0.5}\n"
        "data: {time: k}\npredictor:\n  fixed: '1'\n  effects:\n" + effects
    )


_u = "{type: iid, index: region, precision: 2.0}"
CASES = {
    "weighted": (
        f"    - {{name: u, type: weighted, by: z, effect: {_u}}}",
        Weighted(IID("u", "region", 2.0), "z"),
    ),
    "replicated": (
        f"    - {{name: u, type: replicated, over: firm, effect: {_u}}}",
        Replicated(IID("u", "region", 2.0), "firm"),
    ),
    "replicated_grouped": (
        "    - name: u\n      type: replicated\n      over: firm\n      effect:\n"
        "        type: grouped\n        over: t\n        structure: {type: ar1, rho: 0.6}\n"
        f"        effect: {_u}",
        Replicated(Grouped(IID("u", "region", 2.0), "t", AR1Structure(0.6)), "firm"),
    ),
    "weighted_replicated": (
        "    - name: u\n      type: weighted\n      by: z\n      effect:\n"
        f"        type: replicated\n        over: firm\n        effect: {_u}",
        Weighted(Replicated(IID("u", "region", 2.0), "firm"), "z"),
    ),
}


def _check(loaded: LGM, expected: LGM, block_index: int | None) -> None:
    assert loaded == expected
    frame = _frame()
    if block_index is not None:  # design / precision / labels of the compiled block
        a, b = (_effect_block(m.predictor.effects[block_index], frame) for m in (loaded, expected))
        assert a.labels == b.labels
        assert (a.precision != b.precision).nnz == 0
        assert np.array_equal(np.asarray(a.design.todense()), np.asarray(b.design.todense()))
    ra, rb = loaded.fit(frame), expected.fit(frame)
    assert ra.log_marginal_likelihood == rb.log_marginal_likelihood
    assert ra.labels == rb.labels


@pytest.mark.parametrize("case", CASES)
def test_yaml_modifier_equals_python(tmp_path: Path, case: str) -> None:
    effects, python = CASES[case]
    path = tmp_path / "m.yaml"
    path.write_text(_doc(effects))
    expected = LGM(response="y", likelihood=Gaussian(0.5), time="k",
                   predictor=Fixed("1") + python)
    _check(load_model(path), expected, 1)


def test_yaml_copy_equals_python(tmp_path: Path) -> None:
    path = tmp_path / "m.yaml"
    path.write_text(_doc(
        f"    - {{name: u, type: iid, index: region, precision: 2.0}}\n"
        "    - {name: u, type: copy, index: j, scale: 0.7}"
    ))
    expected = LGM(response="y", likelihood=Gaussian(0.5), time="k",
                   predictor=Fixed("1") + IID("u", "region", 2.0) + Copy("u", "j", 0.7))
    _check(load_model(path), expected, None)


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        ({"name": "w", "type": "weighted", "effect": {"type": "iid", "index": "r"}},
         "requires by, effect"),
        ({"name": "w", "type": "weighted", "by": "z", "over": "f",
          "effect": {"type": "iid", "index": "r"}}, "not valid for effect type"),
        ({"name": "r", "type": "replicated", "over": "f"}, "requires over, effect"),
        ({"name": "r", "type": "replicated", "over": "f", "structure": {"type": "iid"},
          "effect": {"type": "iid", "index": "r"}}, "not valid for effect type"),
        ({"name": "r", "type": "replicated", "over": "f",
          "effect": {"name": "x", "type": "iid", "index": "r"}}, "must not set its own name"),
        ({"name": "c", "type": "copy"}, "index"),
        ({"name": "c", "type": "copy", "index": "j", "precision": 2.0},
         "not valid for effect type"),
        ({"name": "c", "type": "copy", "index": "j", "scale": True}, "must be a number"),
        ({"name": "u", "type": "rw1", "index": "r", "scale": 2.0}, "true or false"),
    ],
)
def test_modifier_schema_rejections(payload: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        EffectConfig.model_validate(payload)


def test_python_constructors_reject_bad_nesting(tmp_path: Path) -> None:
    path = tmp_path / "m.yaml"
    path.write_text(_doc(
        "    - {name: c, type: weighted, by: z, effect: {type: copy, index: j}}"
    ))
    with pytest.raises(ConfigurationError, match="Weighted cannot wrap a Copy"):
        load_model(path)
