from pathlib import Path

import pytest

from pylgm import Fixed, Gaussian, IID, LGM, RW1
from pylgm.config import load_model
from pylgm.exceptions import ConfigurationError


def test_load_model_builds_the_same_declaration_as_python(tmp_path: Path) -> None:
    path = tmp_path / "model.yaml"
    path.write_text(
        """
response: y
likelihood: {family: gaussian, sigma: 0.5}
data: {panel: [region], time: time}
predictor:
  fixed: "1 + x"
  effects:
    - {name: region_effect, type: iid, index: region, precision: 2.0}
    - {name: trend, type: rw1, index: time, precision: 3.0}
""".strip()
    )

    assert load_model(path) == LGM(
        response="y",
        likelihood=Gaussian(0.5),
        predictor=Fixed("1 + x")
        + IID("region_effect", "region", 2.0)
        + RW1("trend", "time", 3.0),
        panel=("region",),
        time="time",
    )


def test_load_model_forbids_unknown_keys(tmp_path: Path) -> None:
    path = tmp_path / "model.yaml"
    path.write_text(
        "response: y\nlikelihood: {family: gaussian, sigma: 1}\npredictor: {fixed: '1'}\nunknown: true"
    )

    with pytest.raises(ConfigurationError, match="unknown"):
        load_model(path)


@pytest.mark.parametrize(
    "document",
    [
        "response: y\nlikelihood: {family: gaussian}\npredictor: {fixed: '1'}",
        "response: y\nlikelihood: {family: gaussian, sigma: 0}\npredictor: {fixed: '1'}",
        "response: y\nlikelihood: {family: gaussian, sigma: .inf}\npredictor: {fixed: '1'}",
        "response: y\nlikelihood: {family: gaussian, sigma: true}\npredictor: {fixed: '1'}",
        "response: y\nlikelihood: {family: gaussian, sigma: 1}\npredictor:\n  fixed: '1'\n  effects: [{name: region, type: iid, index: region}]",
        "response: y\nlikelihood: {family: gaussian, sigma: 1}\npredictor:\n  fixed: '1'\n  effects: [{name: trend, type: rw2, index: time, precision: -1}]",
    ],
)
def test_load_model_requires_finite_positive_explicit_hyperparameters(
    tmp_path: Path, document: str
) -> None:
    path = tmp_path / "model.yaml"
    path.write_text(document)

    with pytest.raises(ConfigurationError, match="sigma|precision|finite|greater than 0|Field required"):
        load_model(path)


@pytest.mark.parametrize(
    "document",
    [
        "response: y\nlikelihood: {family: poisson, sigma: 1}\npredictor: {fixed: '1'}",
        "response: y\nlikelihood: {family: gaussian, sigma: 1}\npredictor:\n  fixed: '1'\n  effects: [{name: trend, type: ar1, index: time, precision: 1}]",
        "response: y\nlikelihood: {family: gaussian, sigma: 1}\npredictor: {fixed: '1', precision: 1}",
    ],
)
def test_load_model_allows_only_the_supported_yaml_vocabulary(
    tmp_path: Path, document: str
) -> None:
    path = tmp_path / "model.yaml"
    path.write_text(document)

    with pytest.raises(ConfigurationError):
        load_model(path)


def test_load_model_rejects_duplicate_mapping_keys(tmp_path: Path) -> None:
    path = tmp_path / "model.yaml"
    path.write_text(
        "response: y\nresponse: other\nlikelihood: {family: gaussian, sigma: 1}\npredictor: {fixed: '1'}"
    )

    with pytest.raises(ConfigurationError, match="duplicate mapping key"):
        load_model(path)
