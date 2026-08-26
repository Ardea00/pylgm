from pathlib import Path

import pytest

from pylgm import Bernoulli, Fixed, Gaussian, IID, LGM, Poisson, RW1
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


def test_load_poisson_model_with_offset(tmp_path):
    path = tmp_path / "m.yaml"
    path.write_text(
        "response: y\n"
        "likelihood:\n  family: poisson\n"
        "offset: logexp\n"
        "data:\n  time: t\n"
        "predictor:\n  fixed: '1 + x'\n"
    )
    model = load_model(path)
    assert isinstance(model.likelihood, Poisson)
    assert model.offset == "logexp"


def test_load_bernoulli_model(tmp_path):
    path = tmp_path / "m.yaml"
    path.write_text(
        "response: y\n"
        "likelihood:\n  family: bernoulli\n"
        "data:\n  time: t\n"
        "predictor:\n  fixed: '1'\n"
    )
    model = load_model(path)
    assert isinstance(model.likelihood, Bernoulli)
    assert model.offset is None


_SPATIAL_GRAPH = {"a": ["b"], "b": ["a", "c"], "c": ["b"]}
_INLINE_GRAPH = "{a: [b], b: [a, c], c: [b]}"


def _spatial_doc(effect_body: str) -> str:
    return (
        "response: y\n"
        "likelihood: {family: gaussian, sigma: 1.0}\n"
        "data: {time: time}\n"
        "predictor:\n"
        "  fixed: '1'\n"
        "  effects:\n"
        f"    - {{{effect_body}}}\n"
    )


def _spatial_lgm(effect):
    return LGM(
        response="y",
        likelihood=Gaussian(1.0),
        predictor=Fixed("1") + effect,
        time="time",
    )


def test_load_model_builds_besag_from_inline_graph(tmp_path: Path) -> None:
    from pylgm import Besag

    path = tmp_path / "m.yaml"
    path.write_text(
        _spatial_doc(f"name: area, type: besag, index: region, precision: 2.0, graph: {_INLINE_GRAPH}")
    )

    assert load_model(path) == _spatial_lgm(Besag("area", "region", _SPATIAL_GRAPH, 2.0))


def test_load_model_builds_besag_from_graph_file(tmp_path: Path) -> None:
    from pylgm import Besag, load_graph_file

    graph_path = tmp_path / "nb.graph"
    graph_path.write_text("3\n1 1 2\n2 2 1 3\n3 1 2\n")  # path graph 1-2-3
    path = tmp_path / "m.yaml"
    path.write_text(_spatial_doc("name: area, type: besag, index: region, graph_file: nb.graph"))

    assert load_model(path) == _spatial_lgm(
        Besag("area", "region", load_graph_file(graph_path), 1.0)
    )


def test_load_model_builds_proper_car_with_fixed_rho(tmp_path: Path) -> None:
    from pylgm import ProperCAR

    path = tmp_path / "m.yaml"
    path.write_text(
        _spatial_doc(f"name: area, type: proper_car, index: region, rho: 0.4, graph: {_INLINE_GRAPH}")
    )

    assert load_model(path) == _spatial_lgm(ProperCAR("area", "region", _SPATIAL_GRAPH, 0.4))


def test_load_model_builds_bym2_with_default_phi(tmp_path: Path) -> None:
    from pylgm import BYM2

    path = tmp_path / "m.yaml"
    path.write_text(
        _spatial_doc(f"name: area, type: bym2, index: region, graph: {_INLINE_GRAPH}")
    )

    assert load_model(path) == _spatial_lgm(BYM2("area", "region", _SPATIAL_GRAPH))


@pytest.mark.parametrize(
    "effect_body, match",
    [
        # proper_car without its required rho
        (f"name: area, type: proper_car, index: region, graph: {_INLINE_GRAPH}", "rho"),
        # rho on a non-proper_car spatial effect
        (f"name: area, type: besag, index: region, rho: 0.5, graph: {_INLINE_GRAPH}", "rho"),
        # graph field on a simple effect
        (f"name: r, type: iid, index: region, precision: 1.0, graph: {_INLINE_GRAPH}", "not valid"),
        # spatial effect with neither graph nor graph_file
        ("name: area, type: besag, index: region", "exactly one"),
        # spatial effect with both graph and graph_file
        (f"name: area, type: besag, index: region, graph_file: x.graph, graph: {_INLINE_GRAPH}", "exactly one"),
    ],
)
def test_load_model_rejects_mismatched_spatial_fields(
    tmp_path: Path, effect_body: str, match: str
) -> None:
    path = tmp_path / "m.yaml"
    path.write_text(_spatial_doc(effect_body))

    with pytest.raises(ConfigurationError, match=match):
        load_model(path)


def test_load_model_reports_missing_graph_file(tmp_path: Path) -> None:
    path = tmp_path / "m.yaml"
    path.write_text(_spatial_doc("name: area, type: besag, index: region, graph_file: nope.graph"))

    with pytest.raises(ConfigurationError):
        load_model(path)
