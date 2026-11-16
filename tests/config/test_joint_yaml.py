"""YAML ``Joint`` == the Python declaration."""
from pathlib import Path

import pytest

from pylgm import Fixed, IID, LGM, Poisson
from pylgm.config import load_joint
from pylgm.exceptions import ConfigurationError
from pylgm.joint import Joint, Shared

DOC = """
outcomes:
  - response: oral
    likelihood: {family: poisson}
    predictor: {fixed: "1"}
  - response: larynx
    likelihood: {family: poisson}
    predictor: {fixed: "1"}
shared:
  - effect: {name: u, type: iid, index: district, precision: 2.0}
    scale: [1.6, 0.625]     # Knorr-Held & Best (delta, 1/delta), delta fixed
"""


def _python() -> Joint:
    return Joint(
        [LGM(response="oral", likelihood=Poisson(), predictor=Fixed("1")),
         LGM(response="larynx", likelihood=Poisson(), predictor=Fixed("1"))],
        shared=[Shared(IID("u", index="district", precision=2.0), scale=(1.6, 0.625))],
    )


def _load(tmp_path: Path, doc: str = DOC) -> Joint:
    path = tmp_path / "joint.yaml"
    path.write_text(doc)
    return load_joint(path)


def test_yaml_joint_equals_python_joint(tmp_path, shared_component_frame):
    loaded, expected = _load(tmp_path), _python()
    assert loaded.submodels == expected.submodels
    assert loaded.shared == expected.shared

    frame, _ = shared_component_frame()
    a, b = loaded.fit(frame), expected.fit(frame)
    assert a.log_marginal_likelihood == b.log_marginal_likelihood
    assert a.labels == b.labels


@pytest.mark.parametrize(
    ("doc", "match"),
    [
        (DOC.replace("shared:", "shared_x:"), "shared_x"),
        (DOC.replace("name: u, ", ""), "needs a name"),
        (DOC.replace("[1.6, 0.625]", "[1.0, 2.0, 3.0]"), "scale tuple has length 3"),
        (DOC.replace("scale: [1.6, 0.625]", "scale: {pair: delta}"), "scale"),
        (DOC.replace("likelihood: {family: poisson}", "likelihood: {family: gaussian}", 1),
         "sigma is required"),
        (DOC.replace("precision: 2.0}", "precision: 2.0, graph_file: g}"), "not valid"),
    ],
)
def test_joint_schema_rejections(tmp_path, doc, match):
    with pytest.raises(ConfigurationError, match=match):
        _load(tmp_path, doc)


def test_joint_needs_two_outcomes(tmp_path):
    one = DOC.split("  - response: larynx")[0]
    with pytest.raises(ConfigurationError, match="at least two"):
        _load(tmp_path, one)
