import numpy as np
import pytest
from scipy.sparse import csr_matrix

from pylgm.exceptions import ModelValidationError
from pylgm.ir import CompiledLGM, LatentBlock


def _block(name: str = "effect", rows: int = 2) -> LatentBlock:
    return LatentBlock(
        name=name,
        labels=("level",),
        design=csr_matrix(np.ones((rows, 1))),
        precision=csr_matrix([[1.0]]),
        constraints=np.empty((0, 1)),
    )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"labels": ()}, "labels"),
        ({"design": csr_matrix([[np.nan], [1.0]])}, "design"),
        ({"precision": csr_matrix([[1.0, 0.0]])}, "precision"),
        ({"precision": csr_matrix([[np.inf]])}, "precision"),
        ({"constraints": np.empty((1, 2))}, "constraints"),
    ],
)
def test_latent_block_centralizes_shape_and_finite_invariants(
    changes: dict[str, object], message: str
) -> None:
    arguments: dict[str, object] = {
        "name": "effect",
        "labels": ("level",),
        "design": csr_matrix(np.ones((2, 1))),
        "precision": csr_matrix([[1.0]]),
        "constraints": np.empty((0, 1)),
    }
    arguments.update(changes)

    with pytest.raises(ModelValidationError, match=message):
        LatentBlock(**arguments)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"observed": np.array([1, 0])}, "boolean"),
        ({"y": np.array([1.0])}, "row counts"),
        ({"offset": np.array([0.0, np.inf])}, "offset"),
        ({"design": csr_matrix([[1.0], [np.nan]])}, "design"),
        ({"constraints": np.empty((0, 2))}, "constraints"),
        ({"labels": ()}, "labels"),
        ({"sigma": np.inf}, "sigma"),
    ],
)
def test_compiled_lgm_centralizes_engine_independent_invariants(
    changes: dict[str, object], message: str
) -> None:
    arguments: dict[str, object] = {
        "y": np.array([1.0, 0.0]),
        "observed": np.array([True, False]),
        "offset": np.zeros(2),
        "design": csr_matrix(np.ones((2, 1))),
        "precision": csr_matrix([[1.0]]),
        "constraints": np.empty((0, 1)),
        "labels": ("effect:level",),
        "sigma": 1.0,
        "blocks": (_block(),),
    }
    arguments.update(changes)

    with pytest.raises(ModelValidationError, match=message):
        CompiledLGM(**arguments)  # type: ignore[arg-type]


def test_compiled_lgm_requires_unique_block_identities() -> None:
    first = _block("duplicate")
    second = _block("duplicate")

    with pytest.raises(ModelValidationError, match="block names must be unique"):
        CompiledLGM(
            y=np.array([1.0, 2.0]),
            observed=np.array([True, True]),
            offset=np.zeros(2),
            design=csr_matrix(np.ones((2, 2))),
            precision=csr_matrix(np.eye(2)),
            constraints=np.empty((0, 2)),
            labels=("duplicate:level", "duplicate:level-2"),
            sigma=1.0,
            blocks=(first, second),
        )


def test_compiled_lgm_aligns_nonempty_blocks_with_global_latent_width() -> None:
    with pytest.raises(ModelValidationError, match="block widths"):
        CompiledLGM(
            y=np.array([1.0, 2.0]),
            observed=np.array([True, True]),
            offset=np.zeros(2),
            design=csr_matrix(np.ones((2, 2))),
            precision=csr_matrix(np.eye(2)),
            constraints=np.empty((0, 2)),
            labels=("effect:level", "orphan"),
            sigma=1.0,
            blocks=(_block(),),
        )


def test_compiled_lgm_rejects_block_with_mismatched_design_rows() -> None:
    with pytest.raises(ModelValidationError, match="block design row counts"):
        CompiledLGM(
            y=np.array([1.0, 2.0]),
            observed=np.array([True, True]),
            offset=np.zeros(2),
            design=csr_matrix(np.ones((2, 1))),
            precision=csr_matrix([[1.0]]),
            constraints=np.empty((0, 1)),
            labels=("effect:level",),
            sigma=1.0,
            blocks=(_block(rows=1),),
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("design", csr_matrix(np.zeros((2, 1))), "design must match"),
        ("precision", csr_matrix([[2.0]]), "precision must match"),
    ],
)
def test_compiled_lgm_requires_global_matrices_to_match_blocks(
    field: str, value: object, message: str
) -> None:
    arguments: dict[str, object] = {
        "y": np.array([1.0, 2.0]),
        "observed": np.array([True, True]),
        "offset": np.zeros(2),
        "design": csr_matrix(np.ones((2, 1))),
        "precision": csr_matrix([[1.0]]),
        "constraints": np.empty((0, 1)),
        "labels": ("effect:level",),
        "sigma": 1.0,
        "blocks": (_block(),),
    }
    arguments[field] = value

    with pytest.raises(ModelValidationError, match=message):
        CompiledLGM(**arguments)  # type: ignore[arg-type]


def test_compiled_lgm_requires_global_constraints_to_match_blocks() -> None:
    block = LatentBlock(
        "effect",
        ("level",),
        csr_matrix(np.ones((2, 1))),
        csr_matrix([[1.0]]),
        np.array([[1.0]]),
    )

    with pytest.raises(ModelValidationError, match="constraints must match"):
        CompiledLGM(
            y=np.array([1.0, 2.0]),
            observed=np.array([True, True]),
            offset=np.zeros(2),
            design=csr_matrix(np.ones((2, 1))),
            precision=csr_matrix([[1.0]]),
            constraints=np.array([[2.0]]),
            labels=("effect:level",),
            sigma=1.0,
            blocks=(block,),
        )
