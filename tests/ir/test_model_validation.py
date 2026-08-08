import numpy as np
import pytest
from scipy.sparse import csr_matrix

from pylgm.likelihoods import CompiledGaussian
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


def _compiled(likelihood: object) -> CompiledLGM:
    return CompiledLGM(
        y=np.array([2.0]),
        observed=np.array([True]),
        offset=np.zeros(1),
        design=csr_matrix([[1.0]]),
        precision=csr_matrix([[1.0]]),
        constraints=np.empty((0, 1)),
        labels=("x",),
        likelihood=likelihood,
        blocks=(),
    )


def test_compiled_lgm_owns_a_likelihood_not_gaussian_state() -> None:
    model = _compiled(CompiledGaussian(1.0))

    assert model.likelihood == CompiledGaussian(model.sigma)
    assert "sigma" not in model.__dataclass_fields__


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
        ({"likelihood": None}, "likelihood"),
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
        "likelihood": CompiledGaussian(1.0),
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
            likelihood=CompiledGaussian(1.0),
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
            likelihood=CompiledGaussian(1.0),
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
            likelihood=CompiledGaussian(1.0),
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
        "likelihood": CompiledGaussian(1.0),
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
            likelihood=CompiledGaussian(1.0),
            blocks=(block,),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("design", csr_matrix([[1.0 + 1.0j], [1.0 + 0.0j]])),
        ("precision", csr_matrix([[1.0 + 0.0j]])),
        ("constraints", np.array([[1.0 + 0.0j]])),
    ],
)
def test_latent_block_rejects_complex_dense_and_sparse_payloads(
    field: str, value: object
) -> None:
    arguments: dict[str, object] = {
        "name": "effect",
        "labels": ("level",),
        "design": csr_matrix(np.ones((2, 1))),
        "precision": csr_matrix([[1.0]]),
        "constraints": np.empty((0, 1)),
    }
    arguments[field] = value

    with pytest.raises(ModelValidationError, match="real numeric"):
        LatentBlock(**arguments)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("y", np.array([1.0 + 0.0j, 2.0 + 1.0j])),
        ("offset", np.array([0.0 + 0.0j, 0.0 + 0.0j])),
        ("design", csr_matrix([[1.0 + 0.0j], [1.0 + 0.0j]])),
        ("precision", csr_matrix([[1.0 + 0.0j]])),
        ("constraints", np.empty((0, 1), dtype=complex)),
    ],
)
def test_compiled_lgm_rejects_complex_dense_and_sparse_payloads(
    field: str, value: object
) -> None:
    arguments: dict[str, object] = {
        "y": np.array([1.0, 2.0]),
        "observed": np.array([True, True]),
        "offset": np.zeros(2),
        "design": csr_matrix(np.ones((2, 1))),
        "precision": csr_matrix([[1.0]]),
        "constraints": np.empty((0, 1)),
        "labels": ("x",),
        "likelihood": CompiledGaussian(1.0),
        "blocks": (),
    }
    arguments[field] = value

    with pytest.raises(ModelValidationError, match="real numeric"):
        CompiledLGM(**arguments)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("y", object()),
        ("offset", [[0.0], [0.0, 1.0]]),
        ("observed", [[True], [False, True]]),
        ("likelihood", None),
    ],
)
def test_invalid_compiled_scalar_and_dtype_inputs_are_typed_model_errors(
    field: str, value: object
) -> None:
    arguments: dict[str, object] = {
        "y": np.array([1.0, 2.0]),
        "observed": np.array([True, True]),
        "offset": np.zeros(2),
        "design": csr_matrix(np.ones((2, 1))),
        "precision": csr_matrix([[1.0]]),
        "constraints": np.empty((0, 1)),
        "labels": ("x",),
        "likelihood": CompiledGaussian(1.0),
        "blocks": (),
    }
    arguments[field] = value

    with pytest.raises(ModelValidationError):
        CompiledLGM(**arguments)  # type: ignore[arg-type]


def test_invalid_latent_constraint_array_is_a_typed_model_error() -> None:
    with pytest.raises(ModelValidationError):
        LatentBlock(
            "effect",
            ("level",),
            csr_matrix(np.ones((2, 1))),
            csr_matrix([[1.0]]),
            [[1.0], [1.0, 2.0]],  # type: ignore[arg-type]
        )
