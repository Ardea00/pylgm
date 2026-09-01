"""Joint latent Gaussian models: several responses stacked into one CompiledLGM.

A joint model is an ordinary :class:`~pylgm.ir.model.CompiledLGM` with more
rows. Responses stack as ``y = (y^(1), ..., y^(K))``; sub-model ``k`` occupies a
contiguous row slice. Private latent blocks are zero-padded outside their slice,
shared blocks carry scaled rows in every slice they enter, and the likelihood
becomes a row-dispatching :class:`~pylgm.likelihoods.CompiledMixture`. Both IR
invariants -- ``design == hstack(blocks)`` and ``precision == block_diag(blocks)``
-- are preserved.
"""

from dataclasses import dataclass

from scipy.sparse import csr_matrix, vstack

from pylgm.ir.model import LatentBlock
from pylgm.parameters import Hyperparameter


def _pad_block_rows(block: LatentBlock, before: int, after: int) -> LatentBlock:
    """Zero-pad a block's design rows into the stacked row space.

    Precision, labels and constraints are row-independent and pass through
    untouched, so a Besag sum-to-zero still constrains exactly what it did.
    """
    if before == 0 and after == 0:
        return block
    width = block.design.shape[1]
    pieces = []
    if before:
        pieces.append(csr_matrix((before, width)))
    pieces.append(block.design)
    if after:
        pieces.append(csr_matrix((after, width)))
    return LatentBlock(
        block.name,
        block.labels,
        vstack(pieces, format="csr"),
        block.precision,
        block.constraints,
    )


# A scaled shared field enters slice k as `scale_k * u`. The sentinel
# ("<name>", "inverse") means "the reciprocal of the hyperparameter <name>",
# which is how the Knorr-Held & Best (delta, delta^-1) pairing is carried
# through compilation without inventing an expression language.
InverseOf = tuple[str, str]


@dataclass(frozen=True)
class Shared:
    """One latent field entering several sub-models with a per-sub-model scaling.

    ``scale`` is a float (broadcast to every sub-model), a ``Hyperparameter``
    (shorthand for the Knorr-Held & Best ``(delta, delta^-1)`` pairing, and
    therefore valid only for exactly two sub-models), or an explicit
    per-sub-model tuple of floats and/or ``Hyperparameter``s.
    """

    effect: object
    scale: object = 1.0
    allow_ragged: bool = False
    """Accept a shared index whose level set differs between sub-models.

    The latent always spans the union of levels; this only silences the report.
    Off by default because an unintended mismatch weakens the shared field
    without any visible symptom.
    """

    def __post_init__(self) -> None:
        if not hasattr(self.effect, "name"):
            raise TypeError("Shared effect must be a latent effect spec")
        if not isinstance(self.allow_ragged, bool):
            raise TypeError("Shared allow_ragged must be a bool")
        scale = self.scale
        if isinstance(scale, (tuple, list)):
            entries = tuple(scale)
            if not entries:
                raise ValueError("Shared scale tuple must be non-empty")
            for entry in entries:
                if not isinstance(entry, (int, float, Hyperparameter)):
                    raise TypeError(
                        "Shared scale entries must be floats or Hyperparameters"
                    )
            object.__setattr__(self, "scale", entries)
        elif not isinstance(scale, (int, float, Hyperparameter)):
            raise TypeError("Shared scale must be a float, Hyperparameter, or tuple")

    @property
    def name(self) -> str:
        return self.effect.name

    def scales_for(self, count: int) -> tuple:
        """Expand ``scale`` to one entry per sub-model."""
        scale = self.scale
        if isinstance(scale, tuple):
            if len(scale) != count:
                raise ValueError(
                    f"Shared {self.name!r} scale tuple has length {len(scale)}, "
                    f"but the joint has {count} sub-models"
                )
            return scale
        if isinstance(scale, Hyperparameter):
            if count != 2:
                raise ValueError(
                    f"Shared {self.name!r} has a scalar Hyperparameter scale, which is "
                    "the (delta, delta^-1) shorthand and requires exactly two "
                    f"sub-models; this joint has {count}. Pass an explicit "
                    "per-sub-model tuple instead."
                )
            return (scale, (scale.name, "inverse"))
        return tuple(float(scale) for _ in range(count))


@dataclass(frozen=True)
class Joint:
    """Several `LGM` sub-models fitted as one stacked latent Gaussian model."""

    submodels: tuple = ()
    shared: tuple = ()

    def __init__(self, submodels, shared=()) -> None:
        submodels = tuple(submodels)
        if len(submodels) < 2:
            raise ValueError("Joint requires at least two sub-models")
        responses = [model.response for model in submodels]
        if len(responses) != len(set(responses)):
            raise ValueError("Joint sub-model response names must be unique")
        shared = tuple(shared)
        for entry in shared:
            if not isinstance(entry, Shared):
                raise TypeError("Joint shared entries must be Shared instances")
            entry.scales_for(len(submodels))
        shared_names = [entry.name for entry in shared]
        if len(shared_names) != len(set(shared_names)):
            raise ValueError("Joint shared effect names must be unique")
        object.__setattr__(self, "submodels", submodels)
        object.__setattr__(self, "shared", shared)

    @property
    def outcomes(self) -> tuple[str, ...]:
        return tuple(model.response for model in self.submodels)

    @classmethod
    def _unchecked(cls, submodels, shared=()):
        """Bypass the two-sub-model minimum. Test-only: used by the reduction test."""
        obj = object.__new__(cls)
        object.__setattr__(obj, "submodels", tuple(submodels))
        object.__setattr__(obj, "shared", tuple(shared))
        return obj
