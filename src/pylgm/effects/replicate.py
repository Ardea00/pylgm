"""Independent replicates of a latent effect, sharing its hyperparameters.

``R`` copies of one effect: precision ``I_R (x) Q``, design on
``(replicate, level)`` pairs, constraints ``I_R (x) C``. The layout is
replicate-major -- ``cell = replicate * n_levels + level`` -- matching
``build_ar1``'s group-major convention, which is what lets a replicated AR1
match the shipped ``AR1(group=)`` implementation bit for bit.
"""

import numpy as np
import pandas as pd
from scipy.sparse import identity

from pylgm.effects.kronecker import kron_block
from pylgm.ir.model import LatentBlock


def replicate_levels(frame: pd.DataFrame, name: str, over: str) -> tuple[str, ...]:
    """The sorted replicate levels, rejecting a missing or null column."""
    if over not in frame.columns:
        raise ValueError(f"{name} replicate column {over!r} not found")
    if frame[over].isna().any():
        raise ValueError(f"{name} replicate column {over!r} must not contain null values")
    return tuple(sorted({str(value) for value in frame[over]}))


def replicated_block(
    inner: LatentBlock,
    frame: pd.DataFrame,
    index: str,
    over: str,
    replicates: tuple[str, ...],
) -> LatentBlock:
    """Compose ``inner`` -- built over the level set alone -- into ``R`` copies.

    ``inner.constraints`` is replicated per copy rather than shared: one
    constraint over ``R`` replicates would leave ``R-1`` directions
    unidentified, and the fit would still converge on plausible numbers.
    """
    levels = inner.labels
    n_replicates = len(replicates)
    level_position = {level: column for column, level in enumerate(levels)}
    replicate_position = {label: row for row, label in enumerate(replicates)}

    keys = frame[index].map(str)
    unknown = sorted({value for value in keys if value not in level_position})
    if unknown:
        raise ValueError(
            f"{inner.name} index {index!r} has level(s) {unknown!r} absent from the "
            "replicated block's own level set"
        )
    replicate_positions = np.array([replicate_position[str(r)] for r in frame[over]])
    level_positions = np.array([level_position[t] for t in keys])
    return kron_block(
        inner.name,
        replicates, identity(n_replicates, format="csr"), np.zeros((n_replicates, 0)),
        levels, inner.precision, inner.constraints.T,
        replicate_positions, level_positions,
        separator="@", orthonormalise=False,
    )
