# src/pylgm/effects/kronecker.py
"""The Kronecker composition shared by Replicated, Grouped and SpaceTime.

One outer factor, one inner factor, cells laid out outer-major:

    Q      = Q_out (x) Q_in
    cell   = outer_position * n_inner + inner_position
    null(Q) = null(Q_out) (x) R^in  +  R^out (x) null(Q_in)

A block's ``constraints`` are a basis of its precision's null space -- RW1
carries one, RW2 two, Besag one per connected component -- so that last line
covers every composition in the library. ``Replicated`` is its degenerate case:
``null(I_R)`` is empty, the first term vanishes, and ``kron(I_R, C)`` remains.

Callers pass already-resolved integer positions and own their own validation:
``replicated_block`` and ``build_spacetime`` raise different, test-pinned
messages for an unknown level. This module is therefore pure numpy/scipy.
"""

import numpy as np
from scipy.sparse import csr_matrix, kron

from pylgm.ir.model import LatentBlock


def kron_null_constraints(
    null_out: np.ndarray,
    null_in: np.ndarray,
    n_out: int,
    n_in: int,
    orthonormalise: bool,
) -> np.ndarray:
    """Constraint rows spanning ``null(Q_out (x) Q_in)``, shape ``(k, n_out*n_in)``.

    ``orthonormalise`` governs the **one-part** case only, and exists to
    preserve two released outputs rather than for any modelling reason -- both
    settings span the same subspace, so no fit changes either way:

    - ``False`` returns the raw ``kron(I_out, N_in).T``, which is the literal
      ``kron(I_R, C)`` ``Replicated`` has always produced and whose bit-for-bit
      equality with the shipped ``AR1(group=)`` is slice 3's safety property.
    - ``True`` matches ``build_spacetime``, which orthonormalises today
      including for types II and III where only one part is present.

    When **both** parts are present the flag is ignored. The two spans overlap
    in ``1_out (x) 1_in``, and dropping that duplicate is what the SVD is for;
    keeping it would return a rank-deficient constraint matrix.
    """
    parts = []
    if null_out.shape[1]:
        parts.append(np.kron(null_out, np.eye(n_in)))
    if null_in.shape[1]:
        parts.append(np.kron(np.eye(n_out), null_in))
    if not parts:
        return np.zeros((0, n_out * n_in))
    if len(parts) == 1 and not orthonormalise:
        return parts[0].T
    b = np.hstack(parts)
    u, singular, _ = np.linalg.svd(b, full_matrices=False)
    rank = int(np.sum(singular > singular[0] * 1e-10)) if singular.size else 0
    return u[:, :rank].T


def kron_block(
    name: str,
    outer_labels: tuple[str, ...],
    outer_precision: csr_matrix,
    null_out: np.ndarray,
    inner_labels: tuple[str, ...],
    inner_precision: csr_matrix,
    null_in: np.ndarray,
    outer_positions: np.ndarray,
    inner_positions: np.ndarray,
    *,
    separator: str,
    orthonormalise: bool,
    precision_scale: float = 1.0,
) -> LatentBlock:
    """Compose two factors into one outer-major ``LatentBlock``.

    ``outer_positions`` and ``inner_positions`` hold one already-validated
    index per frame row. ``separator`` is ``"@"`` for Replicated and Grouped
    and ``"|"`` for SpaceTime, whose labels are user-visible in
    ``result.labels`` and therefore cannot change. ``precision_scale``
    multiplies the composed precision exactly as ``build_spacetime`` applies
    its scalar today -- after the Kronecker product, not folded into either
    factor, since IEEE multiplication is not associative.
    """
    n_out, n_in = len(outer_labels), len(inner_labels)
    width = n_out * n_in
    cells = np.asarray(outer_positions) * n_in + np.asarray(inner_positions)
    rows = len(cells)
    design = csr_matrix(
        (np.ones(rows), (np.arange(rows), cells)), shape=(rows, width)
    )
    precision = csr_matrix(
        precision_scale * kron(outer_precision, inner_precision, format="csr")
    )
    constraints = kron_null_constraints(null_out, null_in, n_out, n_in, orthonormalise)
    labels = tuple(f"{o}{separator}{i}" for o in outer_labels for i in inner_labels)
    return LatentBlock(name, labels, design, precision, constraints)
