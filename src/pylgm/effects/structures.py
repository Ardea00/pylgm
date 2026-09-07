# src/pylgm/effects/structures.py
"""Between-group precisions for ``Grouped`` -- R-INLA's ``control.group``.

Deliberately separate from the effect specs of similar name: an effect carries
an index column and builds a design, whereas a structure carries only a
precision over the group levels and never touches the frame.

Each exposes ``levels`` (the group universe), ``precision``, and
``null_basis``. The null basis is what the Kronecker kernel needs to assemble
the composed block's constraints, and it is *not* derivable from the precision
without an eigendecomposition, so every structure states its own.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from scipy.sparse import csr_matrix, identity
from scipy.sparse.csgraph import connected_components

from pylgm.effects.ar1 import ar1_structure
from pylgm.effects.besag import _scaled_structure
from pylgm.effects.graph import normalize_graph
from pylgm.effects.random_walk import rw_structure


@dataclass(frozen=True)
class IIDStructure:
    """Independent groups. ``Grouped`` with this is exactly ``Replicated``.

    Kept as API because Knorr-Held types I, II and III each have an iid factor,
    so the equivalence oracle against ``SpaceTime`` needs it expressible.
    """

    def levels(self, observed: tuple[str, ...]) -> tuple[str, ...]:
        return observed

    def precision(self, levels: tuple[str, ...]) -> csr_matrix:
        return identity(len(levels), format="csr")

    def null_basis(self, levels: tuple[str, ...]) -> np.ndarray:
        return np.zeros((len(levels), 0))


@dataclass(frozen=True)
class AR1Structure:
    """Autoregressively correlated groups -- the panel case.

    ``rho`` is fixed in this slice, matching the restriction ``Shared`` carries
    today; estimating a structure's own hyperparameter jointly with the inner
    effect's is out of scope.
    """

    rho: float

    def __post_init__(self) -> None:
        if not isinstance(self.rho, (int, float)) or isinstance(self.rho, bool):
            raise TypeError("AR1Structure rho must be a real number")
        if not -1.0 < float(self.rho) < 1.0:
            raise ValueError("AR1Structure rho must lie strictly inside (-1, 1)")

    def levels(self, observed: tuple[str, ...]) -> tuple[str, ...]:
        return observed

    def precision(self, levels: tuple[str, ...]) -> csr_matrix:
        return ar1_structure(len(levels), float(self.rho))

    def null_basis(self, levels: tuple[str, ...]) -> np.ndarray:
        # The stationary AR1 precision is proper: no null space.
        return np.zeros((len(levels), 0))


class _RandomWalkStructure:
    """Shared body of RW1Structure and RW2Structure.

    A plain mixin, not a dataclass: ``order`` is declared by each subclass, and
    a fieldless frozen dataclass reading ``self.order`` would read as a defect.
    """

    order: ClassVar[int]

    def levels(self, observed: tuple[str, ...]) -> tuple[str, ...]:
        return observed

    def precision(self, levels: tuple[str, ...]) -> csr_matrix:
        if len(levels) <= self.order:
            raise ValueError(
                f"RW{self.order}Structure needs more than {self.order} group "
                f"level(s), got {len(levels)}"
            )
        return csr_matrix(rw_structure(len(levels), self.order, scale=True))

    def null_basis(self, levels: tuple[str, ...]) -> np.ndarray:
        count = len(levels)
        columns = [np.ones(count)]
        if self.order == 2:
            coordinate = np.arange(count, dtype=float)
            columns.append(coordinate - coordinate.mean())
        return np.column_stack(columns)


@dataclass(frozen=True)
class RW1Structure(_RandomWalkStructure):
    """First-order random walk between groups: null is the constant."""

    order: ClassVar[int] = 1


@dataclass(frozen=True)
class RW2Structure(_RandomWalkStructure):
    """Second-order random walk between groups: null is the constant and ramp."""

    order: ClassVar[int] = 2


@dataclass(frozen=True)
class BesagStructure:
    """Spatially structured groups (ICAR), aligned to the graph **by name**.

    The graph is the universe, not the observed levels: a node with no
    observations still gets its cell, so the spatial smoothing lends it
    strength. This matches ``Besag`` and ``build_spacetime``.

    Positional alignment would permute the neighbourhood structure silently --
    the fit would converge and return plausible numbers -- so an observed level
    outside the node set is a hard error.
    """

    graph: Mapping

    def _normalized(self):
        return normalize_graph(dict(self.graph))

    def levels(self, observed: tuple[str, ...]) -> tuple[str, ...]:
        nodes, _ = self._normalized()
        unknown = sorted({str(v) for v in observed} - set(nodes))
        if unknown:
            raise ValueError(
                f"BesagStructure graph has no node(s) {unknown!r}; a group level "
                "outside the graph cannot be aligned to a neighbourhood"
            )
        return nodes

    def _checked(self, levels: tuple[str, ...]):
        """The graph, asserting the caller passed the universe ``levels()`` gave.

        Both methods below would otherwise silently ignore their argument and
        return a matrix ordered by the graph while the caller indexed by
        something else -- the permuted-neighbourhood failure this class exists
        to prevent.
        """
        nodes, w = self._normalized()
        if tuple(levels) != nodes:
            raise ValueError(
                f"BesagStructure was given group levels {tuple(levels)!r} but its "
                f"graph orders nodes {nodes!r}; pass the tuple levels() returned"
            )
        return nodes, w

    def precision(self, levels: tuple[str, ...]) -> csr_matrix:
        nodes, w = self._checked(levels)
        return csr_matrix(_scaled_structure(w, nodes, scale=True))

    def null_basis(self, levels: tuple[str, ...]) -> np.ndarray:
        """One constant column per connected component of size >= 2.

        An isolated node (no neighbours) has no null direction: ``precision``
        treats it as an independent unit-variance IID node (see
        ``_scaled_structure``), which is already proper. Mirrors
        ``_component_constraints`` in ``besag.py``.
        """
        nodes, w = self._checked(levels)
        n_components, membership = connected_components(w, directed=False)
        columns = [
            (membership == component).astype(float)
            for component in range(n_components)
            if np.count_nonzero(membership == component) > 1
        ]
        if not columns:
            return np.zeros((len(nodes), 0))
        return np.column_stack(columns)
