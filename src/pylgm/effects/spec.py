"""Declarative latent-effect specifications."""

from collections.abc import Mapping
from dataclasses import dataclass
import math
from typing import TypeAlias
import warnings

from pylgm.parameters import Hyperparameter


def _non_empty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _positive_real(value: object, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive real value")
    return float(value)


def _positive_precision(
    value: float | Hyperparameter, name: str
) -> float | Hyperparameter:
    return value if isinstance(value, Hyperparameter) else _positive_real(value, name)


def _finite_real(value: object, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite real value")
    return float(value)


class _ComposableEffect:
    def __add__(self, other: object) -> "Predictor":
        return Predictor((self,)) + other


@dataclass(frozen=True)
class Fixed(_ComposableEffect):
    """A fixed-effect formula and its Gaussian prior precision."""

    formula: str
    prior_precision: float = 1e-6

    def __post_init__(self) -> None:
        object.__setattr__(self, "formula", _non_empty_string(self.formula, "formula"))
        object.__setattr__(
            self,
            "prior_precision",
            _positive_real(self.prior_precision, "prior_precision"),
        )

    @property
    def name(self) -> str:
        return "fixed"


@dataclass(frozen=True)
class IID(_ComposableEffect):
    """An independent and identically distributed latent effect."""

    name: str
    index: str
    precision: float | Hyperparameter = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _non_empty_string(self.name, "name"))
        object.__setattr__(self, "index", _non_empty_string(self.index, "index"))
        object.__setattr__(
            self, "precision", _positive_precision(self.precision, "precision")
        )


@dataclass(frozen=True)
class RW1(_ComposableEffect):
    """A first-order random-walk latent effect."""

    name: str
    index: str
    precision: float | Hyperparameter = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _non_empty_string(self.name, "name"))
        object.__setattr__(self, "index", _non_empty_string(self.index, "index"))
        object.__setattr__(
            self, "precision", _positive_precision(self.precision, "precision")
        )


@dataclass(frozen=True)
class RW2(_ComposableEffect):
    """A second-order random-walk latent effect."""

    name: str
    index: str
    precision: float | Hyperparameter = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _non_empty_string(self.name, "name"))
        object.__setattr__(self, "index", _non_empty_string(self.index, "index"))
        object.__setattr__(
            self, "precision", _positive_precision(self.precision, "precision")
        )


@dataclass(frozen=True)
class AR1(_ComposableEffect):
    """A stationary first-order autoregressive latent effect.

    With ``replicate`` set, the effect is one independent AR1 series per level
    of that column -- the panel-econometrics case (a separate series per firm,
    country, or region) -- sharing ``precision`` and ``rho`` across replicates
    but not their realizations. This is R-INLA's ``f(index, model=...,
    replicate=r)``; it is unrelated to R-INLA's own ``group``, which means
    *correlated* copies with a between-group structure.

    ``group`` is the deprecated former name for ``replicate`` -- kept for
    backward compatibility and folded into ``replicate`` with a
    ``DeprecationWarning``.
    """

    name: str
    index: str
    precision: float | Hyperparameter = 1.0
    rho: float | Hyperparameter = 0.5
    replicate: str | None = None
    group: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _non_empty_string(self.name, "name"))
        object.__setattr__(self, "index", _non_empty_string(self.index, "index"))
        if self.replicate is not None and self.group is not None:
            raise ValueError(
                "AR1 takes either `replicate` or the deprecated `group`, not both"
            )
        if self.group is not None:
            object.__setattr__(self, "group", _non_empty_string(self.group, "group"))
            warnings.warn(
                "AR1(group=) is R-INLA's `replicate` -- independent series sharing "
                "hyperparameters -- under the wrong name. Use AR1(replicate=) "
                "instead. R-INLA's own `group` means correlated copies with a "
                "between-group structure, which this is not.",
                DeprecationWarning,
                stacklevel=3,
            )
            object.__setattr__(self, "replicate", self.group)
        elif self.replicate is not None:
            object.__setattr__(
                self, "replicate", _non_empty_string(self.replicate, "replicate")
            )
        object.__setattr__(
            self, "precision", _positive_precision(self.precision, "precision")
        )
        if not isinstance(self.rho, Hyperparameter):
            rho = _finite_real(self.rho, "rho")
            if not -1.0 < rho < 1.0:
                raise ValueError("rho must lie strictly inside (-1, 1)")
            object.__setattr__(self, "rho", rho)


@dataclass(frozen=True)
class Seasonal(_ComposableEffect):
    """A slowly-drifting seasonal effect of the declared ``period``.

    ``precision`` penalizes drift away from a repeating pattern; the fixed
    patterns themselves are the signal and stay unpenalized by it, held only by
    the fixed ``ridge`` that keeps the block positive definite. Raise ``ridge``
    to shrink the seasonal amplitude toward zero, lower it to free it further.
    """

    name: str
    index: str
    period: int
    precision: float | Hyperparameter = 1.0
    ridge: float = 1e-6

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _non_empty_string(self.name, "name"))
        object.__setattr__(self, "index", _non_empty_string(self.index, "index"))
        if isinstance(self.period, bool) or not isinstance(self.period, int):
            raise TypeError("period must be an integer")
        if self.period < 2:
            raise ValueError("period must be at least 2")
        object.__setattr__(
            self, "precision", _positive_precision(self.precision, "precision")
        )
        ridge = _finite_real(self.ridge, "ridge")
        if ridge <= 0.0:
            raise ValueError("ridge must be finite and > 0")
        object.__setattr__(self, "ridge", ridge)


@dataclass(frozen=True)
class Besag(_ComposableEffect):
    """A Besag / intrinsic CAR (ICAR) spatial latent effect."""

    name: str
    index: str
    graph: Mapping
    precision: float | Hyperparameter = 1.0
    scale: bool = True

    def __post_init__(self) -> None:
        from pylgm.effects.graph import canonical_graph

        object.__setattr__(self, "name", _non_empty_string(self.name, "name"))
        object.__setattr__(self, "index", _non_empty_string(self.index, "index"))
        object.__setattr__(
            self, "precision", _positive_precision(self.precision, "precision")
        )
        if not isinstance(self.scale, bool):
            raise ValueError("scale must be a boolean")
        object.__setattr__(self, "graph", canonical_graph(self.graph))


@dataclass(frozen=True)
class ProperCAR(_ComposableEffect):
    """A proper conditional autoregressive (proper CAR) spatial latent effect."""

    name: str
    index: str
    graph: Mapping
    rho: float | Hyperparameter
    precision: float | Hyperparameter = 1.0

    def __post_init__(self) -> None:
        from pylgm.effects.graph import canonical_graph

        object.__setattr__(self, "name", _non_empty_string(self.name, "name"))
        object.__setattr__(self, "index", _non_empty_string(self.index, "index"))
        if isinstance(self.rho, Hyperparameter):
            object.__setattr__(self, "rho", self.rho)  # interval resolved at compile time
        else:
            object.__setattr__(self, "rho", _finite_real(self.rho, "rho"))
        object.__setattr__(
            self, "precision", _positive_precision(self.precision, "precision")
        )
        object.__setattr__(self, "graph", canonical_graph(self.graph))


@dataclass(frozen=True)
class SAR(_ComposableEffect):
    """A directed spatial-autoregressive latent effect: precision
    ``precision * (I - ρW)ᵀ(I - ρW)`` on a row-standardized directed graph."""

    name: str
    index: str
    graph: Mapping
    rho: float | Hyperparameter
    precision: float | Hyperparameter = 1.0

    def __post_init__(self) -> None:
        from pylgm.effects.directed_graph import canonical_directed_graph

        object.__setattr__(self, "name", _non_empty_string(self.name, "name"))
        object.__setattr__(self, "index", _non_empty_string(self.index, "index"))
        if isinstance(self.rho, Hyperparameter):
            object.__setattr__(self, "rho", self.rho)
        else:
            rho = _finite_real(self.rho, "rho")
            if not -1.0 < rho < 1.0:
                raise ValueError("rho must lie strictly inside (-1, 1)")
            object.__setattr__(self, "rho", rho)
        object.__setattr__(
            self, "precision", _positive_precision(self.precision, "precision")
        )
        object.__setattr__(self, "graph", canonical_directed_graph(self.graph))


@dataclass(frozen=True)
class DynamicSpatialPanel(_ComposableEffect):
    """A dynamic spatial panel (SDPD): per-period directed networks ``W_t`` with
    contemporaneous (``rho``), temporal (``gamma``) and spatio-temporal-diffusion
    (``eta``) coefficients. Precision ``precision * MᵀM`` for the block-bidiagonal
    operator ``M`` (see the S-slice design spec)."""

    name: str
    unit: str
    time: str
    graphs: Mapping
    rho: float | Hyperparameter
    gamma: float | Hyperparameter = 0.0
    eta: float | Hyperparameter = 0.0
    precision: float | Hyperparameter = 1.0

    def __post_init__(self) -> None:
        from pylgm.effects.directed_graph import (
            canonical_directed_graph,
            graphs_by_label,
            sorted_time_keys,
        )

        object.__setattr__(self, "name", _non_empty_string(self.name, "name"))
        object.__setattr__(self, "unit", _non_empty_string(self.unit, "unit"))
        object.__setattr__(self, "time", _non_empty_string(self.time, "time"))
        if isinstance(self.rho, Hyperparameter):
            object.__setattr__(self, "rho", self.rho)
        else:
            rho = _finite_real(self.rho, "rho")
            if not -1.0 < rho < 1.0:
                raise ValueError("rho must lie strictly inside (-1, 1)")
            object.__setattr__(self, "rho", rho)
        for field_name in ("gamma", "eta"):
            value = getattr(self, field_name)
            if not isinstance(value, Hyperparameter):
                object.__setattr__(self, field_name, _finite_real(value, field_name))
        object.__setattr__(
            self, "precision", _positive_precision(self.precision, "precision")
        )
        if not isinstance(self.graphs, Mapping) or not self.graphs:
            raise ValueError("graphs must be a non-empty mapping of time -> graph")
        by_label = graphs_by_label(self.graphs)
        canonical = tuple(
            (t, canonical_directed_graph(by_label[t]))
            for t in sorted_time_keys(self.graphs)
        )
        object.__setattr__(self, "graphs", canonical)

    @property
    def index(self) -> str:
        return self.unit


@dataclass(frozen=True)
class BYM2(_ComposableEffect):
    """A BYM2 spatial latent effect: scaled ICAR + IID mixed by ``phi``."""

    name: str
    index: str
    graph: Mapping
    precision: float | Hyperparameter = 1.0
    phi: float | Hyperparameter = 0.5

    def __post_init__(self) -> None:
        from pylgm.effects.graph import canonical_graph

        object.__setattr__(self, "name", _non_empty_string(self.name, "name"))
        object.__setattr__(self, "index", _non_empty_string(self.index, "index"))
        object.__setattr__(
            self, "precision", _positive_precision(self.precision, "precision")
        )
        if not isinstance(self.phi, Hyperparameter):
            phi = _finite_real(self.phi, "phi")
            if not 0.0 < phi < 1.0:
                raise ValueError("phi must lie strictly inside (0, 1)")
            object.__setattr__(self, "phi", phi)
        object.__setattr__(self, "graph", canonical_graph(self.graph))


@dataclass(frozen=True)
class MIDAS(_ComposableEffect):
    """A mixed-frequency distributed-lag effect: the caller-supplied ``columns``
    are the HF covariate at each lag, penalised by a random-walk smoothness
    prior over the lag index. Level and slope of the lag curve stay free of the
    smoothing precision (see the S1 design spec)."""

    name: str
    columns: tuple[str, ...]
    precision: float | Hyperparameter = 1.0
    order: int = 2
    ridge: float = 1e-6

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _non_empty_string(self.name, "name"))
        columns = tuple(self.columns)
        for column in columns:
            _non_empty_string(column, "column")
        if self.order not in (1, 2):
            raise ValueError("order must be 1 or 2")
        if len(columns) <= self.order:
            raise ValueError(f"columns must have more than order ({self.order}) entries")
        object.__setattr__(self, "columns", columns)
        object.__setattr__(
            self, "precision", _positive_precision(self.precision, "precision")
        )
        object.__setattr__(self, "ridge", _positive_real(self.ridge, "ridge"))


@dataclass(frozen=True)
class MIDASParametric(_ComposableEffect):
    """Restricted MIDAS: HF lag ``columns`` collapsed into one regressor
    ``beta * sum_k w(k; theta) * x_{t,k}`` under a parametric lag-weight kernel
    (``"beta"`` or ``"exp_almon"``). The two shape parameters are estimated
    (EB) / integrated (INLA) when given as ``Hyperparameter``s, or fixed when
    given as floats; the loading ``beta`` carries a fixed vague Gaussian prior
    (``prior_precision``). See the S2 design spec."""

    name: str
    columns: tuple[str, ...]
    kernel: str = "beta"
    shape1: "float | Hyperparameter | None" = None
    shape2: "float | Hyperparameter | None" = None
    prior_precision: float = 1e-6

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _non_empty_string(self.name, "name"))
        columns = tuple(self.columns)
        for column in columns:
            _non_empty_string(column, "column")
        if len(columns) < 2:
            raise ValueError("MIDASParametric requires at least 2 lag columns")
        object.__setattr__(self, "columns", columns)
        if self.kernel not in ("beta", "exp_almon"):
            raise ValueError("kernel must be 'beta' or 'exp_almon'")
        object.__setattr__(self, "prior_precision", _positive_real(self.prior_precision, "prior_precision"))
        object.__setattr__(self, "shape1", self._resolve_shape(self.shape1, 1))
        object.__setattr__(self, "shape2", self._resolve_shape(self.shape2, 2))

    def _resolve_shape(self, shape, index):
        if shape is None:
            if self.kernel == "beta":
                return Hyperparameter(f"{self.name}.shape{index}", initial=2.0, transform="log")
            initial = 0.0 if index == 1 else -0.1
            return Hyperparameter(f"{self.name}.shape{index}", initial=initial, transform="identity")
        if isinstance(shape, Hyperparameter):
            return shape
        if type(shape) in (int, float):
            return float(shape)
        raise ValueError("shape must be a float, a Hyperparameter, or None")


@dataclass(frozen=True)
class SpaceTime(_ComposableEffect):
    """A Knorr-Held space-time interaction effect (interaction types I-IV)."""

    name: str
    space: str
    time: str
    graph: Mapping | None = None
    interaction: str = "IV"
    order: int = 1
    precision: float | Hyperparameter = 1.0
    scale: bool = True

    def __post_init__(self) -> None:
        from pylgm.effects.graph import canonical_graph

        object.__setattr__(self, "name", _non_empty_string(self.name, "name"))
        object.__setattr__(self, "space", _non_empty_string(self.space, "space"))
        object.__setattr__(self, "time", _non_empty_string(self.time, "time"))
        if self.interaction not in ("I", "II", "III", "IV"):
            raise ValueError("interaction must be one of 'I', 'II', 'III', 'IV'")
        if type(self.order) is not int or self.order not in (1, 2):
            raise ValueError("order must be 1 or 2")
        object.__setattr__(
            self, "precision", _positive_precision(self.precision, "precision")
        )
        if not isinstance(self.scale, bool):
            raise ValueError("scale must be a boolean")
        if self.interaction in ("III", "IV") and self.graph is None:
            raise ValueError(
                f"interaction {self.interaction!r} needs a spatial neighbour graph; "
                "pass graph=..."
            )
        if self.graph is not None:
            object.__setattr__(self, "graph", canonical_graph(self.graph))

    @property
    def index(self) -> str:
        return self.space


@dataclass(frozen=True)
class Copy(_ComposableEffect):
    """A second occurrence of an existing latent field, at a different index.

    ``Copy("u", index="j", scale=beta)`` adds ``beta * A_j`` to the columns of
    the block named ``u``, so the field ``u`` enters the predictor twice: once
    at its own index and once at ``j``, scaled. This is R-INLA's
    ``f(j, copy="u", hyper=list(beta=...))``.

    It produces no block of its own -- there is one latent field, entering
    twice -- so ``name`` is the **target** block's name, and every compiler
    site that reads ``effect.name`` then refers to the block this copy feeds.

    ``scale`` may be a ``Hyperparameter``, which makes the target block's design
    depend on it; the compiler registers it as a ParametricDesignBlock.
    """

    name: str
    index: str
    scale: float | Hyperparameter = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _non_empty_string(self.name, "name"))
        object.__setattr__(self, "index", _non_empty_string(self.index, "index"))
        if not isinstance(self.scale, (int, float, Hyperparameter)) or isinstance(
            self.scale, bool
        ):
            raise TypeError(
                f"Copy scale must be a real number or a Hyperparameter, got "
                f"{type(self.scale).__name__}"
            )


@dataclass(frozen=True)
class Weighted(_ComposableEffect):
    """An indexed latent effect modulated by a numeric column.

    The design becomes ``diag(by) A`` instead of the plain incidence ``A``, so
    the effect's contribution to the predictor is ``by_i * u_{index(i)}``. This
    is R-INLA's ``f(index, weights, model=...)``, and it is what a
    spatially-varying coefficient needs: a covariate whose effect varies over a
    latent field.

    Precision, labels and constraints are the inner effect's, untouched --
    weighting changes how the field enters the predictor, not the field itself.
    """

    effect: object
    by: str

    def __post_init__(self) -> None:
        if isinstance(self.effect, Copy):
            raise TypeError(
                "Weighted cannot wrap a Copy: a copy is a term referencing "
                "another term, not an indexed effect of its own. Weight the "
                "target effect instead."
            )
        if isinstance(self.effect, Weighted):
            raise TypeError(
                "Weighted effect is already weighted; two weight columns on one "
                "block is their product, so multiply them into a single column"
            )
        # Replicated has no `index` of its own -- same reason Weighted itself
        # doesn't (see Replicated.__post_init__): giving it one would silently
        # make joint.Shared's `hasattr(effect, "index")` gate accept
        # Shared(Replicated(...)), which is not supported. So it is named
        # explicitly here rather than caught by the generic hasattr check.
        if not hasattr(self.effect, "index") and not isinstance(self.effect, Replicated):
            raise TypeError(
                f"Weighted requires an indexed effect, got "
                f"{type(self.effect).__name__}, which has no index. A Fixed effect "
                "builds its design from a formula -- multiply the covariate into "
                "the formula instead."
            )
        object.__setattr__(self, "by", _non_empty_string(self.by, "by"))

    @property
    def name(self) -> str:
        return self.effect.name


@dataclass(frozen=True)
class Replicated(_ComposableEffect):
    """``R`` independent copies of an effect, sharing every hyperparameter.

    ``Replicated(AR1("t", index="year"), over="firm")`` is one AR1 series per
    firm: the firms share ``rho`` and ``precision`` but not their realizations.
    This is R-INLA's ``f(index, model=..., replicate=r)``.

    The precision becomes ``I_R (x) Q``, the design is indexed on
    ``(replicate, level)`` pairs, and a constrained inner effect gets **one
    constraint per replicate** -- a single shared constraint would leave
    ``R-1`` directions unidentified while still fitting.

    Not to be confused with R-INLA's ``group``, which is *correlated* copies
    with a between-group structure; that is a separate modifier.
    """

    effect: object
    over: str

    def __post_init__(self) -> None:
        if isinstance(self.effect, Replicated):
            raise TypeError(
                "Replicated effect is already replicated; two replicate columns "
                "is one replicate over their cross product, so combine them into "
                "a single column"
            )
        if getattr(self.effect, "replicate", None) is not None:
            raise TypeError(
                f"{type(self.effect).__name__} already replicates itself through "
                "its own `replicate` argument; wrapping it would give two "
                "replication mechanisms on one effect with no defined interaction"
            )
        # Resolve the index THROUGH a Weighted wrapper rather than giving
        # Weighted an `index` of its own: joint.Shared distinguishes "wrapper,
        # cannot be shared" from "no index at all" by hasattr(effect, "index"),
        # so an index on Weighted turns that guard into dead code. Same unwrap
        # pattern as compiler._build_effect_block and data.spark._required_columns.
        target = self.effect.effect if isinstance(self.effect, Weighted) else self.effect
        if not hasattr(target, "index"):
            raise TypeError(
                f"Replicated requires an indexed effect, got "
                f"{type(self.effect).__name__}, which has no index."
            )
        if isinstance(self.effect, Copy):
            raise TypeError(
                "Replicated cannot wrap a Copy: a copy is a term referencing "
                "another term, not an indexed effect of its own. Replicate the "
                "target effect instead."
            )
        object.__setattr__(self, "over", _non_empty_string(self.over, "over"))

    @property
    def name(self) -> str:
        return self.effect.name


EffectSpec: TypeAlias = (
    Fixed
    | IID
    | RW1
    | RW2
    | AR1
    | Seasonal
    | Besag
    | ProperCAR
    | SAR
    | DynamicSpatialPanel
    | BYM2
    | Copy
    | MIDAS
    | MIDASParametric
    | SpaceTime
    | Weighted
    | Replicated
)


@dataclass(frozen=True)
class Predictor:
    """An ordered, immutable collection of declarative latent effects."""

    effects: tuple[EffectSpec, ...]

    def __post_init__(self) -> None:
        try:
            effects = tuple(self.effects)
        except TypeError as error:
            raise TypeError("effects must be an iterable of effect specifications") from error
        if any(not isinstance(effect, _ComposableEffect) for effect in effects):
            offenders = sorted(
                {type(e).__name__ for e in effects if not isinstance(e, _ComposableEffect)}
            )
            raise TypeError(
                f"effects must contain only effect specifications; got {offenders}"
            )
        # Copy effects reference existing blocks and are not blocks themselves,
        # so they may share names with their targets. Only check uniqueness among
        # non-Copy effects: each block should appear exactly once.
        non_copy_names = [effect.name for effect in effects if not isinstance(effect, Copy)]
        if len(non_copy_names) != len(set(non_copy_names)):
            raise ValueError("effect names must be unique")
        object.__setattr__(self, "effects", effects)

    def __add__(self, other: object) -> "Predictor":
        if isinstance(other, Predictor):
            return Predictor(self.effects + other.effects)
        if isinstance(other, _ComposableEffect):
            return Predictor(self.effects + (other,))
        return NotImplemented


__all__ = [
    "AR1",
    "Seasonal",
    "Besag",
    "BYM2",
    "Copy",
    "DynamicSpatialPanel",
    "Fixed",
    "IID",
    "MIDAS",
    "MIDASParametric",
    "Predictor",
    "ProperCAR",
    "Replicated",
    "RW1",
    "RW2",
    "SAR",
    "SpaceTime",
    "Weighted",
]
