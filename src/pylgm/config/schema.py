from pathlib import Path
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


def _require_ordinary_number(value: object) -> object:
    if type(value) not in (int, float):
        raise ValueError("hyperparameter must be an ordinary int or float number")
    return value


FinitePositiveFloat = Annotated[
    float,
    BeforeValidator(_require_ordinary_number),
    Field(gt=0, allow_inf_nan=False),
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DataConfig(StrictModel):
    time: str
    response: str
    panel: tuple[str, ...] = ()

    @model_validator(mode="after")
    def disjoint_semantic_roles(self) -> "DataConfig":
        if len(self.panel) != len(set(self.panel)):
            raise ValueError("panel dimensions must be unique")
        roles = (self.time, self.response, *self.panel)
        if len(roles) != len(set(roles)):
            raise ValueError("time, response, and panel semantic roles must be disjoint")
        return self


_SIMPLE_EFFECTS = ("iid", "rw1", "rw2")
_TEMPORAL_EFFECTS = ("ar1", "seasonal")
_MIDAS_EFFECTS = ("midas", "midas_parametric")
# Every optional effect field, and the set each type accepts. Anything set but
# not in a type's set is rejected -- one table instead of per-branch lists, so a
# new effect cannot silently swallow a field that belongs to another.
_OPTIONAL_FIELDS = (
    "index", "space", "time", "unit", "columns", "precision", "graph",
    "graph_file", "graphs", "graph_files", "rho", "phi", "scale", "group",
    "period", "order", "ridge", "kernel", "interaction", "gamma", "eta",
    "over", "effect", "structure",
)


class StructureConfig(StrictModel):
    """The between-group precision of a ``grouped`` effect.

    ``iid`` makes the copies independent, which is exactly ``Replicated``; the
    others tie them together -- ``ar1`` for ordered groups, ``rw1``/``rw2`` for a
    smooth walk over them, ``besag`` for groups with their own adjacency.
    """

    type: Literal["iid", "ar1", "rw1", "rw2", "besag"]
    rho: float | None = None
    graph: dict | None = None
    graph_file: str | None = None

    @model_validator(mode="after")
    def _fields_match_type(self) -> "StructureConfig":
        if (self.rho is None) != (self.type != "ar1"):
            raise ValueError("rho is required for structure type 'ar1' and invalid otherwise")
        graphed = self.graph is not None or self.graph_file is not None
        if graphed != (self.type == "besag"):
            raise ValueError(
                "graph or graph_file is required for structure type 'besag' and invalid otherwise"
            )
        if self.graph is not None and self.graph_file is not None:
            raise ValueError("structure requires at most one of graph or graph_file")
        return self
# `grouped` wraps another effect instead of indexing a column, so its fields are
# the wrapper's own: the group column and the between-group structure.
_WRAPPER_EFFECTS = ("grouped",)
_ALLOWED_FIELDS = {
    "grouped": {"over", "effect", "structure"},
    "iid": {"index", "precision"},
    "rw1": {"index", "precision"},
    "rw2": {"index", "precision"},
    "ar1": {"index", "precision", "rho", "group"},
    "seasonal": {"index", "precision", "period", "ridge"},
    "besag": {"index", "precision", "graph", "graph_file", "scale"},
    "proper_car": {"index", "precision", "graph", "graph_file", "rho"},
    "bym2": {"index", "precision", "graph", "graph_file", "phi"},
    "sar": {"index", "precision", "graph", "graph_file", "rho"},
    "spacetime": {"space", "time", "precision", "graph", "graph_file",
                  "interaction", "order", "scale"},
    "dynamicspatialpanel": {"unit", "time", "graphs", "graph_files", "rho",
                            "gamma", "eta", "precision"},
    "midas": {"columns", "precision", "order", "ridge"},
    "midas_parametric": {"columns", "kernel"},
}


class EffectConfig(StrictModel):
    # Empty only for the inner spec of a `grouped` effect, which takes the
    # wrapper's name; `ModelConfig` rejects an empty name at the top level.
    name: str = ""
    type: Literal["iid", "rw1", "rw2", "besag", "proper_car", "bym2", "sar",
                  "ar1", "seasonal", "spacetime", "dynamicspatialpanel",
                  "midas", "midas_parametric", "grouped"]
    # Most effects are indexed by a single column; MIDAS uses a list of HF lag
    # columns instead, spacetime a (space, time) pair, and dynamicspatialpanel a
    # (unit, time) pair. Which fields each type accepts is _ALLOWED_FIELDS above.
    index: str | None = None
    space: str | None = None
    time: str | None = None
    unit: str | None = None
    # Fixed values only; estimating precision/rho/phi (a Hyperparameter) stays
    # Python-API-only. Spatial effects default precision in the builder, so it is
    # optional here and required for the simple effects via the validator below.
    precision: FinitePositiveFloat | None = None
    graph: dict | None = None
    graph_file: str | None = None
    # DynamicSpatialPanel per-period networks: inline {period: graph} or a
    # {period: filename} mapping (each file loaded via load_graph_file).
    graphs: dict | None = None
    graph_files: dict | None = None
    rho: float | None = None
    phi: float | None = None
    scale: bool | None = None
    # DynamicSpatialPanel temporal (gamma) and spatio-temporal-diffusion (eta)
    # coefficients; both default to 0.0 in the builder.
    gamma: float | None = None
    eta: float | None = None
    # Temporal family: group-wise AR1 panel column (ar1); cycle length (seasonal).
    group: str | None = None
    period: int | None = None
    # MIDAS family: lag columns; RW smoothness order and ridge (midas/seasonal);
    # the parametric lag-weight kernel (midas_parametric). spacetime interaction
    # type. Kernel/interaction shapes stay Python-API-only where estimated.
    columns: tuple[str, ...] | None = None
    # `grouped` family: the group column, the effect being copied, and the
    # precision tying the copies together. The inner spec omits `name`.
    over: str | None = None
    effect: "EffectConfig | None" = None
    structure: StructureConfig | None = None
    order: int | None = None
    ridge: FinitePositiveFloat | None = None
    kernel: str | None = None
    interaction: str | None = None

    @model_validator(mode="after")
    def _fields_match_type(self) -> "EffectConfig":
        present = {f for f in _OPTIONAL_FIELDS if getattr(self, f) is not None}
        bad = present - _ALLOWED_FIELDS[self.type]
        if bad:
            raise ValueError(
                f"fields {sorted(bad)} are not valid for effect type {self.type!r}"
            )
        if self.type in _WRAPPER_EFFECTS:
            return self._validate_wrapper()
        if self.type in _MIDAS_EFFECTS:
            return self._validate_midas()
        if self.type == "spacetime":
            return self._validate_spacetime()
        if self.type == "dynamicspatialpanel":
            return self._validate_dsp()
        # Single-index effects (simple, temporal, spatial).
        if self.index is None:
            raise ValueError(f"index is required for effect type {self.type!r}")
        if self.type in _SIMPLE_EFFECTS:
            return self
        if self.type in _TEMPORAL_EFFECTS:
            if self.type == "seasonal" and self.period is None:
                raise ValueError("period is required for effect type 'seasonal'")
            return self
        # spatial
        if (self.graph is None) == (self.graph_file is None):
            raise ValueError(
                f"spatial effect {self.name!r} requires exactly one of graph or graph_file"
            )
        if self.type in ("proper_car", "sar") and self.rho is None:
            raise ValueError(f"rho is required for effect type {self.type!r}")
        return self

    def _validate_wrapper(self) -> "EffectConfig":
        if self.over is None or self.effect is None or self.structure is None:
            raise ValueError(
                f"effect type {self.type!r} requires over, effect and structure"
            )
        if self.effect.type in _WRAPPER_EFFECTS:
            raise ValueError(f"effect type {self.type!r} must not wrap another wrapper")
        if self.effect.name:
            raise ValueError(
                f"the inner effect of {self.name!r} must not set its own name; "
                "it takes the wrapper's"
            )
        return self

    def _validate_midas(self) -> "EffectConfig":
        if not self.columns:
            raise ValueError(f"columns is required for effect type {self.type!r}")
        if self.type == "midas":
            if self.order is not None and self.order not in (1, 2):
                raise ValueError("order must be 1 or 2")
            return self
        if self.kernel is not None and self.kernel not in ("beta", "exp_almon"):
            raise ValueError("kernel must be 'beta' or 'exp_almon'")
        return self

    def _validate_spacetime(self) -> "EffectConfig":
        # Two index columns; interaction/order/graph-required checks are the
        # SpaceTime spec's job (raised at build time, wrapped as ConfigurationError).
        if self.space is None or self.time is None:
            raise ValueError("space and time are required for effect type 'spacetime'")
        if self.graph is not None and self.graph_file is not None:
            raise ValueError("spacetime effect requires at most one of graph or graph_file")
        return self

    def _validate_dsp(self) -> "EffectConfig":
        # (unit, time) indexed with per-period graphs; rho fixed, gamma/eta/
        # precision optional. rho-range and graph shape are the spec's job.
        if self.unit is None or self.time is None:
            raise ValueError("unit and time are required for effect type 'dynamicspatialpanel'")
        if self.rho is None:
            raise ValueError("rho is required for effect type 'dynamicspatialpanel'")
        if (self.graphs is None) == (self.graph_files is None):
            raise ValueError(
                "dynamicspatialpanel requires exactly one of graphs or graph_files"
            )
        return self


def build_structure(config: StructureConfig, base_dir: Path) -> object:
    """Build the between-group precision of a ``grouped`` effect."""
    from pylgm.effects import (
        AR1Structure, BesagStructure, IIDStructure, RW1Structure, RW2Structure,
        load_graph_file,
    )

    if config.type == "ar1":
        return AR1Structure(config.rho)
    if config.type == "besag":
        graph = load_graph_file(base_dir / config.graph_file) if config.graph_file else config.graph
        return BesagStructure(graph)
    return {"iid": IIDStructure, "rw1": RW1Structure, "rw2": RW2Structure}[config.type]()


def build_effect(config: EffectConfig, base_dir: Path) -> object:
    from pylgm.effects import (
        AR1, IID, RW1, RW2, SAR, Besag, BYM2, DynamicSpatialPanel, Grouped, MIDAS,
        MIDASParametric, ProperCAR, Seasonal, SpaceTime, load_graph_file,
    )

    if config.type == "grouped":
        # The inner spec carries no name of its own; `Grouped.name` is the
        # wrapper's, and that is the name hyperparameters are keyed on.
        inner = build_effect(config.effect.model_copy(update={"name": config.name}), base_dir)
        return Grouped(inner, config.over, build_structure(config.structure, base_dir))

    if config.type in _SIMPLE_EFFECTS:
        # Left unset, precision defaults to 1.0 -- as it does for every other type.
        simple = {"iid": IID, "rw1": RW1, "rw2": RW2}
        precision = 1.0 if config.precision is None else config.precision
        return simple[config.type](config.name, config.index, precision)

    if config.type == "ar1":
        precision = 1.0 if config.precision is None else config.precision
        rho = 0.5 if config.rho is None else config.rho
        return AR1(config.name, config.index, precision, rho, replicate=config.group)
    if config.type == "seasonal":
        precision = 1.0 if config.precision is None else config.precision
        ridge = 1e-6 if config.ridge is None else config.ridge
        return Seasonal(config.name, config.index, config.period, precision, ridge)

    if config.type == "midas":
        precision = 1.0 if config.precision is None else config.precision
        order = 2 if config.order is None else config.order
        ridge = 1e-6 if config.ridge is None else config.ridge
        return MIDAS(config.name, config.columns, precision, order, ridge)
    if config.type == "midas_parametric":
        return MIDASParametric(config.name, config.columns, config.kernel or "beta")

    if config.type == "dynamicspatialpanel":
        precision = 1.0 if config.precision is None else config.precision
        if config.graph_files:
            graphs = {k: load_graph_file(base_dir / v) for k, v in config.graph_files.items()}
        else:
            graphs = config.graphs
        return DynamicSpatialPanel(
            config.name, config.unit, config.time, graphs, config.rho,
            config.gamma or 0.0, config.eta or 0.0, precision,
        )

    graph = load_graph_file(base_dir / config.graph_file) if config.graph_file else config.graph
    precision = 1.0 if config.precision is None else config.precision
    if config.type == "spacetime":
        return SpaceTime(
            config.name, config.space, config.time, graph,
            config.interaction or "IV", config.order or 1, precision,
            True if config.scale is None else config.scale,
        )
    if config.type == "besag":
        scale = True if config.scale is None else config.scale
        return Besag(config.name, config.index, graph, precision, scale)
    if config.type == "proper_car":
        return ProperCAR(config.name, config.index, graph, config.rho, precision)
    if config.type == "sar":
        return SAR(config.name, config.index, graph, config.rho, precision)
    phi = 0.5 if config.phi is None else config.phi
    return BYM2(config.name, config.index, graph, precision, phi)


class ModelConfig(StrictModel):
    likelihood: Literal["gaussian"] = "gaussian"
    fixed: str = "1"
    fixed_prior_precision: FinitePositiveFloat = 1e-6
    sigma: FinitePositiveFloat
    effects: tuple[EffectConfig, ...] = ()

    @model_validator(mode="after")
    def unique_effect_names(self) -> "ModelConfig":
        names = [effect.name for effect in self.effects]
        if not all(names):
            raise ValueError("effect names must not be empty")
        if len(names) != len(set(names)):
            raise ValueError("effect names must be unique")
        if "fixed" in names:
            raise ValueError("effect name 'fixed' is reserved for the fixed-effect block")
        return self


class RunConfig(StrictModel):
    schema_version: int
    data: DataConfig
    model: ModelConfig

    @field_validator("schema_version", mode="before")
    @classmethod
    def require_schema_version_one(cls, value: object) -> int:
        if type(value) is not int or value != 1:
            raise ValueError("schema_version must be the integer 1")
        return value
