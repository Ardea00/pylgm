from dataclasses import dataclass
from datetime import date, datetime
from types import MappingProxyType
from typing import Annotated, Literal, Mapping

from formulaic import Formula
from pydantic import BeforeValidator, Field, field_serializer, field_validator, model_validator

from pylgm.config.schema import (
    DataConfig,
    EffectConfig,
    FinitePositiveFloat,
    ModelConfig,
    StrictModel,
)


def _require_exact_integer(value: object) -> object:
    if type(value) is not int:
        raise ValueError("value must be an integer")
    return value


def _require_ordinary_real(value: object) -> object:
    if type(value) not in (int, float):
        raise ValueError("value must be an ordinary int or float number")
    return value


StrictPositiveInt = Annotated[
    int, BeforeValidator(_require_exact_integer), Field(gt=0)
]
StrictRollingLength = Annotated[
    int, BeforeValidator(_require_exact_integer), Field(ge=2)
]
StrictUnitFloat = Annotated[
    float, BeforeValidator(_require_ordinary_real), Field(ge=0, le=1, allow_inf_nan=False)
]
StrictNonnegativeFloat = Annotated[
    float, BeforeValidator(_require_ordinary_real), Field(ge=0, allow_inf_nan=False)
]


class CovariateConfig(StrictModel):
    availability: Literal["known_future", "observed_with_lag", "unavailable_future"]
    lag: StrictPositiveInt | None = None


class ExperimentDataConfig(DataConfig):
    vintage: str | None = None
    covariates: Mapping[str, CovariateConfig] = Field(
        default_factory=dict, validate_default=True
    )

    @field_validator("covariates")
    @classmethod
    def freeze_covariates(
        cls, value: Mapping[str, CovariateConfig]
    ) -> Mapping[str, CovariateConfig]:
        return MappingProxyType(dict(value))

    @field_serializer("covariates")
    def serialize_covariates(
        self, value: Mapping[str, CovariateConfig]
    ) -> dict[str, CovariateConfig]:
        return dict(value)


class ParameterBounds(StrictModel):
    initial: FinitePositiveFloat
    lower: FinitePositiveFloat
    upper: FinitePositiveFloat

    @model_validator(mode="after")
    def ordered(self) -> "ParameterBounds":
        if not self.lower <= self.initial <= self.upper or self.lower == self.upper:
            raise ValueError("parameter bounds must satisfy lower <= initial <= upper")
        return self


class ModelOverride(StrictModel):
    fixed: str | None = None
    fixed_prior_precision: FinitePositiveFloat | None = None
    sigma: FinitePositiveFloat | None = None
    effects: tuple[EffectConfig, ...] | None = None


class CandidateConfig(StrictModel):
    name: str
    overrides: ModelOverride = Field(default_factory=ModelOverride)
    optimize: Mapping[str, ParameterBounds] | None = None

    @field_validator("optimize")
    @classmethod
    def freeze_optimize(
        cls, value: Mapping[str, ParameterBounds] | None
    ) -> Mapping[str, ParameterBounds] | None:
        return None if value is None else MappingProxyType(dict(value))

    @field_serializer("optimize")
    def serialize_optimize(
        self, value: Mapping[str, ParameterBounds] | None
    ) -> dict[str, ParameterBounds] | None:
        return None if value is None else dict(value)


class HyperparameterConfig(StrictModel):
    strategy: Literal["empirical_bayes"]
    optimize: Mapping[str, ParameterBounds]

    @field_validator("optimize")
    @classmethod
    def freeze_optimize(
        cls, value: Mapping[str, ParameterBounds]
    ) -> Mapping[str, ParameterBounds]:
        return MappingProxyType(dict(value))

    @field_serializer("optimize")
    def serialize_optimize(
        self, value: Mapping[str, ParameterBounds]
    ) -> dict[str, ParameterBounds]:
        return dict(value)


class InferenceConfig(StrictModel):
    engine: Literal["exact_gaussian"]
    hyperparameters: HyperparameterConfig


class OriginConfig(StrictModel):
    last: StrictPositiveInt | None = None
    values: tuple[str | int | date | datetime, ...] | None = None

    @field_validator("values", mode="before")
    @classmethod
    def require_exact_numeric_origins(cls, value: object) -> object:
        if value is not None:
            for origin in value:  # type: ignore[union-attr]
                if isinstance(origin, (bool, float)):
                    raise ValueError("numeric origins must be integers")
        return value

    @model_validator(mode="after")
    def exactly_one_source(self) -> "OriginConfig":
        if (self.last is None) == (self.values is None):
            raise ValueError("origins require exactly one of last or values")
        return self


class WindowConfig(StrictModel):
    type: Literal["expanding", "rolling"] = "expanding"
    length: StrictRollingLength | None = None

    @model_validator(mode="after")
    def rolling_length(self) -> "WindowConfig":
        if self.type == "rolling" and self.length is None:
            raise ValueError("rolling windows require a positive length")
        if self.type != "rolling" and self.length is not None:
            raise ValueError("window length is only valid for rolling windows")
        return self


class EvaluationConfig(StrictModel):
    mode: Literal["latest", "vintage"] = "latest"
    horizons: tuple[StrictPositiveInt, ...]
    origins: OriginConfig
    window: WindowConfig = WindowConfig()
    interval_levels: tuple[StrictUnitFloat, ...] = (0.5, 0.8, 0.95)
    max_abs_coverage_error: StrictUnitFloat = 0.15
    rmse_tie_tolerance: StrictNonnegativeFloat = 1e-9

    @model_validator(mode="after")
    def valid_horizons_and_levels(self) -> "EvaluationConfig":
        if not self.horizons or any(horizon <= 0 for horizon in self.horizons):
            raise ValueError("horizons must be positive")
        if len(self.horizons) != len(set(self.horizons)):
            raise ValueError("horizons must be unique")
        if any(not 0 < level < 1 for level in self.interval_levels):
            raise ValueError("interval levels must be strictly inside (0, 1)")
        return self


class ExperimentConfig(StrictModel):
    schema_version: Literal[2]
    data: ExperimentDataConfig
    model: ModelConfig
    candidates: tuple[CandidateConfig, ...]
    inference: InferenceConfig
    evaluation: EvaluationConfig

    @field_validator("schema_version", mode="before")
    @classmethod
    def exact_version_two(cls, value: object) -> int:
        if type(value) is not int or value != 2:
            raise ValueError("schema_version must be the integer 2")
        return value

    @model_validator(mode="after")
    def valid_candidates_and_vintage_mode(self) -> "ExperimentConfig":
        names = [candidate.name for candidate in self.candidates]
        if not names:
            raise ValueError("candidates must not be empty")
        if len(names) != len(set(names)):
            raise ValueError("candidate names must be unique")
        if (self.evaluation.mode == "vintage") != (self.data.vintage is not None):
            raise ValueError("a vintage column is required exactly when mode is vintage")
        return self


@dataclass(frozen=True)
class ResolvedCandidate:
    name: str
    model: ModelConfig
    optimize: Mapping[str, ParameterBounds]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready, defensive snapshot of this resolved candidate."""
        return {
            "name": self.name,
            "model": self.model.model_dump(mode="json"),
            "optimize": {
                parameter: bounds.model_dump(mode="json")
                for parameter, bounds in sorted(self.optimize.items())
            },
        }


def _validate_formula_availability(data: ExperimentDataConfig, model: ModelConfig) -> None:
    roles = {data.time, data.response, *data.panel}
    missing = set(Formula(model.fixed).required_variables) - roles - set(data.covariates)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"formula variables require availability declarations: {names}")


def _validate_optimized_parameters(
    optimize: Mapping[str, ParameterBounds], model: ModelConfig
) -> None:
    valid_parameters = {"sigma", *(f"{effect.name}.precision" for effect in model.effects)}
    for parameter in optimize:
        if parameter not in valid_parameters:
            if parameter.endswith(".precision"):
                raise ValueError(f"unknown optimized effect: {parameter}")
            raise ValueError(f"unknown optimized parameter: {parameter}")


def resolve_candidates(config: ExperimentConfig) -> tuple[ResolvedCandidate, ...]:
    resolved: list[ResolvedCandidate] = []
    inherited = config.inference.hyperparameters.optimize
    for candidate in config.candidates:
        update = candidate.overrides.model_dump(exclude_none=True)
        if candidate.overrides.effects is not None:
            update["effects"] = candidate.overrides.effects
        model = config.model.model_copy(update=update)
        model = ModelConfig.model_validate(model.model_dump())
        optimize = candidate.optimize if candidate.optimize is not None else inherited
        _validate_formula_availability(config.data, model)
        _validate_optimized_parameters(optimize, model)
        resolved.append(
            ResolvedCandidate(candidate.name, model, MappingProxyType(dict(optimize)))
        )
    return tuple(resolved)
