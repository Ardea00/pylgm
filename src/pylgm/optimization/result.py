from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from pylgm.inference import GaussianResult


def _immutable_float_mapping(values: Mapping[str, float]) -> Mapping[str, float]:
    return MappingProxyType({name: float(value) for name, value in values.items()})


@dataclass(frozen=True, init=False)
class OptimizationDiagnostics:
    converged: bool
    objective: float
    evaluations: int
    cache_hits: int
    elapsed_seconds: float
    initial: Mapping[str, float]
    optimum: Mapping[str, float]
    active_bounds: tuple[str, ...]
    numerical_failures: tuple[str, ...]

    def __init__(
        self,
        *,
        converged: bool,
        objective: float,
        evaluations: int,
        cache_hits: int,
        elapsed_seconds: float,
        initial: Mapping[str, float],
        optimum: Mapping[str, float],
        active_bounds: tuple[str, ...],
        numerical_failures: tuple[str, ...],
    ) -> None:
        object.__setattr__(self, "converged", bool(converged))
        object.__setattr__(self, "objective", float(objective))
        object.__setattr__(self, "evaluations", int(evaluations))
        object.__setattr__(self, "cache_hits", int(cache_hits))
        object.__setattr__(self, "elapsed_seconds", float(elapsed_seconds))
        object.__setattr__(self, "initial", _immutable_float_mapping(initial))
        object.__setattr__(self, "optimum", _immutable_float_mapping(optimum))
        object.__setattr__(self, "active_bounds", tuple(active_bounds))
        object.__setattr__(self, "numerical_failures", tuple(numerical_failures))


@dataclass(frozen=True, init=False)
class EmpiricalBayesResult:
    parameters: Mapping[str, float]
    fit: GaussianResult
    diagnostics: OptimizationDiagnostics

    def __init__(
        self,
        parameters: Mapping[str, float],
        fit: GaussianResult,
        diagnostics: OptimizationDiagnostics,
    ) -> None:
        object.__setattr__(self, "parameters", _immutable_float_mapping(parameters))
        object.__setattr__(self, "fit", fit)
        object.__setattr__(self, "diagnostics", diagnostics)
