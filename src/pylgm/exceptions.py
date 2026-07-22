from numpy.linalg import LinAlgError


class PyLGMError(Exception):
    """Base class for typed pyLGM errors."""


class ConfigurationError(PyLGMError):
    """Configuration could not be parsed or validated."""


class DataContractError(PyLGMError):
    """Input data violates the canonical panel contract."""


class CompilationError(PyLGMError):
    """Validated configuration and data could not be compiled into model IR."""


class ModelValidationError(PyLGMError, ValueError):
    """An immutable model IR object violates engine-independent invariants."""


class InferenceError(PyLGMError):
    """Inference cannot produce a valid result for a compiled model."""


class NumericalError(InferenceError, LinAlgError, ArithmeticError):
    """Numerical inference produced an invalid or non-finite result."""


class DenseReferenceLimitError(InferenceError):
    """The small/medium dense reference engine's safety limit was exceeded."""


class OptimizationError(PyLGMError):
    """Hyperparameter optimization did not produce a valid optimum."""

    def __init__(
        self,
        message: str,
        *,
        evaluations: int = 0,
        cache_hits: int = 0,
        numerical_failures: tuple[str, ...] = (),
    ) -> None:
        super().__init__(message)
        self._evaluations = int(evaluations)
        self._cache_hits = int(cache_hits)
        self._numerical_failures = tuple(numerical_failures)

    @property
    def evaluations(self) -> int:
        return self._evaluations

    @property
    def cache_hits(self) -> int:
        return self._cache_hits

    @property
    def numerical_failures(self) -> tuple[str, ...]:
        return self._numerical_failures

    def to_dict(self) -> dict[str, object]:
        cause = self.__cause__
        root_cause = (
            None
            if cause is None
            else {"type": type(cause).__name__, "message": str(cause)}
        )
        return {
            "message": str(self),
            "evaluations": self.evaluations,
            "cache_hits": self.cache_hits,
            "numerical_failures": list(self.numerical_failures),
            "root_cause": root_cause,
        }
