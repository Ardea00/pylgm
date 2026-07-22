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


class NumericalError(InferenceError, ArithmeticError):
    """Numerical inference produced an invalid or non-finite result."""
