class PyLGMError(Exception):
    """Base class for typed pyLGM errors."""


class ConfigurationError(PyLGMError):
    """Configuration could not be parsed or validated."""


class DataContractError(PyLGMError):
    """Input data violates the canonical panel contract."""
