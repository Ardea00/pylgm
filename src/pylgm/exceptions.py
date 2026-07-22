class PyLGMError(Exception):
    """Base class for typed pyLGM errors."""


class ConfigurationError(PyLGMError):
    """Configuration could not be parsed or validated."""
