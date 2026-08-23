"""Every name a package advertises in __all__ must actually be importable.

Ruff does not enforce F822 inside __init__.py, so an __all__ entry whose import
was never added passes lint and the whole suite, and only fails for a user
writing `from pylgm.x import *`.
"""

import importlib

import pytest

PACKAGES = [
    "pylgm",
    "pylgm.effects",
    "pylgm.inference",
    "pylgm.ir",
    "pylgm.optimization",
]


@pytest.mark.parametrize("name", PACKAGES)
def test_all_exports_resolve(name):
    module = importlib.import_module(name)
    declared = getattr(module, "__all__", None)
    if declared is None:
        pytest.skip(f"{name} declares no __all__")
    missing = [entry for entry in declared if not hasattr(module, entry)]
    assert not missing, f"{name}.__all__ names unimportable symbols: {missing}"
