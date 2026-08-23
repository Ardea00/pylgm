"""Internal, unstable compiled representations.

These objects support pyLGM's compilers and inference engines but are not part
of the top-level compatibility contract. Prefer :class:`pylgm.LGM` or
``pylgm.config.load_model`` for model construction.
"""

from pylgm.ir.family import (
    CompiledFamily,
    CompiledGaussianFamily,
    Hyperparameters,
    ParametricBlock,
    ScalableBlock,
)
from pylgm.ir.model import CompiledLGM, LatentBlock

__all__ = [
    "CompiledFamily",
    "CompiledGaussianFamily",
    "CompiledLGM",
    "Hyperparameters",
    "LatentBlock",
    "ParametricBlock",
    "ScalableBlock",
]
