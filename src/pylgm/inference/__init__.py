from pylgm.inference.gaussian import fit_gaussian, preflight_dense_reference
from pylgm.inference.laplace import fit_laplace
from pylgm.inference.result import GaussianMarginals, GaussianResult, LaplaceResult

__all__ = [
    "GaussianMarginals",
    "GaussianResult",
    "LaplaceResult",
    "fit_gaussian",
    "fit_laplace",
    "preflight_dense_reference",
]
