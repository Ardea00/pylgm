from __future__ import annotations

import pandas as pd

from pylgm.config.experiment import ExperimentDataConfig
from pylgm.exceptions import CovariateAvailabilityError


def validate_future_covariates(
    future_frame: pd.DataFrame,
    data: ExperimentDataConfig,
    horizon: int,
) -> None:
    """Require every model covariate needed through the target to be known at origin."""
    for name, policy in data.covariates.items():
        if name not in future_frame.columns:
            raise CovariateAvailabilityError(f"missing future covariate column {name!r}")
        if policy.availability == "unavailable_future":
            raise CovariateAvailabilityError(
                f"covariate {name!r} is unavailable for future prediction"
            )
        if policy.availability == "observed_with_lag":
            if policy.lag is None or policy.lag < horizon:
                raise CovariateAvailabilityError(
                    f"covariate {name!r} requires declared lag >= horizon {horizon}"
                )
        if future_frame[name].isna().any():
            raise CovariateAvailabilityError(
                f"covariate {name!r} has no value available at the fold origin"
            )
