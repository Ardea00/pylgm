import pytest

from pylgm.parameters import Hyperparameter


def test_logit_hyperparameter_allows_nonpositive_initial_and_deferred_bounds():
    hp = Hyperparameter("region.rho", initial=0.0, transform="logit")
    assert hp.transform == "logit"
    assert hp.initial == 0.0
    assert hp.lower is None and hp.upper is None


def test_log_hyperparameter_still_positive_only():
    with pytest.raises(ValueError):
        Hyperparameter("tau", initial=-1.0)          # log default
    with pytest.raises(ValueError):
        Hyperparameter("tau", initial=1.0, transform="log", lower=-1.0)


def test_identity_hyperparameter_accepts_nonpositive_initial_with_finite_bounds():
    hp = Hyperparameter("g.precision", initial=-1.0, transform="identity")
    assert hp.initial == pytest.approx(-1.0)
    assert hp.lower == pytest.approx(-11.0)
    assert hp.upper == pytest.approx(9.0)
