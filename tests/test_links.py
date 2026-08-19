import numpy as np

from pylgm.links import IdentityLink, LogLink, LogitLink


def test_log_link_inverse_is_exp():
    link = LogLink()
    assert link.name == "log"
    np.testing.assert_allclose(link.inverse(np.array([0.0, 1.0])), np.exp([0.0, 1.0]))


def test_logit_link_inverse_is_stable_sigmoid():
    link = LogitLink()
    assert link.name == "logit"
    np.testing.assert_allclose(link.inverse(np.array([0.0])), [0.5])
    # large-magnitude inputs must not overflow
    values = link.inverse(np.array([1000.0, -1000.0]))
    np.testing.assert_allclose(values, [1.0, 0.0])


def test_identity_link_unchanged():
    assert IdentityLink().name == "identity"
