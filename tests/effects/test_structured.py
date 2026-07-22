import numpy as np
import pandas as pd

from pylgm.effects import build_iid, build_random_walk


def test_iid_uses_sorted_levels() -> None:
    frame = pd.DataFrame({"region": ["B", "A", "B"]})
    block = build_iid(frame, "region_effect", "region", 2.0)
    assert block.labels == ("A", "B")
    np.testing.assert_allclose(block.design.toarray(), [[0, 1], [1, 0], [0, 1]])
    np.testing.assert_allclose(block.precision.toarray(), np.eye(2) * 2.0)


def test_rw2_has_rank_two_null_space_constraints(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "pylgm.effects.random_walk.np.eye",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("random-walk construction must remain sparse")
        ),
        raising=False,
    )
    frame = pd.DataFrame({"month": [1, 2, 3, 4]})
    block = build_random_walk(frame, "trend", "month", 3.0, order=2)
    assert block.precision.shape == (4, 4)
    assert block.constraints.shape == (2, 4)
    assert np.linalg.matrix_rank(block.constraints) == 2
    np.testing.assert_allclose(block.constraints[1], [-1.5, -0.5, 0.5, 1.5])
    np.testing.assert_allclose(block.constraints @ block.precision.toarray(), 0.0, atol=1e-12)
