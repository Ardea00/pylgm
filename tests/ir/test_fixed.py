import numpy as np
import pandas as pd

from pylgm.effects.fixed import build_fixed


def test_fixed_block_has_stable_columns_and_precision() -> None:
    frame = pd.DataFrame({"x": [2.0, 3.0]})
    block = build_fixed(frame, "1 + x", prior_precision=0.25)
    assert block.name == "fixed"
    assert block.labels == ("Intercept", "x")
    np.testing.assert_allclose(block.design.toarray(), [[1.0, 2.0], [1.0, 3.0]])
    np.testing.assert_allclose(block.precision.toarray(), np.eye(2) * 0.25)
    assert block.constraints.shape == (0, 2)
