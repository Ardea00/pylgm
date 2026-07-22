import numpy as np
from scipy.sparse import block_diag, hstack

from pylgm.config import RunConfig
from pylgm.data import CanonicalPanel
from pylgm.effects import build_fixed, build_iid, build_random_walk
from pylgm.ir.model import CompiledLGM, LatentBlock


def _structured_blocks(config: RunConfig, panel: CanonicalPanel) -> list[LatentBlock]:
    blocks: list[LatentBlock] = []
    for effect in config.model.effects:
        if effect.type == "iid":
            blocks.append(
                build_iid(panel.frame, effect.name, effect.index, effect.precision)
            )
        else:
            order = 1 if effect.type == "rw1" else 2
            blocks.append(
                build_random_walk(
                    panel.frame,
                    effect.name,
                    effect.index,
                    effect.precision,
                    order,
                )
            )
    return blocks


def compile_model(config: RunConfig, panel: CanonicalPanel) -> CompiledLGM:
    blocks = [
        build_fixed(
            panel.frame,
            config.model.fixed,
            config.model.fixed_prior_precision,
        )
    ]
    blocks.extend(_structured_blocks(config, panel))
    widths = [block.design.shape[1] for block in blocks]
    total = sum(widths)
    constraint_rows: list[np.ndarray] = []
    start = 0
    for block, width in zip(blocks, widths, strict=True):
        for local in block.constraints:
            row = np.zeros(total)
            row[start : start + width] = local
            constraint_rows.append(row)
        start += width
    constraints = (
        np.vstack(constraint_rows) if constraint_rows else np.empty((0, total))
    )
    y = panel.frame[panel.response].fillna(0.0).to_numpy(dtype=float)
    return CompiledLGM(
        y=y,
        observed=panel.observed,
        offset=np.zeros(len(panel.frame)),
        design=hstack([block.design for block in blocks], format="csr"),
        precision=block_diag([block.precision for block in blocks], format="csr"),
        constraints=constraints,
        labels=tuple(
            f"{block.name}:{label}" for block in blocks for label in block.labels
        ),
        sigma=float(config.model.sigma),
        blocks=tuple(blocks),
    )
