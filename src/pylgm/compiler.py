import numpy as np
from scipy.sparse import block_diag, hstack

from pylgm.config import RunConfig
from pylgm.data import CanonicalPanel
from pylgm.effects import build_fixed, build_iid, build_random_walk
from pylgm.exceptions import CompilationError, DataContractError
from pylgm.ir.model import CompiledLGM, LatentBlock


def _structured_blocks(config: RunConfig, panel: CanonicalPanel) -> list[LatentBlock]:
    blocks: list[LatentBlock] = []
    frame = panel.frame
    for effect in config.model.effects:
        try:
            if effect.type == "iid":
                block = build_iid(
                    frame, effect.name, effect.index, effect.precision
                )
            else:
                order = 1 if effect.type == "rw1" else 2
                block = build_random_walk(
                    frame,
                    effect.name,
                    effect.index,
                    effect.precision,
                    order,
                )
        except Exception as error:
            raise CompilationError(
                f"failed to compile effect {effect.name!r}: {error}"
            ) from error
        blocks.append(block)
    return blocks


def _qualified_labels(blocks: list[LatentBlock]) -> tuple[str, ...]:
    labels = tuple(
        f"{block.name}:{label}" for block in blocks for label in block.labels
    )
    if len(labels) != len(set(labels)):
        raise CompilationError("duplicate latent labels after qualification")
    return labels


def _validate_required_columns(config: RunConfig, frame_columns: object) -> None:
    columns = set(frame_columns)
    required_data = {*config.data.panel, config.data.time, config.data.response}
    missing_data = sorted(required_data.difference(columns))
    if missing_data:
        raise DataContractError(f"missing configured data columns: {missing_data}")
    missing_indexes = sorted(
        {effect.index for effect in config.model.effects}.difference(columns)
    )
    if missing_indexes:
        raise DataContractError(
            f"missing configured effect index columns: {missing_indexes}"
        )


def compile_model(config: RunConfig, panel: CanonicalPanel) -> CompiledLGM:
    if panel.response != config.data.response:
        raise DataContractError(
            "panel response metadata does not match configuration: "
            f"{panel.response!r} != {config.data.response!r}"
        )
    expected_keys = (*config.data.panel, config.data.time)
    if panel.key_columns != expected_keys:
        raise DataContractError(
            "panel key metadata does not match configuration: "
            f"{panel.key_columns!r} != {expected_keys!r}"
        )
    frame = panel.frame
    _validate_required_columns(config, frame.columns)
    try:
        fixed = build_fixed(
            frame,
            config.model.fixed,
            config.model.fixed_prior_precision,
        )
    except Exception as error:
        raise CompilationError(f"failed to compile fixed formula: {error}") from error
    blocks = [fixed]
    blocks.extend(_structured_blocks(config, panel))
    wrong_rows = [block.name for block in blocks if block.design.shape[0] != len(frame)]
    if wrong_rows:
        raise CompilationError(
            f"latent block design row count does not match the panel: {wrong_rows}"
        )
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
    try:
        y = frame[panel.response].fillna(0.0).to_numpy(dtype=float)
        return CompiledLGM(
            y=y,
            observed=panel.observed,
            offset=np.zeros(len(frame)),
            design=hstack([block.design for block in blocks], format="csr"),
            precision=block_diag([block.precision for block in blocks], format="csr"),
            constraints=constraints,
            labels=_qualified_labels(blocks),
            sigma=float(config.model.sigma),
            blocks=tuple(blocks),
        )
    except (TypeError, ValueError) as error:
        raise CompilationError(f"compiled model is invalid: {error}") from error
