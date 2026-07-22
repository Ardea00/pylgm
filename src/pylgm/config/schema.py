from typing import Literal

from pydantic import BaseModel, ConfigDict, PositiveFloat, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DataConfig(StrictModel):
    time: str
    response: str
    panel: tuple[str, ...] = ()


class EffectConfig(StrictModel):
    name: str
    type: Literal["iid", "rw1", "rw2"]
    index: str
    precision: PositiveFloat = 1.0


class ModelConfig(StrictModel):
    likelihood: Literal["gaussian"] = "gaussian"
    fixed: str = "1"
    fixed_prior_precision: PositiveFloat = 1e-6
    sigma: PositiveFloat
    effects: tuple[EffectConfig, ...] = ()

    @model_validator(mode="after")
    def unique_effect_names(self) -> "ModelConfig":
        names = [effect.name for effect in self.effects]
        if len(names) != len(set(names)):
            raise ValueError("effect names must be unique")
        return self


class RunConfig(StrictModel):
    schema_version: Literal[1]
    data: DataConfig
    model: ModelConfig
