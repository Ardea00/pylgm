from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


FinitePositiveFloat = Annotated[float, Field(gt=0, allow_inf_nan=False)]


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
    precision: FinitePositiveFloat = 1.0


class ModelConfig(StrictModel):
    likelihood: Literal["gaussian"] = "gaussian"
    fixed: str = "1"
    fixed_prior_precision: FinitePositiveFloat = 1e-6
    sigma: FinitePositiveFloat
    effects: tuple[EffectConfig, ...] = ()

    @model_validator(mode="after")
    def unique_effect_names(self) -> "ModelConfig":
        names = [effect.name for effect in self.effects]
        if len(names) != len(set(names)):
            raise ValueError("effect names must be unique")
        if "fixed" in names:
            raise ValueError("effect name 'fixed' is reserved for the fixed-effect block")
        return self


class RunConfig(StrictModel):
    schema_version: int
    data: DataConfig
    model: ModelConfig

    @field_validator("schema_version", mode="before")
    @classmethod
    def require_schema_version_one(cls, value: object) -> int:
        if type(value) is not int or value != 1:
            raise ValueError("schema_version must be the integer 1")
        return value
