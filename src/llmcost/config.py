from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass(frozen=True, slots=True)
class ModelCfg:
    name: str
    backend: Literal["anthropic", "ollama"]
    model: str
    thinking: str | None = None
    effort: str | None = None
    cache: bool = False
    think: bool | None = None
    num_ctx: int | None = None


@dataclass(frozen=True, slots=True)
class RunCfg:
    max_tokens: int = 8192
    seed: int = 42
    api_concurrency: int = 4
    code_timeout_s: float = 60.0
    idle_seconds: float = 30.0


@dataclass(frozen=True, slots=True)
class Economics:
    laptop_price_usd: float
    lifetimes_years: list[int]
    electricity_usd_per_kwh: list[float]
    wear_usd_per_hour: float = 0.0

    @property
    def base_electricity(self) -> float:
        return self.electricity_usd_per_kwh[len(self.electricity_usd_per_kwh) // 2]


@dataclass(frozen=True, slots=True)
class Config:
    run: RunCfg
    economics: Economics
    hardware: dict
    models: list[ModelCfg] = field(default_factory=list)

    def model(self, name: str) -> ModelCfg:
        for m in self.models:
            if m.name == name:
                return m
        raise KeyError(f"model '{name}' not in config; available: {[m.name for m in self.models]}")


def load_config(path: Path) -> Config:
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    names = [m["name"] for m in raw["models"]]
    if len(names) != len(set(names)):
        raise ValueError("model names in config must be unique")
    return Config(
        run=RunCfg(**raw.get("run", {})),
        economics=Economics(**raw["economics"]),
        hardware=raw.get("hardware", {}),
        models=[ModelCfg(**m) for m in raw["models"]],
    )


@dataclass(frozen=True, slots=True)
class ModelPrice:
    input: float
    output: float
    cache_write_5m: float
    cache_read: float


@dataclass(frozen=True, slots=True)
class Prices:
    fixed_on: str
    source: str
    batch_discount: float
    models: dict[str, ModelPrice]

    def cost_usd(
        self, model: str, input_tokens: int, output_tokens: int, cache_write: int = 0, cache_read: int = 0
    ) -> tuple[float, float]:
        p = self.models[model]
        standard = (
            input_tokens * p.input
            + output_tokens * p.output
            + cache_write * p.cache_write_5m
            + cache_read * p.cache_read
        ) / 1_000_000
        return standard, standard * (1 - self.batch_discount)


def load_prices(path: Path) -> Prices:
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    return Prices(
        fixed_on=raw["fixed_on"],
        source=raw["source"],
        batch_discount=raw["batch_discount"],
        models={k: ModelPrice(**v) for k, v in raw["models"].items()},
    )
