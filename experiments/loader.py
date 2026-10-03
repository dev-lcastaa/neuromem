"""YAML scenario loader."""

from __future__ import annotations

from pathlib import Path

import yaml

from experiments.models import Scenario


def load_scenario(path: Path) -> Scenario:
    with path.open("r", encoding="utf-8") as fp:
        raw = yaml.safe_load(fp)
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: top-level YAML must be a mapping")
    return Scenario.model_validate(raw)


def load_scenarios(directory: Path) -> list[Scenario]:
    scenarios: list[Scenario] = []
    for path in sorted(directory.glob("*.yaml")):
        scenarios.append(load_scenario(path))
    return scenarios
