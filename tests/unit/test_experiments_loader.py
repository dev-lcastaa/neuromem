from __future__ import annotations

from pathlib import Path

import pytest

from experiments.loader import load_scenario, load_scenarios

VALID_YAML = """\
id: "t-scenario"
title: "Test scenario"
memories:
  - memory_type: semantic
    content: "for {run_id} test content"
    subject: "subj"
    importance: 0.7
queries:
  - id: q1
    question: "for {run_id} what?"
    top_k: 3
    expected_recall_contains_any: ["opensearch"]
    expected_answer_contains: ["opensearch"]
"""


def test_load_scenario_parses_valid_yaml(tmp_path: Path) -> None:
    path = tmp_path / "s.yaml"
    path.write_text(VALID_YAML, encoding="utf-8")

    s = load_scenario(path)

    assert s.id == "t-scenario"
    assert s.title == "Test scenario"
    assert len(s.memories) == 1
    assert s.memories[0].memory_type == "semantic"
    assert s.queries[0].top_k == 3


def test_load_scenario_rejects_extra_fields(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(VALID_YAML + "\nweird_field: 1\n", encoding="utf-8")

    with pytest.raises(Exception):
        load_scenario(path)


def test_load_scenarios_returns_sorted_list(tmp_path: Path) -> None:
    for name in ["b.yaml", "a.yaml", "c.yaml"]:
        (tmp_path / name).write_text(
            VALID_YAML.replace('"t-scenario"', f'"{name}"'), encoding="utf-8"
        )

    ss = load_scenarios(tmp_path)
    assert [s.id for s in ss] == ["a.yaml", "b.yaml", "c.yaml"]


def test_load_scenarios_empty_dir(tmp_path: Path) -> None:
    assert load_scenarios(tmp_path) == []


def test_bundled_scenarios_all_parse() -> None:
    directory = Path(__file__).resolve().parents[2] / "experiments" / "scenarios"
    scenarios = load_scenarios(directory)
    assert len(scenarios) >= 5
    for s in scenarios:
        assert s.id
        assert s.queries, f"{s.id} has no queries"
