import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "build_multiscale_causal_incidents.py"
SPEC = importlib.util.spec_from_file_location("multiscale_incidents", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def rows(scores, selected, attacks=()):
    return [
        {
            "window": str(index),
            "file": f"w{index}.csv",
            "start_ns": str(index * 10),
            "end_ns": str(index * 10 + 9),
            "causal_collective_score": str(score),
            "selected_for_review": str(index in selected),
            "is_attack_window": str(index in attacks),
        }
        for index, score in enumerate(scores)
    ]


def test_rank_percentiles_are_bounded_and_deterministic():
    ranked = MODULE.rank_percentiles(rows([2.0, 3.0, 1.0], set()))
    assert ranked[1] == (1, 1.0)
    assert ranked[2] == (3, 0.0)


def test_build_seed_union_and_episode_coverage():
    scale_rows = {
        "scale1": rows([5, 4, 3, 2, 1], {0, 4}, attacks={0, 4}),
        "scale4": rows([1, 2, 3, 4, 5], {0, 1}, attacks={0, 4}),
    }
    candidates, incidents, summary, evaluation = MODULE.build_seed(
        0, scale_rows, max_gap=2, top_score_count=3, expected_selected_per_scale=2
    )
    assert {row["window"] for row in candidates} == {0, 1, 4}
    assert summary["intersection_windows"] == 1
    assert summary["union_windows"] == 3
    assert len(incidents) == 2
    assert evaluation["attack_episode_recall"] == 1.0


def test_alignment_mismatch_is_rejected():
    left = rows([2, 1], {0})
    right = rows([2, 1], {0})
    right[1]["file"] = "different.csv"
    with pytest.raises(ValueError, match="alignment differs"):
        MODULE.validate_alignment({"scale1": left, "scale4": right})


def test_selected_budget_mismatch_is_rejected():
    scale_rows = {
        "scale1": rows([2, 1], {0}),
        "scale4": rows([2, 1], {0}),
    }
    with pytest.raises(ValueError, match="expected 20"):
        MODULE.build_seed(
            0, scale_rows, max_gap=2, top_score_count=3,
            expected_selected_per_scale=20,
        )
