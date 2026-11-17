from copy import deepcopy

import pytest

from scripts.build_causal_incidents import (
    cluster_records,
    evaluate_incidents,
    score_clusters,
)


def row(window, score, selected=True, attack=False):
    return {
        "window": window,
        "file": f"{window}.csv",
        "start_ns": window * 10,
        "end_ns": window * 10 + 9,
        "causal_collective_score": score,
        "selected_for_review": selected,
        "is_attack_window": attack,
    }


def test_one_missing_window_is_bridged_but_larger_gap_is_not():
    records = [row(1, 1.0), row(3, 2.0), row(6, 3.0)]
    clusters = cluster_records(records, max_gap=2)
    assert [[item["window"] for item in group] for group in clusters] == [
        [1, 3],
        [6],
    ]


def test_top_three_mean_ranks_incidents_without_labels():
    records = [
        row(1, 5.0), row(2, 4.0), row(3, 3.0), row(10, 3.5),
    ]
    incidents = score_clusters(records, max_gap=2, top_score_count=3)
    assert incidents[0]["first_window"] == 1
    assert incidents[0]["incident_score"] == pytest.approx(4.0)
    assert incidents[0]["incident_rank"] == 1


def test_attack_labels_cannot_change_incident_scores():
    records = [row(1, 2.0), row(2, 3.0), row(10, 1.0)]
    changed = deepcopy(records)
    for item in changed:
        item["is_attack_window"] = not item["is_attack_window"]
    assert score_clusters(records) == score_clusters(changed)


def test_evaluation_is_separate_and_reports_episode_recall():
    records = [
        row(1, 5.0, attack=True),
        row(2, 4.0, attack=True),
        row(10, 3.0, attack=True),
        row(11, 2.0, selected=False, attack=True),
    ]
    incidents = score_clusters(records)
    result = evaluate_incidents(records, incidents, max_gap=2)
    assert result["total_attack_episodes"] == 2
    assert result["covered_attack_episodes"] == 2
    assert result["attack_episode_recall"] == 1.0
