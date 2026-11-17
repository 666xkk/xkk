import math

import pytest

from scripts.audit_conditional_tail import (
    MondrianTailCalibrator,
    ranking_metrics,
    simes_p_value,
)


def test_type_conditioning_removes_raw_scale_difference():
    calibrator = MondrianTailCalibrator().fit({
        "LOW": [1.0, 2.0, 3.0, 4.0],
        "HIGH": [10.0, 20.0, 30.0, 40.0],
    })
    assert calibrator.p_value(3.0, "LOW") == pytest.approx(3 / 5)
    assert calibrator.p_value(30.0, "HIGH") == pytest.approx(3 / 5)


def test_add_one_tail_is_finite_and_unseen_type_is_conservative():
    calibrator = MondrianTailCalibrator().fit({"A": [1.0, 2.0]})
    assert calibrator.p_value(100.0, "A") == pytest.approx(1 / 3)
    assert calibrator.p_value(100.0, "UNSEEN") == 1.0


def test_simes_corrects_for_exposure_and_accumulates_repeated_evidence():
    assert simes_p_value([0.01]) == pytest.approx(0.01)
    assert simes_p_value([0.01, 1.0]) == pytest.approx(0.02)
    assert simes_p_value([0.01, 0.01]) == pytest.approx(0.01)
    assert simes_p_value([]) == 1.0


def test_ranking_metrics_reports_fixed_budgets():
    scores = {
        node: {
            "conditional_tail_score": float(10 - node),
            "node_p_value": 10 ** -(10 - node),
        }
        for node in range(10)
    }
    truth = {node: int(node in {0, 9}) for node in range(10)}
    result = ranking_metrics(scores, truth)
    assert result["positives"] == 2
    assert result["hits_at_25"] == 2
    assert math.isfinite(result["ap"])
