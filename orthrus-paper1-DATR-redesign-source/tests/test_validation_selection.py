import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

from detection.orthrus_gnn_testing import select_validation_checkpoint


def test_validation_selection_uses_minimum_loss():
    rows = [
        {"checkpoint": "model_epoch_1", "epoch": 1, "validation_mean_event_loss": 0.5},
        {"checkpoint": "model_epoch_2", "epoch": 2, "validation_mean_event_loss": 0.3},
        {"checkpoint": "model_epoch_3", "epoch": 3, "validation_mean_event_loss": 0.4},
    ]
    assert select_validation_checkpoint(rows) == "model_epoch_2"


def test_validation_selection_tie_breaks_to_earlier_epoch():
    rows = [
        {"checkpoint": "model_epoch_8", "epoch": 8, "validation_mean_event_loss": 0.3},
        {"checkpoint": "model_epoch_4", "epoch": 4, "validation_mean_event_loss": 0.3},
    ]
    assert select_validation_checkpoint(rows) == "model_epoch_4"


def test_validation_selection_rejects_empty_candidates():
    try:
        select_validation_checkpoint([])
    except ValueError as error:
        assert "empty" in str(error)
    else:
        raise AssertionError("empty candidates must fail")
