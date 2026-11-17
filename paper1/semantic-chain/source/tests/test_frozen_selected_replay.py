from pathlib import Path

import pytest

from scripts.run_frozen_selected_replay import (
    checkpoint_name_from_weight,
    validate_replayed_events,
)


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("model_epoch_20_state_dict.pkl", "model_epoch_20"),
        ("model_epoch_7.pkl", "model_epoch_7"),
    ],
)
def test_checkpoint_name_from_weight(filename, expected):
    assert checkpoint_name_from_weight(Path(filename)) == expected


def test_checkpoint_name_rejects_ambiguous_files():
    with pytest.raises(ValueError, match="weight filename"):
        checkpoint_name_from_weight(Path("state_dict.pkl"))


def test_replayed_event_count_must_match():
    validate_replayed_events(123, 123)
    with pytest.raises(RuntimeError, match="neighbor state mismatch"):
        validate_replayed_events(122, 123)
