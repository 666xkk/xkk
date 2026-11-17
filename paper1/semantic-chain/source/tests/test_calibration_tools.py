from pathlib import Path

import pytest

from scripts.build_calibration_split import ensure_isolated_output
from scripts.run_calibration_inference import (
    checkpoint_loader_path,
    validate_neighbor_offset,
)


def test_calibration_output_cannot_overwrite_standard_splits(tmp_path):
    standard = [tmp_path / name for name in ("train", "val", "test")]
    with pytest.raises(ValueError, match="must not overwrite"):
        ensure_isolated_output(tmp_path / "val", standard)
    ensure_isolated_output(tmp_path / "calibration_day6", standard)


def test_neighbor_offset_must_equal_training_event_count():
    validate_neighbor_offset(123, 123)
    with pytest.raises(RuntimeError, match="offset mismatch"):
        validate_neighbor_offset(122, 123)


def test_checkpoint_loader_prefers_local_then_shared(tmp_path):
    checkpoint = tmp_path / "model_epoch_20"
    checkpoint.mkdir()
    shared = tmp_path / "shared_neighbor_loader.pkl"
    shared.write_bytes(b"shared")
    assert checkpoint_loader_path(checkpoint) == shared
    local = checkpoint / "neighbor_loader.pkl"
    local.write_bytes(b"local")
    assert checkpoint_loader_path(checkpoint) == local


def test_checkpoint_loader_reports_both_missing_paths(tmp_path):
    checkpoint = tmp_path / "model_epoch_20"
    checkpoint.mkdir()
    with pytest.raises(FileNotFoundError, match="neighbor loader missing"):
        checkpoint_loader_path(checkpoint)
