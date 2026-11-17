import math
import subprocess
import sys
import csv
from pathlib import Path

from src.detection.semantic_chain import (
    CanonicalEvent,
    MotifModel,
    NodeSemantic,
    RobustLocationScale,
    SemanticChainConfig,
    build_catalog,
    canonicalize_event,
    categorize_file,
    categorize_subject,
    fit_model,
    parse_interval,
    score_events,
    validate_split_intervals,
    validate_chronology,
)
from src.detection.semantic_chain_evaluate import read_frozen_rows


def event(time, actor, token, sequence=0):
    return CanonicalEvent(time, actor, 99, token[0], token, sequence)


def test_help_does_not_import_training_stack():
    result = subprocess.run(
        [sys.executable, "-m", "src.detection.semantic_chain", "--help"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--calibration-day" in result.stdout


def test_categories_are_bounded_and_do_not_retain_identity():
    assert categorize_file("/tmp/secret-unique-name") == "temporary"
    assert categorize_file("/etc/very-unique.conf") == "configuration"
    assert categorize_file("") == "unknown"
    assert categorize_subject("/usr/bin/python2.7", "python2.7 x.py") == "interpreter"
    assert categorize_subject("", "com.android.browser") == "browser"


def test_catalog_matches_verified_database_column_shapes():
    catalog = build_catalog({
        "subject": [(1, "", "bash")],
        "file": [(2, "/tmp/x")],
        "netflow": [(3, "10.0.0.1", "1234", "8.8.8.8", "53")],
    })
    assert catalog[1].role == "subject"
    assert catalog[2] == NodeSemantic("file", "temporary")
    assert catalog[3].category.endswith("_dns")


def test_canonicalization_uses_subject_as_actor_in_both_directions():
    catalog = {
        1: NodeSemantic("subject", "shell"),
        2: NodeSemantic("file", "temporary"),
    }
    forward = canonicalize_event((1, "EVENT_WRITE", 2, 10), catalog)
    reverse = canonicalize_event((2, "EVENT_READ", 1, 20), catalog)
    assert forward.actor_id == reverse.actor_id == 1
    assert forward.token[1] == "subject_as_src"
    assert reverse.token[1] == "subject_as_dst"


def test_non_subject_edge_is_not_forced_into_a_chain():
    catalog = {
        1: NodeSemantic("file", "home"),
        2: NodeSemantic("netflow", "public_web"),
    }
    assert canonicalize_event((1, "EVENT_SEND", 2, 10), catalog) is None


def test_seen_motif_is_less_surprising_than_unseen_motif():
    a = ("READ", "subject_as_src", "file", "configuration")
    b = ("WRITE", "subject_as_src", "file", "temporary")
    c = ("CONNECT", "subject_as_src", "netflow", "public_web")
    model = MotifModel()
    history = []
    for token in [a, b] * 50:
        model.observe(token, history)
        history = (history + [token])[-2:]
    assert model.surprisal(a, [a, b]) < model.surprisal(c, [a, b])


def test_disjoint_sequences_do_not_create_cross_day_transition():
    a = ("A", "subject_as_src", "file", "other_file")
    b = ("B", "subject_as_src", "file", "other_file")
    model = fit_model(
        [event(1, 1, a, 0), event(2, 1, b, 1)],
        SemanticChainConfig(),
    )
    assert model.pairs[(a, b)] == 0


def test_future_append_cannot_change_frozen_window_prefix():
    token = ("READ", "subject_as_src", "file", "configuration")
    model = fit_model([event(i, 10, token) for i in range(1, 20)], SemanticChainConfig())
    calibration = {"READ": RobustLocationScale(0.0, 1.0, 1), "__GLOBAL__": RobustLocationScale(0.0, 1.0, 1)}
    config = SemanticChainConfig(window_minutes=1, review_budget=1)
    prefix = [event(1_000_000_000, 1, token), event(61_000_000_000, 1, token)]
    before, _ = score_events(prefix, model, calibration, config)
    after, _ = score_events(prefix + [event(121_000_000_000, 1, token)], model, calibration, config)
    assert [row["semantic_chain_score"] for row in after[:2]] == [
        row["semantic_chain_score"] for row in before
    ]


def test_attack_interval_parser_uses_explicit_timezone():
    start, end = parse_interval(
        "2018-04-11 13:54:00~2018-04-11 14:48:00", "America/New_York"
    )
    assert end > start


def test_configuration_rejects_nonpositive_decay():
    try:
        SemanticChainConfig(actor_decay_minutes=0).validate()
    except ValueError as error:
        assert "actor_decay_minutes" in str(error)
    else:
        raise AssertionError("nonpositive decay must be rejected")


def test_split_overlap_is_rejected_before_database_scoring():
    try:
        validate_split_intervals({"train": [(0, 10)], "calibration": [(9, 20)], "test": [(20, 30)]})
    except ValueError as error:
        assert "time leakage" in str(error)
    else:
        raise AssertionError("overlapping temporal splits must be rejected")


def test_future_training_day_is_rejected_by_default_protocol():
    try:
        validate_chronology([(20, 30)], [(0, 10)], [(10, 20)])
    except ValueError as error:
        assert "non-causal split" in str(error)
    else:
        raise AssertionError("future training relative to test must be rejected")


def test_frozen_evaluator_reads_selection_without_rescoring(tmp_path):
    path = tmp_path / "windows.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["window_start_ns", "window_end_ns", "semantic_chain_score", "selected_for_review"],
        )
        writer.writeheader()
        writer.writerow({
            "window_start_ns": 0,
            "window_end_ns": 10,
            "semantic_chain_score": 3.5,
            "selected_for_review": True,
        })
    rows = read_frozen_rows(path)
    assert rows[0]["semantic_chain_score"] == "3.5"
    assert rows[0]["selected_for_review"] is True
