"""Label-free semantic causal-chain anomaly scoring for provenance events.

The detector is deliberately independent of the ORTHRUS training stack.  It
learns short normal event motifs from training days, calibrates surprisal on a
separate normal validation day, and scores test events in timestamp order.
Attack intervals are optional and are consulted only after window scores and
the fixed review-budget selection have been frozen.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import ipaddress
import json
import math
import os
import random
import statistics
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path, PurePosixPath
from typing import Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo


Token = Tuple[str, str, str, str]
Interval = Tuple[int, int]


@dataclass(frozen=True)
class NodeSemantic:
    role: str
    category: str


@dataclass(frozen=True)
class CanonicalEvent:
    timestamp_ns: int
    actor_id: int
    object_id: int
    operation: str
    token: Token
    sequence: int = 0


@dataclass(frozen=True)
class SemanticChainConfig:
    backoff_strength: float = 5.0
    additive_alpha: float = 0.1
    window_minutes: float = 15.0
    actor_top_k: int = 10
    actor_motif_top_k: int = 3
    review_budget: int = 20
    calibration_reservoir: int = 50_000
    min_operation_calibration: int = 100
    minimum_scale: float = 0.25
    evidence_top_n: int = 200

    def validate(self) -> None:
        positive = {
            "backoff_strength": self.backoff_strength,
            "additive_alpha": self.additive_alpha,
            "window_minutes": self.window_minutes,
            "actor_top_k": self.actor_top_k,
            "actor_motif_top_k": self.actor_motif_top_k,
            "review_budget": self.review_budget,
            "calibration_reservoir": self.calibration_reservoir,
            "min_operation_calibration": self.min_operation_calibration,
            "minimum_scale": self.minimum_scale,
            "evidence_top_n": self.evidence_top_n,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")


def _clean(value: object) -> str:
    text = "" if value is None else str(value).strip()
    return "" if text.lower() in {"", "none", "null", "nan"} else text


def categorize_subject(path: object, command: object) -> str:
    """Return a bounded cross-platform process family, never a raw identity."""
    text = f"{_clean(path)} {_clean(command).split(' ', 1)[0]}".lower()
    name = PurePosixPath(text.split(" ", 1)[0]).name if text else ""
    if any(word in text for word in ("firefox", "chrome", "chromium", "browser")):
        return "browser"
    if name in {"sh", "bash", "dash", "csh", "tcsh", "zsh"} or "/sh " in text:
        return "shell"
    if any(word in name for word in ("python", "perl", "ruby", "java", "node")):
        return "interpreter"
    if any(word in text for word in ("postgres", "mysql", "mongo", "redis")):
        return "database"
    if text.startswith(("/sbin/", "/usr/sbin/", "/system/")) or any(
        word in text for word in ("system_server", "systemd", "init ")
    ):
        return "system_service"
    if text.startswith(("/home/", "/usr/home/", "/data/data/")):
        return "user_application"
    return "unknown" if not text else "other_process"


def categorize_file(path: object) -> str:
    """Coarsen paths so motifs cannot memorize dataset-specific filenames."""
    value = _clean(path).lower()
    if not value:
        return "unknown"
    if value.startswith(("/tmp/", "/var/tmp/", "/run/shm/")):
        return "temporary"
    if "/cache/" in value or "/.cache/" in value:
        return "cache"
    if value.startswith(("/proc/", "/sys/", "/dev/")):
        return "kernel_device"
    if value.startswith("/var/log/") or value.endswith((".log", ".journal")):
        return "log"
    if value.startswith("/etc/") or value.endswith((".conf", ".cfg", ".ini")):
        return "configuration"
    if value.startswith(("/bin/", "/sbin/", "/usr/bin/", "/usr/sbin/")):
        return "executable"
    if value.endswith((".so", ".dll", ".dylib", ".jar")):
        return "library"
    if value.startswith(("/home/", "/usr/home/")):
        return "home"
    if value.startswith("/data/data/"):
        return "application_data"
    return "other_file"


_SERVICE_PORTS = {
    22: "ssh", 25: "mail", 53: "dns", 80: "web", 123: "ntp",
    443: "web", 465: "mail", 587: "mail", 993: "mail",
}


def _address_scope(value: object) -> str:
    try:
        address = ipaddress.ip_address(_clean(value))
    except ValueError:
        return "unknown"
    if address.is_loopback:
        return "loopback"
    if address.is_private:
        return "private"
    if address.is_link_local:
        return "link_local"
    return "public"


def categorize_netflow(src_addr: object, src_port: object, dst_addr: object, dst_port: object) -> str:
    scope = _address_scope(dst_addr)
    try:
        port = int(_clean(dst_port))
    except ValueError:
        port = -1
    service = _SERVICE_PORTS.get(port, "ephemeral" if port >= 49152 else "other")
    return f"{scope}_{service}"


def build_catalog(rows_by_role: Mapping[str, Iterable[Sequence[object]]]) -> Dict[int, NodeSemantic]:
    """Build global index-id metadata; conflicting IDs are rejected."""
    catalog: Dict[int, NodeSemantic] = {}
    for role, rows in rows_by_role.items():
        for row in rows:
            add_catalog_row(catalog, role, row)
    return catalog


def add_catalog_row(catalog: Dict[int, NodeSemantic], role: str, row: Sequence[object]) -> None:
    if role == "subject":
        index_id, path, command = row
        category = categorize_subject(path, command)
    elif role == "file":
        index_id, path = row
        category = categorize_file(path)
    elif role == "netflow":
        index_id, src_addr, src_port, dst_addr, dst_port = row
        category = categorize_netflow(src_addr, src_port, dst_addr, dst_port)
    else:
        raise ValueError(f"unsupported role: {role}")
    key = int(index_id)
    semantic = NodeSemantic(role, category)
    previous = catalog.get(key)
    if previous is not None and previous != semantic:
        raise ValueError(f"node index has conflicting semantics: {key}")
    catalog[key] = semantic


def canonicalize_event(row: Sequence[object], catalog: Mapping[int, NodeSemantic]) -> Optional[CanonicalEvent]:
    """Choose the subject endpoint as actor and preserve its edge position."""
    src_id, operation, dst_id, timestamp_ns = int(row[0]), str(row[1]), int(row[2]), int(row[3])
    src, dst = catalog.get(src_id), catalog.get(dst_id)
    if src is None or dst is None:
        return None
    operation = operation.upper().strip()
    if src.role == "subject":
        actor_id, object_id, obj, direction = src_id, dst_id, dst, "subject_as_src"
    elif dst.role == "subject":
        actor_id, object_id, obj, direction = dst_id, src_id, src, "subject_as_dst"
    else:
        return None
    token: Token = (operation, direction, obj.role, obj.category)
    return CanonicalEvent(timestamp_ns, actor_id, object_id, operation, token)


class MotifModel:
    """Second-order Markov model with hierarchical additive backoff."""

    def __init__(self, alpha: float = 0.1, backoff_strength: float = 5.0):
        self.alpha = float(alpha)
        self.backoff_strength = float(backoff_strength)
        self.tokens: Counter[Token] = Counter()
        self.context1: Counter[Token] = Counter()
        self.pairs: Counter[Tuple[Token, Token]] = Counter()
        self.context2: Counter[Tuple[Token, Token]] = Counter()
        self.triples: Counter[Tuple[Token, Token, Token]] = Counter()
        self.total = 0
        self.vocabulary: set[Token] = set()

    def observe(self, token: Token, history: Sequence[Token]) -> None:
        self.tokens[token] += 1
        self.total += 1
        self.vocabulary.add(token)
        if history:
            self.context1[history[-1]] += 1
            self.pairs[(history[-1], token)] += 1
        if len(history) >= 2:
            context = (history[-2], history[-1])
            self.context2[context] += 1
            self.triples[(history[-2], history[-1], token)] += 1

    def probability(self, token: Token, history: Sequence[Token]) -> float:
        vocabulary = max(1, len(self.vocabulary) + (token not in self.vocabulary))
        p = (self.tokens[token] + self.alpha) / (self.total + self.alpha * vocabulary)
        strength = self.backoff_strength
        if history:
            previous = history[-1]
            p = (self.pairs[(previous, token)] + strength * p) / (self.context1[previous] + strength)
        if len(history) >= 2:
            context = (history[-2], history[-1])
            p = (self.triples[(history[-2], history[-1], token)] + strength * p) / (
                self.context2[context] + strength
            )
        return max(p, 1e-300)

    def surprisal(self, token: Token, history: Sequence[Token]) -> float:
        return -math.log(self.probability(token, history))


def fit_model(events: Iterable[CanonicalEvent], config: SemanticChainConfig) -> MotifModel:
    model = MotifModel(config.additive_alpha, config.backoff_strength)
    histories: MutableMapping[Tuple[int, int], deque[Token]] = defaultdict(lambda: deque(maxlen=2))
    for event in events:
        history = histories[(event.sequence, event.actor_id)]
        model.observe(event.token, history)
        history.append(event.token)
    return model


class Reservoir:
    def __init__(self, size: int, seed: int):
        self.size, self.seen = size, 0
        self.values: List[float] = []
        self.random = random.Random(seed)

    def add(self, value: float) -> None:
        self.seen += 1
        if len(self.values) < self.size:
            self.values.append(float(value))
        else:
            index = self.random.randrange(self.seen)
            if index < self.size:
                self.values[index] = float(value)


@dataclass(frozen=True)
class RobustLocationScale:
    median: float
    scale: float
    samples: int


def _robust_stats(values: Sequence[float], minimum_scale: float) -> RobustLocationScale:
    if not values:
        return RobustLocationScale(0.0, minimum_scale, 0)
    median = statistics.median(values)
    mad = statistics.median(abs(value - median) for value in values)
    return RobustLocationScale(median, max(minimum_scale, 1.4826 * mad), len(values))


def calibrate_model(
    model: MotifModel,
    events: Iterable[CanonicalEvent],
    config: SemanticChainConfig,
) -> Dict[str, RobustLocationScale]:
    reservoirs: Dict[str, Reservoir] = {}
    global_reservoir = Reservoir(config.calibration_reservoir, 0)
    histories: MutableMapping[Tuple[int, int], deque[Token]] = defaultdict(lambda: deque(maxlen=2))
    for event in events:
        history = histories[(event.sequence, event.actor_id)]
        value = model.surprisal(event.token, history)
        reservoirs.setdefault(
            event.operation,
            Reservoir(config.calibration_reservoir, sum(map(ord, event.operation))),
        ).add(value)
        global_reservoir.add(value)
        history.append(event.token)
    result = {
        operation: _robust_stats(reservoir.values, config.minimum_scale)
        for operation, reservoir in reservoirs.items()
    }
    result["__GLOBAL__"] = _robust_stats(global_reservoir.values, config.minimum_scale)
    return result


def calibration_for_operation(
    operation: str,
    calibration: Mapping[str, RobustLocationScale],
    minimum_samples: int,
) -> RobustLocationScale:
    candidate = calibration.get(operation)
    if candidate is None or candidate.samples < minimum_samples:
        return calibration["__GLOBAL__"]
    return candidate


def standardized_excess(
    value: float,
    operation: str,
    calibration: Mapping[str, RobustLocationScale],
    minimum_samples: int = 100,
) -> float:
    stats = calibration_for_operation(operation, calibration, minimum_samples)
    return max(0.0, (float(value) - stats.median) / stats.scale)


def select_indices(scores: Sequence[float], budget: int) -> List[int]:
    return sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i))[: min(budget, len(scores))]


def score_events(
    events: Iterable[CanonicalEvent],
    model: MotifModel,
    calibration: Mapping[str, RobustLocationScale],
    config: SemanticChainConfig,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """Score a stream using distinct actor-motif evidence per window.

    Repetition of the same actor/token pair cannot increase its contribution:
    only the pair's maximum standardized excess is retained.  This prevents a
    high-rate benign loop from masquerading as a semantic causal chain.
    """
    histories: MutableMapping[Tuple[int, int], deque[Token]] = defaultdict(lambda: deque(maxlen=2))
    # window -> actor -> token -> (excess, surprisal, timestamp, object)
    windows: Dict[
        int,
        Dict[Tuple[int, int], Dict[Token, Tuple[float, float, int, int]]],
    ] = defaultdict(lambda: defaultdict(dict))
    event_counts: Counter[int] = Counter()
    window_ns = int(config.window_minutes * 60 * 1e9)

    for event in events:
        actor_key = (event.sequence, event.actor_id)
        history = histories[actor_key]
        raw = model.surprisal(event.token, history)
        excess = standardized_excess(
            raw,
            event.operation,
            calibration,
            config.min_operation_calibration,
        )
        history.append(event.token)

        window = event.timestamp_ns // window_ns
        event_counts[window] += 1
        previous = windows[window][actor_key].get(event.token)
        if previous is None or excess > previous[0]:
            windows[window][actor_key][event.token] = (
                excess,
                raw,
                event.timestamp_ns,
                event.object_id,
            )

    rows: List[Dict[str, object]] = []
    evidence_candidates: List[Tuple[float, int, int, Dict[str, object]]] = []
    serial = 0
    for window, actor_motifs in sorted(windows.items()):
        actor_scores: List[float] = []
        unique_motifs: set[Token] = set()
        actor_motif_pairs = 0
        for actor_key, motif_values in actor_motifs.items():
            actor_motif_pairs += len(motif_values)
            unique_motifs.update(motif_values)
            top_motifs = sorted(
                motif_values.items(),
                key=lambda item: (-item[1][0], item[0]),
            )[: config.actor_motif_top_k]
            actor_score = sum(value[0] for _, value in top_motifs)
            actor_scores.append(actor_score)
            for token, (excess, raw, timestamp_ns, object_id) in top_motifs:
                detail = {
                    "window_start_ns": window * window_ns,
                    "timestamp_ns": timestamp_ns,
                    "actor_id": actor_key[1],
                    "object_id": object_id,
                    "operation": token[0],
                    "direction": token[1],
                    "object_role": token[2],
                    "object_category": token[3],
                    "surprisal": raw,
                    "standardized_excess": excess,
                    "actor_distinct_motif_evidence": actor_score,
                }
                item = (actor_score, serial, actor_key[1], detail)
                serial += 1
                if len(evidence_candidates) < config.evidence_top_n:
                    heapq.heappush(evidence_candidates, item)
                elif item[:3] > evidence_candidates[0][:3]:
                    heapq.heapreplace(evidence_candidates, item)

        tail = sorted(actor_scores, reverse=True)[: config.actor_top_k]
        rows.append({
            "window_start_ns": window * window_ns,
            "window_end_ns": (window + 1) * window_ns,
            "event_count": event_counts[window],
            "actors": len(actor_motifs),
            "unique_motifs": len(unique_motifs),
            "actor_motif_pairs": actor_motif_pairs,
            "semantic_chain_score": sum(tail) / len(tail) if tail else 0.0,
        })
    selection_specs = {
        "selected_for_review": "semantic_chain_score",
        "selected_by_event_count": "event_count",
        "selected_by_actor_count": "actors",
    }
    selections = {
        output_key: set(select_indices([float(row[score_key]) for row in rows], config.review_budget))
        for output_key, score_key in selection_specs.items()
    }
    for index, row in enumerate(rows):
        for output_key, selected in selections.items():
            row[output_key] = index in selected
    evidence_rows = [item[3] for item in sorted(evidence_candidates, reverse=True)]
    return rows, evidence_rows


def parse_time(value: str, timezone: str) -> int:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=ZoneInfo(timezone))
    return int(parsed.timestamp() * 1e9)


def parse_interval(value: str, timezone: str) -> Interval:
    try:
        start, end = value.split("~", 1)
    except ValueError as error:
        raise ValueError("interval must be START~END") from error
    interval = (parse_time(start, timezone), parse_time(end, timezone))
    if interval[1] <= interval[0]:
        raise ValueError("interval end must follow start")
    return interval


def day_interval(value: str, timezone: str) -> Interval:
    start = datetime.fromisoformat(value).replace(tzinfo=ZoneInfo(timezone))
    return int(start.timestamp() * 1e9), int((start + timedelta(days=1)).timestamp() * 1e9)


def load_catalog(connection: object) -> Dict[int, NodeSemantic]:
    queries = {
        "subject": "SELECT index_id, path, cmd FROM subject_node_table",
        "file": "SELECT index_id, path FROM file_node_table",
        "netflow": "SELECT index_id, src_addr, src_port, dst_addr, dst_port FROM netflow_node_table",
    }
    catalog: Dict[int, NodeSemantic] = {}
    for role, query in queries.items():
        print(f"[semantic-chain] loading {role} catalog ...", flush=True)
        cursor = connection.cursor(name=f"semantic_catalog_{role}")
        cursor.itersize = 50_000
        cursor.execute(query)
        try:
            for row in cursor:
                add_catalog_row(catalog, role, row)
        finally:
            cursor.close()
        print(f"[semantic-chain] catalog nodes after {role}: {len(catalog)}", flush=True)
    return catalog


def iter_database_events(
    connection: object,
    catalog: Mapping[int, NodeSemantic],
    intervals: Sequence[Interval],
    cursor_prefix: str,
) -> Iterator[CanonicalEvent]:
    for number, (start_ns, end_ns) in enumerate(intervals):
        print(
            f"[semantic-chain] {cursor_prefix} interval={number} "
            f"start_ns={start_ns} end_ns={end_ns}",
            flush=True,
        )
        cursor = connection.cursor(name=f"{cursor_prefix}_{number}")
        cursor.itersize = 50_000
        cursor.execute(
            "SELECT src_index_id, operation, dst_index_id, timestamp_rec "
            "FROM event_table WHERE timestamp_rec >= %s AND timestamp_rec < %s "
            "ORDER BY timestamp_rec, _id",
            (start_ns, end_ns),
        )
        accepted = 0
        scanned = 0
        try:
            for row in cursor:
                scanned += 1
                event = canonicalize_event(row, catalog)
                if event is not None:
                    # Prevent a motif or decayed state from spanning disjoint
                    # train/test days while retaining the original actor id.
                    accepted += 1
                    yield replace(event, sequence=number)
                if scanned % 1_000_000 == 0:
                    print(
                        f"[semantic-chain] {cursor_prefix} interval={number} "
                        f"scanned={scanned} accepted={accepted}",
                        flush=True,
                    )
        finally:
            cursor.close()
        print(
            f"[semantic-chain] {cursor_prefix} interval={number} complete "
            f"scanned={scanned} accepted={accepted}",
            flush=True,
        )


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _overlaps(start: int, end: int, intervals: Sequence[Interval]) -> bool:
    return any(start < other_end and end > other_start for other_start, other_end in intervals)


def evaluate_frozen_windows(
    rows: List[Dict[str, object]],
    attacks: Sequence[Interval],
    selection_key: str = "selected_for_review",
) -> Dict[str, object]:
    if not attacks:
        return {}
    attack_indices = {
        i for i, row in enumerate(rows)
        if _overlaps(int(row["window_start_ns"]), int(row["window_end_ns"]), attacks)
    }
    selected = {i for i, row in enumerate(rows) if bool(row[selection_key])}
    return {
        "attack_windows": len(attack_indices),
        "covered_attack_windows": len(attack_indices & selected),
        "attack_window_recall": len(attack_indices & selected) / len(attack_indices) if attack_indices else 0.0,
    }


def _collect_intervals(days: Sequence[str], explicit: Sequence[str], timezone: str) -> List[Interval]:
    return sorted([day_interval(day, timezone) for day in days] + [parse_interval(v, timezone) for v in explicit])


def validate_split_intervals(splits: Mapping[str, Sequence[Interval]]) -> None:
    names = list(splits)
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1:]:
            for left in splits[left_name]:
                for right in splits[right_name]:
                    if _overlaps(left[0], left[1], [right]):
                        raise ValueError(
                            f"time leakage: {left_name} interval {left} overlaps "
                            f"{right_name} interval {right}"
                        )


def validate_chronology(train: Sequence[Interval], calibration: Sequence[Interval], test: Sequence[Interval]) -> None:
    latest_reference = max(end for _, end in [*train, *calibration])
    earliest_test = min(start for start, _ in test)
    if latest_reference > earliest_test:
        raise ValueError(
            "non-causal split: a training/calibration interval occurs after "
            "the beginning of the test period"
        )


def run(args: argparse.Namespace) -> Dict[str, object]:
    try:
        import psycopg2
    except ImportError as error:
        raise RuntimeError("psycopg2 is required for database execution") from error

    config = SemanticChainConfig(
        backoff_strength=args.backoff_strength,
        additive_alpha=args.additive_alpha,
        window_minutes=args.window_minutes,
        actor_top_k=args.actor_top_k,
        actor_motif_top_k=args.actor_motif_top_k,
        review_budget=args.review_budget,
        calibration_reservoir=args.calibration_reservoir,
        min_operation_calibration=args.min_operation_calibration,
        minimum_scale=args.minimum_scale,
        evidence_top_n=args.evidence_top_n,
    )
    config.validate()
    train = _collect_intervals(args.train_day, args.train_interval, args.timezone)
    calibration = _collect_intervals(args.calibration_day, args.calibration_interval, args.timezone)
    test = _collect_intervals(args.test_day, args.test_interval, args.timezone)
    if not train or not calibration or not test:
        raise ValueError("train, calibration, and test intervals must all be non-empty")
    validate_split_intervals({"train": train, "calibration": calibration, "test": test})
    if not args.allow_nonchronological_splits:
        validate_chronology(train, calibration, test)

    password = os.environ.get(args.db_password_env)
    connection = psycopg2.connect(
        database=args.database, host=args.db_host, port=args.db_port,
        user=args.db_user, password=password,
    )
    connection.set_client_encoding("UTF8")
    try:
        catalog = load_catalog(connection)
        model = fit_model(iter_database_events(connection, catalog, train, "semantic_train"), config)
        calibration_stats = calibrate_model(
            model,
            iter_database_events(connection, catalog, calibration, "semantic_calibration"),
            config,
        )
        window_rows, evidence_rows = score_events(
            iter_database_events(connection, catalog, test, "semantic_test"),
            model,
            calibration_stats,
            config,
        )
    finally:
        connection.close()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "window_scores.csv", window_rows)
    write_csv(output / "top_evidence.csv", evidence_rows)
    attacks = [parse_interval(value, args.timezone) for value in args.attack_interval]
    summary: Dict[str, object] = {
        "database": args.database,
        "catalog_nodes": len(catalog),
        "normal_events": model.total,
        "motif_vocabulary": len(model.vocabulary),
        "windows": len(window_rows),
        "selected_windows": sum(bool(row["selected_for_review"]) for row in window_rows),
        "config": asdict(config),
        "calibration": {key: asdict(value) for key, value in calibration_stats.items()},
        **evaluate_frozen_windows(window_rows, attacks),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", required=True)
    parser.add_argument("--db-host", default="127.0.0.1")
    parser.add_argument("--db-port", type=int, default=5432)
    parser.add_argument("--db-user", default="postgres")
    parser.add_argument("--db-password-env", default="PGPASSWORD")
    parser.add_argument("--timezone", default="America/New_York")
    for split in ("train", "calibration", "test"):
        parser.add_argument(f"--{split}-day", action="append", default=[])
        parser.add_argument(f"--{split}-interval", action="append", default=[])
    parser.add_argument("--attack-interval", action="append", default=[], help="Evaluation only; START~END")
    parser.add_argument(
        "--allow-nonchronological-splits",
        action="store_true",
        help="Disable the default train/calibration-before-test safety check.",
    )
    parser.add_argument("--backoff-strength", type=float, default=5.0)
    parser.add_argument("--additive-alpha", type=float, default=0.1)
    parser.add_argument("--window-minutes", type=float, default=15.0)
    parser.add_argument("--actor-top-k", type=int, default=10)
    parser.add_argument("--actor-motif-top-k", type=int, default=3)
    parser.add_argument("--review-budget", type=int, default=20)
    parser.add_argument("--calibration-reservoir", type=int, default=50_000)
    parser.add_argument("--min-operation-calibration", type=int, default=100)
    parser.add_argument("--minimum-scale", type=float, default=0.25)
    parser.add_argument("--evidence-top-n", type=int, default=200)
    parser.add_argument("--output-dir", required=True)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
