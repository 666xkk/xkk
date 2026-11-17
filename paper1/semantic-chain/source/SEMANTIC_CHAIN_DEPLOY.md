# Semantic causal-chain detector v2

This module is an offline, label-free detector built directly from the DARPA
TC PostgreSQL event tables. It does not change ORTHRUS training or consume GNN
test labels.

## Protocol

1. Learn unigram, transition and length-three motif counts on normal training
   days.
2. Freeze the model and estimate robust per-operation location/scale on the
   normal validation day.
3. Score test events in timestamp order. The normal model is never updated.
4. Within each 15-minute window, retain only the maximum evidence for each
   distinct `(subject, semantic motif)` pair. Sum the Top-3 distinct motifs per
   subject, then average the Top-10 subjects for the window score.
5. Attack intervals, if supplied manually, are evaluated only after scores and
   selections are frozen.

Version 2 deliberately removes the v1 event-level additive accumulator. That
accumulator allowed a millisecond-scale benign loop to grow evidence without
bound. In v2, repeating an identical actor/motif pair cannot increase semantic
evidence, although raw `event_count` is retained as an explicit activity
baseline.

The runner also rejects nonchronological splits. In particular, CADETS_E3 uses
days 3--5 for training rather than the historical ORTHRUS inclusion of days
7--10, because day 6 is part of the test set. This deliberate difference must
be reported when comparing results.

Raw UUIDs, hashes, full paths, full command lines and IP addresses are never
included in motif tokens. Event endpoints are joined through the verified
global `index_id`; file/process/network metadata is reduced to bounded semantic
categories.

## Install

Extract the release archive at the ORTHRUS repository root. It adds:

- `src/detection/semantic_chain.py`
- `scripts/run_semantic_chain_e3.sh`
- `tests/test_semantic_chain.py`
- this document

The included lazy `src/detection/__init__.py` matches the earlier causal-bridge
release and prevents standalone commands from importing the CUDA stack.

## Verify

```bash
python -m py_compile src/detection/semantic_chain.py
python -m pytest -q tests/test_semantic_chain.py
python -m src.detection.semantic_chain --help
```

## First formal screen

Use ClearScope first because the previous strictly causal collective-window
result provides the clearest comparison. Password is read from `PGPASSWORD`,
not stored in logs or command arguments.

```bash
cd /hy-tmp/orthrus
export PGPASSWORD='YOUR_POSTGRES_PASSWORD'

DATASET=CLEARSCOPE_E3 \
OUT=paper1/results/CLEARSCOPE_E3/SEMANTIC_CHAIN_V2 \
bash scripts/run_semantic_chain_e3.sh
```

Inspect label-free outputs first:

```bash
cat paper1/results/CLEARSCOPE_E3/SEMANTIC_CHAIN_V2/summary.json
head -20 paper1/results/CLEARSCOPE_E3/SEMANTIC_CHAIN_V2/window_scores.csv
head -20 paper1/results/CLEARSCOPE_E3/SEMANTIC_CHAIN_V2/top_evidence.csv
```

Only after those scores are frozen, evaluate the known ClearScope interval:

```bash
python -m src.detection.semantic_chain_evaluate \
  --window-scores paper1/results/CLEARSCOPE_E3/SEMANTIC_CHAIN_V2/window_scores.csv \
  --attack-interval '2018-04-11 13:54:00~2018-04-11 14:48:00' \
  --output paper1/results/CLEARSCOPE_E3/SEMANTIC_CHAIN_V2/frozen_attack_evaluation.json
```

The frozen evaluation reports three equal-budget rankings: semantic chain,
raw event count and unique actor count. Continue to another dataset only if
semantic-chain coverage exceeds both activity baselines; matching an activity
baseline is not evidence of semantic novelty.

Do not tune parameters against attack intervals. First compare fixed defaults
across ClearScope, THEIA and CADETS. Ground-truth attack-window evaluation is a
separate frozen-score audit.

## Representative ClearScope calibration day

The historical day-2 validation split contains only 2,273 edges and is not
representative of the day-11/12 test stream. Day 6 is an unused normal day with
all eight test operation families. Build it as an isolated split without
changing `val_files` or overwriting existing preprocessing:

```bash
python -u scripts/build_calibration_split.py \
  --dataset CLEARSCOPE_E3 \
  --graph-name graph_6 \
  --split-name calibration_day6
```

Then run a selected frozen checkpoint only on that split:

```bash
python -u scripts/run_calibration_inference.py \
  --dataset CLEARSCOPE_E3 \
  --seed 0 \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --split-name calibration_day6 \
  --batch-size 2048 \
  --output-root paper1/results/CLEARSCOPE_E3/DAY6_CALIBRATION_LOSSES
```

The inference tool constructs `full_data` in the exact order
`train + calibration_day6` and rejects the run unless the checkpoint neighbor
loader's `cur_e_id` equals the number of training edges. It does not train the
model or run evaluation/tracing.

## Conditional tail audit

The historical ClearScope validation stream omits four test operation types
and contains only one `EVENT_CONNECT`. Do not use its raw maximum loss as a
representative operational threshold. After generating frozen Day-6 losses for
all seeds, run the label-free conditional-tail audit:

```bash
python -u scripts/audit_conditional_tail.py \
  --bundle-root paper1/results/CLEARSCOPE_E3/VALIDATION_SELECTED_BASELINE \
  --calibration-root paper1/results/CLEARSCOPE_E3/DAY6_CALIBRATION_5SEEDS \
  --output-dir paper1/results/CLEARSCOPE_E3/CONDITIONAL_TAIL_AUDIT
```

Each edge loss is converted to a finite-sample, event-type-conditioned
upper-tail p-value using only Day-6 normal telemetry. Incident edge evidence is
combined per node with the Simes correction, which adjusts for node exposure
while retaining repeated weak evidence. Test labels are loaded only after
scores are frozen and are used solely for AP, AUC and fixed-budget recall.
