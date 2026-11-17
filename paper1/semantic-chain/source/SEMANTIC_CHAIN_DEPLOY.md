# Semantic causal-chain detector v1

This module is an offline, label-free detector built directly from the DARPA
TC PostgreSQL event tables. It does not change ORTHRUS training or consume GNN
test labels.

## Protocol

1. Learn unigram, transition and length-three motif counts on normal training
   days.
2. Freeze the model and estimate robust per-operation location/scale on the
   normal validation day.
3. Score test events in timestamp order. The normal model is never updated.
4. Accumulate standardized weak evidence causally per subject and rank fixed
   15-minute windows under a fixed Top-20 review budget.
5. Attack intervals, if supplied manually, are evaluated only after scores and
   selections are frozen.

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
OUT=paper1/results/CLEARSCOPE_E3/SEMANTIC_CHAIN_V1 \
bash scripts/run_semantic_chain_e3.sh
```

Inspect label-free outputs first:

```bash
cat paper1/results/CLEARSCOPE_E3/SEMANTIC_CHAIN_V1/summary.json
head -20 paper1/results/CLEARSCOPE_E3/SEMANTIC_CHAIN_V1/window_scores.csv
head -20 paper1/results/CLEARSCOPE_E3/SEMANTIC_CHAIN_V1/top_evidence.csv
```

Only after those scores are frozen, evaluate the known ClearScope interval:

```bash
python -m src.detection.semantic_chain_evaluate \
  --window-scores paper1/results/CLEARSCOPE_E3/SEMANTIC_CHAIN_V1/window_scores.csv \
  --attack-interval '2018-04-11 13:54:00~2018-04-11 14:48:00' \
  --output paper1/results/CLEARSCOPE_E3/SEMANTIC_CHAIN_V1/frozen_attack_evaluation.json
```

Do not tune parameters against attack intervals. First compare fixed defaults
across ClearScope, THEIA and CADETS. Ground-truth attack-window evaluation is a
separate frozen-score audit.
