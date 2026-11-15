#!/usr/bin/env bash
set -euo pipefail

# Cheap, falsifiable screen before any five-seed run.  Graphs and embeddings
# must already exist. Use seed 0 to compare one uncertainty candidate against
# the fixed-EMA mean-CE baseline. A grid is allowed only after this gate.

DATASET="${DATASET:-CADETS_E3}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SEED="${SEED:-0}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
OUT="${OUT:-runs/PAPER1_ECHO_SCREEN}"
WEIGHT="${WEIGHT:-0.5}"

mkdir -p "$OUT"

TAG="$(printf '%s' "$WEIGHT" | tr '.' '_')"
LOG="$OUT/${DATASET}_echo_w${TAG}_s${SEED}.log"

PYTHONHASHSEED="$SEED" "$PYTHON_BIN" -u src/orthrus.py "$DATASET" \
    --run_from_training \
    --seed="$SEED" \
    --detection.gnn_training.num_epochs="$EPOCHS" \
    --detection.gnn_training.encoder.batch_size="$BATCH_SIZE" \
    --detection.gnn_training.alignment.enabled=false \
    --detection.gnn_training.alignment.uncertainty.enabled=true \
    --detection.gnn_training.alignment.uncertainty.num_heads=3 \
    --detection.gnn_training.alignment.uncertainty.bootstrap_keep_probability=0.8 \
    --detection.gnn_training.alignment.uncertainty.disagreement_weight="$WEIGHT" \
    --detection.gnn_training.alignment.ema_enabled=true \
    --detection.gnn_training.alignment.ema_decay=0.995 \
    --detection.gnn_training.alignment.ema_start_epoch=2 \
    --detection.gnn_training.alignment.checkpoint_save_policy=ema_only \
    --detection.gnn_testing.checkpoint_policy=ema_only \
    --detection.evaluation.checkpoint_selection=fixed_ema \
    --detection.evaluation.node_evaluation.score_normalization=none \
  --exp="P1_ECHO_${DATASET}_w${TAG}_s${SEED}" \
  2>&1 | tee "$LOG"
