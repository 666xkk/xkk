#!/usr/bin/env bash
set -euo pipefail

# One-candidate falsification screen. Graphs and embeddings must already exist.
# Do not expand to more seeds unless this run clears the documented gate.

DATASET="${DATASET:-CADETS_E3}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SEED="${SEED:-0}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
OUT="${OUT:-runs/PAPER1_TRACE_SCREEN}"

mkdir -p "$OUT"
LOG="$OUT/${DATASET}_trace_s${SEED}.log"

PYTHONHASHSEED="$SEED" "$PYTHON_BIN" -u src/orthrus.py "$DATASET" \
  --run_from_training \
  --seed="$SEED" \
  --detection.gnn_training.num_epochs="$EPOCHS" \
  --detection.gnn_training.encoder.batch_size="$BATCH_SIZE" \
  --detection.gnn_training.alignment.enabled=false \
  --detection.gnn_training.alignment.uncertainty.enabled=false \
  --detection.gnn_training.alignment.trace.enabled=true \
  --detection.gnn_training.alignment.trace.num_heads=3 \
  --detection.gnn_training.alignment.trace.bootstrap_keep_probability=0.8 \
  --detection.gnn_training.alignment.trace.entity_weight=0.25 \
  --detection.gnn_training.alignment.trace.temporal_weight=0.10 \
  --detection.gnn_training.alignment.trace.stats_start_epoch=5 \
  --detection.gnn_training.alignment.ema_enabled=true \
  --detection.gnn_training.alignment.ema_decay=0.995 \
  --detection.gnn_training.alignment.ema_start_epoch=2 \
  --detection.gnn_training.alignment.checkpoint_save_policy=ema_only \
  --detection.gnn_testing.checkpoint_policy=ema_only \
  --detection.evaluation.checkpoint_selection=fixed_ema \
  --detection.evaluation.node_evaluation.score_normalization=none \
  --detection.evaluation.node_evaluation.trace_uncertainty_quantile=0.99 \
  --exp="P1_TRACE_${DATASET}_s${SEED}" \
  2>&1 | tee "$LOG"
