#!/usr/bin/env bash
set -euo pipefail

# Leak-free checkpoint-selection screen. Graph construction and featurization
# must already exist; all detector epochs share one neighbor-loader snapshot.
DATASET="${DATASET:-CADETS_E3}"
SEED="${SEED:-0}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
OUT="${OUT:-runs/PAPER1_VALIDATION_SELECTED_BASELINE}"

mkdir -p "$OUT"
LOG="$OUT/${DATASET}_validation_selected_s${SEED}.log"
set -o pipefail

PYTHONHASHSEED="$SEED" python -u src/orthrus.py "$DATASET" \
  --run_from_training \
  --seed="$SEED" \
  --detection.gnn_training.num_epochs="$EPOCHS" \
  --detection.gnn_training.encoder.batch_size="$BATCH_SIZE" \
  --detection.gnn_training.alignment.enabled=false \
  --detection.gnn_training.alignment.uncertainty.enabled=false \
  --detection.gnn_training.alignment.trace.enabled=false \
  --detection.gnn_training.alignment.trace_open.enabled=false \
  --detection.gnn_training.alignment.ema_enabled=false \
  --detection.gnn_training.alignment.save_ema_checkpoint=false \
  --detection.gnn_training.alignment.checkpoint_save_policy=validation_all \
  --detection.gnn_testing.checkpoint_policy=validation_loss \
  --detection.gnn_testing.retain_validation_candidates=false \
  --detection.evaluation.checkpoint_selection=validation_loss \
  --detection.evaluation.node_evaluation.score_normalization=none \
  --exp="P1_VAL_SELECTED_BASELINE_${DATASET}_s${SEED}" \
  2>&1 | tee "$LOG"

status=${PIPESTATUS[0]}
echo "validation_selected_seed=${SEED} python_exit=${status}" | tee -a "$LOG"
exit "$status"
