#!/usr/bin/env bash
set -euo pipefail

# Single-seed falsification screen. Reuses existing CADETS graph/embeddings.
# Do not expand to more seeds until the pre-registered gates in the research
# plan pass.
DATASET="${DATASET:-CADETS_E3}"
SEED="${SEED:-0}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
OUT="${OUT:-runs/PAPER1_TRACE_OPEN_SCREEN}"

mkdir -p "$OUT"
set -o pipefail

PYTHONHASHSEED="$SEED" python -u src/orthrus.py "$DATASET" \
  --run_from_training \
  --seed="$SEED" \
  --detection.gnn_training.num_epochs=20 \
  --detection.gnn_training.encoder.batch_size="$BATCH_SIZE" \
  --detection.gnn_training.alignment.enabled=false \
  --detection.gnn_training.alignment.uncertainty.enabled=false \
  --detection.gnn_training.alignment.trace.enabled=true \
  --detection.gnn_training.alignment.trace_open.enabled=true \
  --detection.gnn_training.alignment.ema_enabled=true \
  --detection.gnn_training.alignment.ema_decay=0.995 \
  --detection.gnn_training.alignment.ema_start_epoch=2 \
  --detection.gnn_training.alignment.checkpoint_save_policy=ema_only \
  --detection.gnn_testing.checkpoint_policy=ema_only \
  --detection.evaluation.checkpoint_selection=fixed_ema \
  --detection.evaluation.node_evaluation.score_normalization=none \
  --exp="P1_TRACE_OPEN_${DATASET}_s${SEED}" \
  2>&1 | tee "$OUT/${DATASET}_s${SEED}.log"

status=${PIPESTATUS[0]}
echo "trace_open_seed=${SEED} python_exit=${status}" | tee -a "$OUT/${DATASET}_s${SEED}.log"
exit "$status"
