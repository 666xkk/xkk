#!/usr/bin/env bash
set -euo pipefail

# Fair primary experiment for Detection-Aligned Tail Risk (DATR).
# Graphs/embeddings must already exist. Both baseline and DATR use the same
# label-free EMA checkpoint; only the training objective changes.

DATASET="${DATASET:-CADETS_E3}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SEEDS="${SEEDS:-0 1 2 3 4}"
EPOCHS="${EPOCHS:-20}"
OUT="${OUT:-runs/PAPER1_DATR}"

mkdir -p "$OUT"

for METHOD in baseline datr; do
  for SEED in $SEEDS; do
    ENABLED=false
    NORMALIZATION=none
    if [[ "$METHOD" == "datr" ]]; then
      ENABLED=true
      NORMALIZATION=edge_type_robust
    fi

    LOG="$OUT/${DATASET}_${METHOD}_s${SEED}.log"
    echo "===== $METHOD seed=$SEED =====" | tee "$LOG"

    PYTHONHASHSEED="$SEED" "$PYTHON_BIN" -u src/orthrus.py "$DATASET" \
      --run_from_training \
      --seed="$SEED" \
      --detection.gnn_training.num_epochs="$EPOCHS" \
      --detection.gnn_training.alignment.enabled="$ENABLED" \
      --detection.gnn_training.alignment.ema_enabled=true \
      --detection.gnn_training.alignment.checkpoint_save_policy=ema_only \
      --detection.gnn_testing.checkpoint_policy=ema_only \
      --detection.evaluation.checkpoint_selection=fixed_ema \
      --detection.evaluation.node_evaluation.score_normalization="$NORMALIZATION" \
      --exp="P1_DATR_${METHOD}_${DATASET}_s${SEED}" \
      2>&1 | tee -a "$LOG"
  done
done
