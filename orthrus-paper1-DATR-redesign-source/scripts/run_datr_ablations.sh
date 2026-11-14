#!/usr/bin/env bash
set -euo pipefail

# Three-seed attribution study. Run only after the primary configuration has
# been frozen; never select an ablation using final-test performance.
DATASET="${DATASET:-CADETS_E3}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SEEDS="${SEEDS:-0 1 2}"
OUT="${OUT:-runs/PAPER1_DATR_ABLATIONS}"
mkdir -p "$OUT"

for VARIANT in mean_ema score_only tail_only balance_only datr_full; do
  ENABLED=false; TAIL=0.0; BETA=0.0; NORMALIZATION=none
  case "$VARIANT" in
    mean_ema) ;;
    score_only) NORMALIZATION=edge_type_robust ;;
    tail_only) ENABLED=true; TAIL=0.5 ;;
    balance_only) ENABLED=true; BETA=0.999 ;;
    datr_full)
      ENABLED=true; TAIL=0.5; BETA=0.999
      NORMALIZATION=edge_type_robust
      ;;
  esac

  for SEED in $SEEDS; do
    LOG="$OUT/${DATASET}_${VARIANT}_s${SEED}.log"
    echo "===== $VARIANT seed=$SEED =====" | tee "$LOG"
    PYTHONHASHSEED="$SEED" "$PYTHON_BIN" -u src/orthrus.py "$DATASET" \
      --run_from_training --seed="$SEED" \
      --detection.gnn_training.num_epochs=20 \
      --detection.gnn_training.alignment.enabled="$ENABLED" \
      --detection.gnn_training.alignment.tail_weight="$TAIL" \
      --detection.gnn_training.alignment.class_balance_beta="$BETA" \
      --detection.gnn_training.alignment.ema_enabled=true \
      --detection.gnn_training.alignment.checkpoint_save_policy=ema_only \
      --detection.gnn_testing.checkpoint_policy=ema_only \
      --detection.evaluation.checkpoint_selection=fixed_ema \
      --detection.evaluation.node_evaluation.score_normalization="$NORMALIZATION" \
      --exp="P1_DATR_ABL_${VARIANT}_${DATASET}_s${SEED}" \
      2>&1 | tee -a "$LOG"
  done
done
