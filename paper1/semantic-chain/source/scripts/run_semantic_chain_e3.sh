#!/usr/bin/env bash
set -euo pipefail

DATASET="${DATASET:-CLEARSCOPE_E3}"
OUT="${OUT:-paper1/results/${DATASET}/SEMANTIC_CHAIN_V1}"
DB_HOST="${DB_HOST:-127.0.0.1}"
DB_PORT="${DB_PORT:-5432}"
DB_USER="${DB_USER:-postgres}"

case "$DATASET" in
  CADETS_E3)
    DATABASE="cadets_e3"
    # The historical ORTHRUS split trains on days 7--10 while also testing
    # day 6.  This detector uses only days 3--5 so every test event remains
    # strictly later than all fitted normal events.
    TRAIN_DAYS=(03 04 05)
    CALIBRATION_DAYS=(02)
    TEST_DAYS=(06 11 12 13)
    ;;
  THEIA_E3)
    DATABASE="theia_e3"
    TRAIN_DAYS=(02 03 04 05)
    CALIBRATION_DAYS=(09)
    TEST_DAYS=(10 12 13)
    ;;
  CLEARSCOPE_E3)
    DATABASE="clearscope_e3"
    TRAIN_DAYS=(03 04 05 07 08 09 10)
    CALIBRATION_DAYS=(02)
    TEST_DAYS=(11 12)
    ;;
  *)
    echo "Unsupported DATASET=$DATASET" >&2
    exit 2
    ;;
esac

mkdir -p "$OUT"
LOG="$OUT/run.log"
ARGS=()
for DAY in "${TRAIN_DAYS[@]}"; do ARGS+=(--train-day "2018-04-$DAY"); done
for DAY in "${CALIBRATION_DAYS[@]}"; do ARGS+=(--calibration-day "2018-04-$DAY"); done
for DAY in "${TEST_DAYS[@]}"; do ARGS+=(--test-day "2018-04-$DAY"); done

set -o pipefail
python -u -m src.detection.semantic_chain \
  --database "$DATABASE" \
  --db-host "$DB_HOST" \
  --db-port "$DB_PORT" \
  --db-user "$DB_USER" \
  --timezone America/New_York \
  "${ARGS[@]}" \
  --actor-top-k 10 \
  --actor-decay-minutes 60 \
  --window-minutes 15 \
  --review-budget 20 \
  --output-dir "$OUT" \
  2>&1 | tee "$LOG"

status=${PIPESTATUS[0]}
echo "semantic_chain dataset=$DATASET exit=$status" | tee -a "$LOG"
exit "$status"
