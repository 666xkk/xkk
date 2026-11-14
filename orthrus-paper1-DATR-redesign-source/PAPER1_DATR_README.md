# Paper 1 redesign: Detection-Aligned Tail Risk (DATR)

## Motivation

Original ORTHRUS minimizes mean edge-type cross entropy but raises alarms from
the extreme tail of reconstruction loss (node maxima plus a maximum-validation
threshold). Mean validation loss is therefore not a reliable proxy for the
deployed detection operating point. Selecting the epoch by test MCC hides this
mismatch and leaks test labels.

DATR changes the scientific question from "which test epoch is best?" to
"can a label-free, pre-specified final model learn a better benign tail?"

## Method

1. **Rare-event balance.** Effective-number weights prevent frequent benign
   edge types from dominating training. This is important because rare but
   benign system actions otherwise populate the false-positive tail.
2. **CVaR-style tail risk.** The objective blends mean per-event loss with the
   mean of the highest-loss fraction. The training target now matches the part
   of the benign score distribution that determines operational false alarms.
3. **Fixed EMA checkpoint.** An exponential moving average is saved as epoch
   `N+1` and selected before any test label is opened. No validation/test epoch
   search is used in the headline protocol.
4. **Bounded artifacts.** The primary configuration tests and stores only the
   EMA checkpoint, avoiding dozens of redundant neighbor-loader snapshots.
5. **Conditional robust scoring.** Validation-only median/IQR statistics remove
   intrinsic edge-type difficulty before node aggregation. This is label-free
   and must be ablated separately from the training objective.
6. **Seed isolation.** Detector seeds receive distinct training/testing paths,
   while graph construction and edge embeddings are intentionally shared.

The method does **not** guarantee improved MCC. Its falsifiable hypothesis is:
tail-aligned benign training reduces false positives without sacrificing too
much attack recall. Report negative results if this trade-off is not observed.

## Primary fair comparison

Database credentials can be supplied without editing source code:

```bash
export ORTHRUS_DB_HOST=127.0.0.1
export ORTHRUS_DB_USER=postgres
export ORTHRUS_DB_PASSWORD='your-password'
export ORTHRUS_DB_PORT=5432
```

Run the original mean objective and DATR with the same seeds and the same fixed
EMA protocol:

```bash
cd /path/to/orthrus-main
chmod +x scripts/run_datr_protocol.sh
PYTHON_BIN=python DATASET=CADETS_E3 bash scripts/run_datr_protocol.sh
```

Existing graph construction and edge embeddings are reused. To perform the
first run from raw data, omit `--run_from_training` in the script once; all
later runs should reuse the exact same preprocessing artifacts.

The primary `baseline` row is the original mean objective with the same EMA
selection protocol and no conditional normalization. The `datr` row enables
both aligned tail training and robust conditional scoring. Use the ablation
script below to attribute any gain:

```bash
PYTHON_BIN=python DATASET=CADETS_E3 bash scripts/run_datr_ablations.sh
```

## Required ablations

Use the same fixed EMA protocol for every row:

| Variant | enabled | tail_weight | class_balance_beta |
|---|---:|---:|---:|
| Mean-risk baseline + EMA | false | any | any |
| Class balance only | true | 0.0 | 0.999 |
| Tail risk only | true | 0.5 | 0.0 |
| DATR full | true | 0.5 | 0.999 |

Also compare `score_normalization=none` and `edge_type_robust` for both the
baseline and DATR. This separates representation improvements from a change in
the decision score.

Do not choose a variant after viewing final test labels. Freeze the primary row
on a development dataset/timeline and report all rows.

## Sensitivity (development timeline only)

- `tail_fraction`: 0.05, 0.10, 0.20
- `tail_weight`: 0.25, 0.50, 0.75
- `ema_decay`: 0.99, 0.995, 0.999

Do not conduct a full Cartesian grid on the final test period.

## Headline reporting

For each method and seed report MCC, precision, recall, FPR, AP, AUC, TP, FP,
TN, FN, training time, peak GPU memory, and artifact size. The primary result
is `[protocol] FINAL (fixed_ema:...)`. A run with
`checkpoint_policy=all` and `checkpoint_selection=test_mcc_oracle` is allowed
only as a clearly labelled diagnostic upper bound.

Evaluate at least CADETS_E3 and THEIA_E3. A top-tier claim additionally needs
cross-environment transfer or another detector/dataset; five seeds on one TC
trace are not enough evidence of general unknown-threat detection.

## Reproducibility checks

```bash
python -m py_compile \
  src/objectives.py src/model.py src/factory.py \
  src/detection/orthrus_gnn_training.py \
  src/detection/orthrus_gnn_testing.py \
  src/detection/evaluation.py

pytest -q tests/test_detection_aligned_objective.py
```

## Important interpretation limits

- DATR trains on benign data; it does not use attack labels.
- EMA is a selection protocol, not evidence of a better representation by
  itself; therefore both baseline and DATR must use EMA.
- Optimizing benign tail risk may reduce anomaly separation. Recall and AP
  must be reported alongside FPR.
- The test-MCC oracle must never be the headline number.
