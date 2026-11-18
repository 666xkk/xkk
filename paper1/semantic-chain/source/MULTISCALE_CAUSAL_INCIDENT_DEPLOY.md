# Frozen multi-scale causal incidents

Coverage mode unions already frozen scale-1 and scale-4 Top-20 window sets,
normalizes scores by within-scale rank, and merges temporally adjacent windows
into ranked incidents. Labels are optional and are read only after candidate
selection and ranking for the frozen evaluation table.

Efficient mode remains the original scale-4 Top-20 pipeline. Coverage mode is
not an equal-window-budget replacement: it explicitly spends a larger window
candidate budget to preserve short and sustained attacks, then reduces analyst
work through incident merging.

```bash
python -m pytest -q tests/test_multiscale_causal_incidents.py

python -u scripts/build_multiscale_causal_incidents.py \
  --scale-root scale1 /path/to/scale1/window/root \
  --scale-root scale4 /path/to/scale4/window/root \
  --expected-selected-per-scale 20 \
  --max-gap 2 \
  --top-score-count 3 \
  --output-dir /path/to/multiscale/incidents
```

The implementation verifies identical window identities and seed sets across
scales. `summary.csv` records candidate overlap, union size, incident count,
and reduction. `frozen_label_evaluation.csv`, when labels exist, records only
post-generation attack coverage and incident ranks.
