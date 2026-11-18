# Exact fixed-budget incident benchmark

This evaluator consumes already frozen `window_scores_seedN.csv` files.  It
does not select windows, retrain a detector, or change incident parameters.
It reports window and attack-episode recall, incident reduction and ranks, and
an exact uniform-random fixed-budget baseline using the multivariate
hypergeometric distribution.

```bash
python -m pytest -q tests/test_incident_benchmarks.py

python -u scripts/evaluate_incident_benchmarks.py \
  --entry CLEARSCOPE_E3 /path/to/clearscope/window/root \
  --entry THEIA_E3 /path/to/theia/window/root \
  --entry CADETS_E3 /path/to/cadets/window/root \
  --max-gap 2 \
  --top-score-count 3 \
  --output-dir paper1/results/CAUSAL_INCIDENT_BENCHMARK
```

`random_p_episode_coverage_ge_observed` is the exact probability that a
uniform random selection of the same number of windows covers at least as many
attack episodes as the frozen method.  Labels are used only by this evaluator.
