# TRACE: causal temporal multi-view selective provenance detection

## Motivation grounded in recent work

Recent top-tier systems leave adjacent gaps: TAPAS is efficient but cannot
identify fine-grained malicious edges; ProvFusion fuses entity and interaction
views but discards event order and timing; DUPIN needs previously observed
attack purposes; KnowHow cannot detect fully novel mechanisms; ProvX assumes a
useful alert subgraph and leaves robust feedback learning open.  Recent NIDS
work also shows that reconstruction plus ensemble uncertainty is no longer a
sufficient novelty claim by itself.

TRACE therefore treats uncertainty as a routing signal, not evidence of
malice.  It produces three causal, per-event views:

1. interaction semantic reconstruction;
2. cross-endpoint entity-state reconstruction; and
3. log inter-event-time prediction.

Each view is standardized using training-only running moments. The anomaly
score is the second-largest positive standardized residual, so one noisy view
cannot create an alert. Bootstrap semantic-head disagreement is exported
separately. A validation-only uncertainty quantile routes nodes into trusted,
confirmed-alert, and review queues without changing the primary binary metric.

## Safety and falsification gates

- TRACE, ECHO and DATR cannot be enabled together.
- No test labels select a checkpoint or uncertainty threshold.
- Run seed 0 only. Proceed only if MCC improves by at least 0.02 over the
  matched fixed-EMA baseline, AP/AUC do not materially collapse, and review
  volume remains operationally bounded.
- Before a five-seed experiment, inspect malicious, recurring-benign and
  cold-start-benign uncertainty distributions. If uncertainty merely detects
  cold start, reject the hypothesis. `scripts/analyze_trace_strata.py` creates
  this mandatory audit from validation CSVs and the saved node result.
- TrustedDualMemory does not update model parameters. Automatic adaptation is
  prohibited until contamination and catastrophic-forgetting tests pass.

## Evidence required for a CCF-A submission

At minimum: multiple datasets, paired five-seed intervals, leave-one-campaign-
out unknown attacks, temporal and cross-host drift, benign cold start,
training-data contamination, LoTL/mimicry, detection delay, alert-subgraph
purity, and runtime/memory comparisons. Code alone is not evidence of venue-
level novelty or effectiveness.
