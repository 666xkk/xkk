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

## TRACE-Open extension after the DATR falsification

The fixed-EMA DATR experiment underperformed its matched baseline and the
tail-only diagnostic reproduced the baseline (MCC 0.2100 on seed 0). We
therefore reject unconditional high-loss tail optimization rather than tuning
it further. TRACE-Open addresses a different hypothesis informed by recent
TIFS work on open-world inter-class relationships and provenance-level APT
generation:

1. a dedicated classifier reserves one virtual-unknown class;
2. normal events update full prediction-distribution prototypes for each
   relation, instead of reducing novelty to entropy or disagreement;
3. virtual unknowns are type-preserving endpoint/context interventions inside
   a causal batch, requiring no attack labels, CTI, ATT&CK policy, or test data;
4. the primary score requires agreement between two of three independent
   mechanisms: TRACE consensus, prototype divergence, and virtual-unknown
   probability. Ensemble disagreement remains review routing only.

After the final EMA weights are installed, prototypes and running moments are
rebuilt from the training split only (prototype pass, then moment pass). This
prevents a hidden mismatch between non-EMA training statistics and the fixed
EMA model used at test time; validation and test labels are never consulted.

This implementation deliberately does **not** claim to synthesize complete APT
chains. TAGAPT-style stage generation and CrossHound-style host/network fusion
are outside the present code and must not appear as implemented contributions.

Run `scripts/run_trace_open_screen.sh` for seed 0. Continue only if the matched
fixed-EMA baseline is improved by at least 0.02 MCC, AP does not fall by more
than 0.02 absolute, prototype and virtual-unknown evidence are non-degenerate,
and benign cold-start review volume remains bounded. Otherwise reject the
TRACE-Open hypothesis before any multi-seed run.

## Evidence required for a CCF-A submission

At minimum: multiple datasets, paired five-seed intervals, leave-one-campaign-
out unknown attacks, temporal and cross-host drift, benign cold start,
training-data contamination, LoTL/mimicry, detection delay, alert-subgraph
purity, and runtime/memory comparisons. Code alone is not evidence of venue-
level novelty or effectiveness.

## Leak-free baseline checkpoint protocol

The fixed final EMA protocol is label-free but empirically weak on CADETS_E3.
Before testing another detector hypothesis, the baseline must be established by
standard validation-only model selection. `checkpoint_save_policy=validation_all`
stores every epoch's parameter state while sharing one large, weight-independent
neighbor-loader snapshot. `checkpoint_policy=validation_loss` then:

1. computes mean per-event loss for every epoch on validation only;
2. selects the minimum (earlier epoch breaks an exact tie);
3. records every candidate in `validation_selection.csv`;
4. reloads the selected checkpoint and replays validation to reconstruct the
   causal state; and
5. evaluates the test split exactly once.

No test label, test MCC, or attack reconstruction output participates in model
selection. Run `scripts/run_validation_selected_baseline_screen.sh` on seed 0
before any new detector objective.
