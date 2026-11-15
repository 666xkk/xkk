# ECHO research plan (post-DATR negative result)

## What the negative result established

The five-seed CADETS_E3 experiment showed that unconditional upper-tail risk
is not a reliable proxy for unknown attacks.  The full DATR combination reduced
mean MCC from 0.1623 to 0.0795 (one win in five seeds).  Unique checkpoint and
result hashes plus per-epoch objective logs rule out cache reuse.  This is a
valid negative result, not an implementation failure.

## Revised hypothesis

Deterministic reconstruction error conflates two sources of difficulty:

1. aleatoric difficulty from rare but benign event semantics; and
2. epistemic uncertainty from contexts unsupported by benign training data.

ECHO (Epistemic Contextual Head ORTHRUS) keeps one temporal encoder but trains
independent edge-semantic heads on bootstrap sub-samples.  Its event score is

    mean reconstruction loss + lambda * head disagreement.

The first term preserves the ORTHRUS signal.  The second is label-free and
targets model uncertainty rather than the unconditional high-loss tail.  The
shared encoder makes three heads much cheaper than three complete GNNs.

## Relation to top-tier PIDS lines and their open gap

Representative high-level systems such as THREATRACE, KAIROS, MAGIC and
ORTHRUS advance temporal/graph representation, reconstruction and attack-graph
recovery.  Their deployed anomaly scores remain predominantly point estimates.
That leaves rare-benign difficulty and unknown-context uncertainty entangled.
ECHO tests whether scalable epistemic decomposition closes this gap.

This document deliberately does not claim novelty merely from using an
ensemble.  A publishable claim requires evidence that the provenance-specific
shared/bootstrap construction improves unknown-context strata, remains robust
under drift, and offers a better accuracy-cost frontier than full ensembles.

## Falsifiable stage gates

1. Screen only seed 0 with disagreement weight 0.5. Do not launch a grid.
2. A candidate proceeds only if MCC improves by at least 0.02 over the matched
   fixed-EMA baseline without a material AP/AUC collapse.
3. Only after that gate, isolate weights 0 and 0.25, then confirm the selected
   candidate on seeds 1 and 2 before running five seeds.
4. Final evidence: at least three datasets, five seeds, paired confidence
   intervals, unseen-node/seen-node strata, temporal drift, contamination and
   runtime/memory analysis.
5. Compare against a three-complete-model ensemble to establish that the
   shared encoder is not merely a weaker reimplementation of a standard deep
   ensemble.

If the seed-0 and three-seed gates fail, ECHO is rejected; no headline claim is
made and no further full experiment is authorized.

## Literature verification note

Before submission, update the related-work table from the official proceedings
for the target venue and verify publication metadata.  The implementation was
designed from the documented methodological gap; it is not evidence by itself
of CCF-A-level novelty or acceptance.
