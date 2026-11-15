# ORTHRUS Paper-1 TRACE-Open revision

This repository contains a research redesign based on the original ORTHRUS
codebase.

## Download

- [orthrus-paper1-DATR-redesign.zip](./orthrus-paper1-DATR-redesign.zip)
- SHA-256: `6c0ec1aed984893ffcba04833a6dfd7f27ff0b33e437c38b3bde0dd76dd4abed`

The unpacked source is available under
[`orthrus-paper1-DATR-redesign-source/`](./orthrus-paper1-DATR-redesign-source/).
Read `PAPER1_TRACE_RESEARCH_PLAN.md` before running experiments.

## Scope

The original DATR combination produced a valid negative result. ECHO remains an
uncertainty component, not a headline novelty. TRACE adds temporally ordered
interaction, entity-state, and inter-event-time views; requires two-view
evidence consensus; and exports uncertainty separately for trusted/review/
confirmed-alert routing. TRACE-Open additionally learns a virtual-unknown class
from type-preserving endpoint/context interventions and compares full relation
prediction distributions with training-only prototypes. Final EMA statistics
are recalibrated on the training split only. Mean CE and raw scoring remain the
safe defaults. No empirical gain or venue acceptance is claimed until the
documented single-seed falsification gate passes.
