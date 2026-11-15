# ORTHRUS Paper-1 TRACE-Open revision

This repository contains a research redesign based on the original ORTHRUS
codebase.

## Download

- [orthrus-paper1-DATR-redesign.zip](./orthrus-paper1-DATR-redesign.zip)
- SHA-256: `6f14045eded60bcc2a2b6b26d6d0963bdd8984732c08ac09d46e4628a35fb96f`

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

The TRACE and TRACE-Open screens were retained as explicit negative-result
paths. The current baseline protocol uses validation-only checkpoint selection:
all 20 lightweight parameter checkpoints share one neighbor-state snapshot,
the minimum validation mean event loss chooses the epoch, and the test split is
evaluated exactly once after replaying the causal validation history. Run
`scripts/run_validation_selected_baseline_screen.sh` for the seed-0 screen.
