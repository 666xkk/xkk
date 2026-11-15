# ORTHRUS Paper-1 TRACE selective multi-view revision

This repository contains a research redesign based on the original ORTHRUS
codebase.

## Download

- [orthrus-paper1-DATR-redesign.zip](./orthrus-paper1-DATR-redesign.zip)
- SHA-256: `6ea1cb25a38a01259b16f67465404a2ed50e6be4111c5e806bc4dba29b8c8d74`

The unpacked source is available under
[`orthrus-paper1-DATR-redesign-source/`](./orthrus-paper1-DATR-redesign-source/).
Read `PAPER1_TRACE_RESEARCH_PLAN.md` before running experiments.

## Scope

The original DATR combination produced a valid negative result. ECHO remains an
uncertainty component, not a headline novelty. TRACE adds temporally ordered
interaction, entity-state, and inter-event-time views; requires two-view
evidence consensus; and exports uncertainty separately for trusted/review/
confirmed-alert routing. Mean CE and raw scoring remain the safe defaults. No
empirical gain or venue acceptance is claimed until the documented gates pass.
