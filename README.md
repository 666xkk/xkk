# ORTHRUS Paper-1 DATR redesign

This repository contains a research redesign based on the original ORTHRUS
codebase.

## Download

- [orthrus-paper1-DATR-redesign.zip](./orthrus-paper1-DATR-redesign.zip)
- SHA-256: `3542c61d7c3c1e580f1fa1694aaec5cf03d3a55d125e66beb6ad4ab4fa2930af`

The unpacked source is available under
[`orthrus-paper1-DATR-redesign-source/`](./orthrus-paper1-DATR-redesign-source/).
Read `PAPER1_DATR_README.md` before running experiments.

## Scope

The redesign adds detection-aligned tail-risk training, rare-event balancing,
validation-only robust conditional scoring, and a fixed label-free EMA
checkpoint protocol. It does not claim an empirical improvement before the
included fair baseline and ablation experiments are run.
