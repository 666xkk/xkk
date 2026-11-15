# ORTHRUS Paper-1 ECHO uncertainty revision

This repository contains a research redesign based on the original ORTHRUS
codebase.

## Download

- [orthrus-paper1-DATR-redesign.zip](./orthrus-paper1-DATR-redesign.zip)
- SHA-256: `6bec5ad0bd17eb7511ef896fa6d97ce14dcd3d25fc202d1ca7e8ded4f3484fc5`

The unpacked source is available under
[`orthrus-paper1-DATR-redesign-source/`](./orthrus-paper1-DATR-redesign-source/).
Read `PAPER1_ECHO_RESEARCH_PLAN.md` before running experiments.

## Scope

The original DATR combination produced a valid negative result. The revised
source therefore keeps DATR only for reproduction and adds ECHO: a shared
temporal encoder with bootstrapped semantic heads and uncertainty-aware
scoring. Mean CE and raw scoring remain the safe defaults. No empirical gain or
venue acceptance is claimed until the documented stage gates are passed.
