# Semantic causal-chain v2

Cross-dataset, label-free provenance motif detector for the ORTHRUS paper-1
research line. The release archive is installed at the ORTHRUS repository root.

The implementation was created only after verifying that CADETS_E3, THEIA_E3
and CLEARSCOPE_E3 share the same event and node-table schemas. See
`source/SEMANTIC_CHAIN_DEPLOY.md` for the leak-free protocol and commands.

Current release: `orthrus-semantic-chain-v2-20261117.tar.gz`

Version 2 replaces the unbounded event-frequency accumulator with window-local,
distinct-motif evidence and adds equal-budget event-count and actor-count
baselines. The checksum is recorded beside the release archive.

SHA-256: `76f206bc8b3fd982e6c08149e0d70c4b39c1f4a1106fbbed248d8e574db60fc6`

The calibration-day extension adds isolated graph-6 embedding and
calibration-only inference tools. It is released separately so prior v1/v2
archives remain immutable.

Calibration tools release: `orthrus-day6-calibration-tools-20261117.tar.gz`

SHA-256: `aa02d800f5a61b57a5ed3633530915195df2b29fcc27bfc31bd2068230145d9e`

The conditional-tail audit converts frozen edge losses to finite-sample,
event-type-conditioned p-values and combines incident evidence with an
exposure-corrected Simes node score. Test labels are used only after scores are
frozen for ranking metrics.

Conditional-tail release: `orthrus-conditional-tail-audit-20261117.tar.gz`

SHA-256: `9e408d84f10dcec75870c6b369f748b57035e5dfe1640aa7ef167548c47cbfff`

The causal-incident extension merges stable collective weak-evidence windows
into ranked operational incidents. Its fixed aggregation protocol was frozen on
ClearScope and then reproduced on THEIA before implementation.

Causal-incident release: `orthrus-causal-incident-aggregation-20261117.tar.gz`

SHA-256: `5e86251ce6cc97dff0341930f12027f8e49b769bc85a2683520d19035eb57b61`

The frozen-selected replay extension reconstructs a deleted final neighbor
loader deterministically from the training stream, validates its event offset,
and replays validation/test for archived selected weights without retraining.

Frozen replay release: `orthrus-frozen-selected-replay-20261117.tar.gz`

SHA-256: `7734b436d5a03dfdc8a48b130ccb7d6541ce93eef8be747714e767cfc7b41577`

The incident-benchmark extension evaluates already frozen window selections
against an exact uniform-random fixed-budget baseline. It reports per-seed and
per-dataset attack-window recall, attack-episode recall, incident reduction,
incident ranks, and exact hypergeometric tail probabilities without changing
detector outputs or selection parameters.

Incident benchmark release: `orthrus-incident-benchmark-20261118.tar.gz`

SHA-256: `f9bace9de10d5ddcc5bd3b6bf376eac5429ab751d5a60f44022e0ec83e8e3267`

Version 2 adds an exact cross-dataset convolution of the random-budget null
distributions. Distinct datasets are treated as independent; model seeds are
reported separately and are not misrepresented as independent attack corpora.

Incident benchmark v2: `orthrus-incident-benchmark-v2-20261118.tar.gz`

SHA-256: `474d7b1cce4e2f8610cfa35ac56303e0b55133d7ef84d2c19e53c30808f1d945`

The frozen multi-scale causal-incident extension implements the coverage
operating mode. It unions preselected scale-1 and scale-4 windows, applies
label-free within-scale rank normalization, and merges adjacent candidates
into ranked incidents. Efficient scale-4 mode remains unchanged and is
reported separately because the coverage mode spends a larger window budget.

Multi-scale incident release:
`orthrus-multiscale-causal-incidents-20261118.tar.gz`

SHA-256: `b51cce5a8a44d43173f3870697d4fb427f645058601220b843680b5e29ec4803`

The IEEE TDSC figure release contains a single entry point that runs the
frozen module ablations and regenerates the complete ten-figure paper set in
PDF, SVG, and 600-dpi PNG formats. It includes the method overview, collective
weak-evidence ablation, multiscale-fusion ablation, and incident-reconstruction
sensitivity analysis. Attack labels enter only after scores and fixed-budget
selections are frozen.

TDSC figure release:
`orthrus-tdsc-complete-figures-20261119.tar.gz`

SHA-256: `f0a8ff69f7c736aecb3c511ef79fc85ae35701945c674b0265065431de3caaee`
