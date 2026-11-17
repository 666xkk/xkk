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
