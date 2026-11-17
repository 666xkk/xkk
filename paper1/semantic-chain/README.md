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

SHA-256: `66f2127eaca81ef2d4a38c95773c1a7c166072c33923fd3822ba5909c4aa3517`
