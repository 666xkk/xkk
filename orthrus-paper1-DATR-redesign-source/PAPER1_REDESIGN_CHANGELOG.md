# Redesign changelog

- Added `DetectionAlignedObjective`: class-balanced mean/CVaR training risk.
- Added label-free EMA checkpointing with initialization-bias correction.
- Added checkpoint save/test policies to avoid 20 redundant large snapshots.
- Added robust edge-type conditional score normalization fitted on validation
  telemetry only.
- Added fixed-checkpoint evaluation and a scalar protocol artifact; test-MCC
  oracle is no longer the default.
- Fixed `--run_from_training` timing initialization and CPU memory logging.
- Added primary five-seed and three-seed ablation launchers.
- Added environment-variable database credentials to avoid hard-coded secrets.
- Added detector-seed isolation to task hashes while deliberately sharing the
  same preprocessing artifacts across seeds.
- Added focused unit tests and a protocol log summarizer.

No empirical improvement is claimed by the code change alone. The included
protocol is designed to make a positive or negative result scientifically
interpretable without test-label checkpoint selection.
