# Redesign changelog

## TRACE multi-view selective detection

- Repositioned ECHO as an uncertainty component after comparison with recent
  reconstruction-plus-uncertainty zero-day work.
- Added causal interaction, cross-endpoint entity, and inter-event-time views.
- Added training-only running moments and two-of-three evidence consensus.
- Exported per-event raw losses, standardized evidence, uncertainty and final
  consensus to testing CSVs.
- Added validation-only trusted/review/confirmed-alert routing diagnostics.
- Added a conservative dual memory that quarantines suspicious observations
  and never updates detector parameters automatically.
- Preserved state-dict compatibility when TRACE is disabled.

## ECHO revision after the DATR falsification experiment

- Recorded the five-seed DATR negative result instead of hiding it.
- Added an opt-in, shared-encoder bootstrap ensemble (`ECHO`) that separates
  mean reconstruction surprisal from semantic-head disagreement.
- Added a bootstrap training risk and an uncertainty-fused inference score.
- Added unit tests for fusion and exact baseline-recovery invariants.
- Added a seed-0 screening script with explicit stage gates; five-seed runs are
  not justified until a candidate clears the screen.
- Changed defaults back to mean CE and raw scores. Failed DATR components are
  no longer silently active.

## Initial DATR redesign

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
