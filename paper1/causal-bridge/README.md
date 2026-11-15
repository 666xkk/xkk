# ORTHRUS Paper 1: causal-bridge module

This directory contains the frozen asymmetric role-constrained causal-bridge
implementation produced from the 2026-11-15 source snapshot.

## Download

- `orthrus-causal-bridge-20261115.tar.gz`
- SHA256: `7cf66f7bb449cdf2795cab844019a400d6597dec9e9b60a61c35740959ddbabc`

Extract the archive from the ORTHRUS repository root:

```bash
tar -xzf /path/to/orthrus-causal-bridge-20261115.tar.gz \
  -C /hy-tmp/orthrus-main
```

Then follow `CAUSAL_BRIDGE_DEPLOY.md` from the extracted archive. The module is
offline and does not retrain ORTHRUS. Candidate generation uses only frozen
scores, time-window provenance edges, and node roles; `y_true` is read only by
the optional post-generation evaluation summary.
