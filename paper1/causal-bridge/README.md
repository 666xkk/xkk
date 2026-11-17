# ORTHRUS Paper 1: causal-bridge module

This directory contains the frozen asymmetric role-constrained causal-bridge
implementation produced from the 2026-11-15 source snapshot.

The 2026-11-17 revision adds a label-free, per-relation density gate. It keeps
relation families whose unique-candidate expansion fits the frozen review
budget and rejects dataset-specific high-fanout expansions. The ungated path
remains available for controlled ablations.

## Download

- `orthrus-causal-bridge-20261115.tar.gz`
- SHA256: `7cf66f7bb449cdf2795cab844019a400d6597dec9e9b60a61c35740959ddbabc`
- `orthrus-density-gated-causal-bridge-20261117.tar.gz`
- SHA256: `9831f6f6dcdbf1e32796cb5031d6421ccd9cb35baea361dbfe718dc37f8c91e4`
- `orthrus-causal-collective-windows-20261117.tar.gz`
- SHA256: `d77db183f347fcd8cfc7bca8464c24d4cec73f98c5f560b24a466146bfffbdd1`

Extract the archive from the ORTHRUS repository root:

```bash
tar -xzf /path/to/orthrus-causal-bridge-20261115.tar.gz \
  -C /hy-tmp/orthrus-main
```

Then follow `CAUSAL_BRIDGE_DEPLOY.md` from the extracted archive. The module is
offline and does not retrain ORTHRUS. Candidate generation uses only frozen
scores, time-window provenance edges, and node roles; `y_true` is read only by
the optional post-generation evaluation summary.

For the revised method, add:

```bash
--density-gate \
--max-relation-expansion-ratio 1.0
```

The collective-window package additionally provides strictly causal
four-window weak-evidence accumulation over frozen edge-loss CSVs. Historical
scores are invariant to future-window appends; optional attack intervals are
used only for post-generation evaluation. See `COLLECTIVE_WINDOW_DEPLOY.md` in
the archive.
