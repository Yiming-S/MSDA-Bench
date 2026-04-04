# MSDA-Bench

**Multi-Source Domain Adaptation Benchmark for Cross-Session EEG Classification**

An interactive dashboard for comparing how different source session utilization strategies handle distribution shift in EEG-based brain-computer interfaces.

## Pipelines

| Pipeline | Strategy | Sessions Used |
|---|---|---|
| **MAP** | Merge All & Predict — uniform pooling of all source sessions | 100% |
| **DWP** | Distance-Weighted Pooling — soft inverse-distance weighting | 100% |
| **MMP_mta** | Minimum-distance Multi-source, Merge-Then-Adapt — CI-gated nearest selection + weighted merge | ~40% |
| **MMP_moe** | Minimum-distance Multi-source, Mixture-of-Experts — CI-gated selection + weighted voting | ~40% |
| **BDP_fb** | Bridge-Domain Pipeline (far-to-bridge) — hierarchical bridge/far partition with proxy tuning | ~57% |
| **BDP_bf** | Bridge-Domain Pipeline (bridge-to-far) — reverse proxy direction | ~57% |

## Quick Start

```bash
git clone https://github.com/YimingShen/MSDA-Bench.git
cd MSDA-Bench
pip install -r requirements.txt
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

## Dashboard Pages

1. **Overview** — Dataset summary, completion matrix, QC checks
2. **Pipeline Benchmark** — Head-to-head pipeline comparison with multiple metrics
3. **Stability & Sensitivity** — Config selection premium, ranking stability, variance analysis
4. **Config Explorer** — 24-config heatmap, feature contribution
5. **Subject Explorer** — Single-subject deep-dive
6. **DA Analysis** — Domain adaptation gain/harm analysis
7. **Mechanism Explorer** — Session role visualization (BDP bridge/far, MMP selection, DWP weights)
8. **Target Session** — Accuracy by target session, difficulty ranking
9. **Prediction Error** — Per-class accuracy, confusion matrix, hard sessions
10. **Efficiency & Progress** — Timing, accuracy-time tradeoff

## Datasets

The `data/` directory contains experiment results:

- **bnci004** — BNCI2014-004 (9 subjects, 5 sessions, 3 EEG channels, Motor Imagery left/right hand)

To add a new dataset, place its result files (`{dataset}_{subject}_{pipeline}.pkl`, `*_detail.pkl`, `*_roles.csv`) in a new subdirectory under `data/`. The dashboard will auto-detect it.

## Statistical Definitions

| Symbol | Definition |
|---|---|
| `M(s,p)` | Mean accuracy over all configs — primary metric, no config selection bias |
| `B(s,p)` | Best config accuracy — oracle upper bound |
| `G(s,p)` | Mean DA gain — average (acc_DA - baseline) |
| `H(s,p)` | DA helps rate — fraction of configs where DA improves accuracy |

All summary statistics (Mean, Median, SD, percentiles) are computed on **subject-level values**, not raw cells.

## Authors

**Yiming Shen** and **David Degras**

Department of Mathematics, University of Massachusetts Boston

## License

MIT License
