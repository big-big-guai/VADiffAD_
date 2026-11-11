# VADiffAD: Variable-Aware Diffusion-Path Anomaly Detection

Anonymous implementation of **When Anomalies Emerge along Diffusion Paths: Variable-Aware Perturbation for Detection**.

## Abstract

> Multivariate time-series anomalies can be difficult to identify from a final reconstruction discrepancy alone. VADiffAD examines how observations respond at intermediate stages of a diffusion process. A variable-aware noise schedule adjusts perturbation strength according to each channel's normal temporal variation. A shared temporal convolutional noise predictor then generates localized responses along deterministic diffusion paths. Response magnitude, response variation, and deviation from normal path signatures are calibrated against normal references and selectively combined with contextual and historical residuals. A validation-only proxy gate rejects path evidence when it does not provide sufficiently consistent benefit.

## Framework

The SMAP implementation contains three connected stages:

```text
  Already-prepared SMAP normal observations
                   |
         Channel-wise training statistics
                   |
        Variable-aware noise schedule
                   |
       Shared Temporal CNN noise field
                   |
       +-----------+---------------------------+
       |                                       |
  Normal Path Bank                       Test windows
  (normal training)                           |
       |                              Coarse retention probing
       |                                       |
       |                              Deterministic epsilon path
       |                                       |
       +-------------------------------+-------+
                                       |
                          Local magnitude / variation /
                          regularized Mahalanobis distance
                                       |
                         Empirical normal CDF calibration
                                       |
                    Context + history residual calibration
                                       |
                    Validation-only, rejectable path gate
                                       |
                            Local fused Top-q score
                                       |
                    First 5% unlabeled SMAP warm-up cutoff
                                       |
                     Strict point-wise metrics (no PA)
```

The normal path bank stores **one signature mean and regularized covariance per variable**, not a cluster ensemble. Intermediate path features are computed directly from the noise predictor's output, rather than from a reconstructed-clean residual.

## Repository Structure

```text
VADiffAD_SMAP/
├── README.md                 # The only README in the release
├── requirements.txt
├── main.py                   # Single SMAP train + inference + evaluation entry
├── base/
│   ├── __init__.py
│   ├── dataset.py            # Read prepared SMAP arrays; segmentation-safe windows
│   └── metrics.py            # Strict point-wise Precision / Recall / F1 / AP
├── config/
│   ├── __init__.py
│   ├── parser.py             # SMAP configuration schema and validation
│   └── smap.yaml             # Main-method settings (no ablation presets)
├── vadiffad/
│   ├── __init__.py
│   ├── diffusion.py          # Training-only variable MAD, c_d and alpha_{t,d}
│   ├── model.py              # Shared Temporal CNN and masked-context estimator
│   ├── path.py               # Adaptive range and deterministic epsilon responses
│   ├── reference.py          # Variable-specific Normal Path Bank and shrinkage
│   ├── scoring.py            # Normal ECDF, 5 evidence terms, fusion, Top-q
│   ├── trainer.py            # Normal-only optimization and early stopping
│   └── engine.py             # Full SMAP orchestration
├── utils/
│   ├── __init__.py
│   └── proxy.py              # Validation-only synthetic proxies and gate
├── scripts/
│   └── SMAP.sh               # One-command SMAP launcher
└── tests/
    └── test_core.py         # Formula-level tests; not additional experiments
```

## Get Started

### 1. Install the environment

Use Python 3.9 or newer with a PyTorch installation appropriate for your CPU/CUDA environment. The implementation uses PyTorch, NumPy, SciPy, scikit-learn and PyYAML.

```bash
python -m pip install -r requirements.txt
```

### 2. Train and evaluate the SMAP main method

From the repository root, run:

```bash
bash scripts/SMAP.sh cuda
```

Or explicitly:

```bash
python main.py --config config/smap.yaml --data data/SMAP.npz \
  --output runs/SMAP --device cuda --seeds 42,43,44
```

## Method and Evaluation Details

This implementation follows the manuscript's specified operations:

- **Variable-aware diffusion:** first-difference MAD from normal training data, a bounded per-variable coefficient with mean one, and the shared diffusion noise budget (Eqs. 7–12).
- **Intermediate paths:** coarse retention-based depth selection, deterministic Euler-style response updates, and **signed noise-prediction responses** `epsilon_theta(X_t,t,c)` (Eqs. 13–17).
- **Local statistics:** interval-weighted response energy, normalized response changes, a length `2K-1` signature, and a Mahalanobis distance using `Sigma_tilde = (1-eta) Sigma + eta * tr(Sigma)/m * I + epsilon I` (Eqs. 18–21).
- **Five calibrated components:** per-variable, per-statistic empirical upper-tail CDF (`V`, `C`, `D`, contextual residual `R`, historical residual `A`). The path and residual branches use the configured nonnegative, fixed weights (Eqs. 22–25).

The normal-only training objective includes the shared noise regression loss and an auxiliary masked-context residual estimator. Its optimization does not access anomaly annotations. A historical residual is computed causally from prior observations within each entity.
