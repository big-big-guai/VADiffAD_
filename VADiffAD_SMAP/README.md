# VADiffAD: Variable-Aware Diffusion-Path Anomaly Detection

Anonymous implementation of **When Anomalies Emerge along Diffusion Paths: Variable-Aware Perturbation for Detection**.

This repository releases the **SMAP main-method pipeline only**. It preserves the method's normal-only training, variable-aware diffusion, localized deterministic path statistics, normal-path reference construction, validation-guided selective path integration, and strict point-wise evaluation. It is a focused anonymous research artifact rather than a reproduction of every table and figure in the paper.

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

**Intentionally not included:** any dataset, download link handler, preprocessing/conversion program, alternative benchmark script, baseline reproduction, standalone ablation/sensitivity/efficiency experiment, result figures, pretrained model, or previously generated benchmark metrics.

## Get Started

### 1. Install the environment

Use Python 3.9 or newer with a PyTorch installation appropriate for your CPU/CUDA environment. The implementation uses PyTorch, NumPy, SciPy, scikit-learn and PyYAML.

```bash
python -m pip install -r requirements.txt
```

### 2. Supply already-prepared SMAP data

**This repository does not download, split, transform, or prepare SMAP.** Provide your existing, correctly separated `data/SMAP.npz`, or pass a different file location through `--data` (the filename must remain `SMAP.npz`).

| Array | Shape | Role |
| --- | --- | --- |
| `train` | `[N_train, 25]` | Normal training observations |
| `validation` | `[N_val, 25]` | Normal validation and proxy-selection reference |
| `calibration` | `[N_cal, 25]` | Independent normal score-calibration observations |
| `test` | `[N_test, 25]` | SMAP test observations |
| `labels` | `[N_test]` | Binary annotations used **only for final evaluation** |

The file can optionally contain `train_boundaries`, `validation_boundaries`, `calibration_boundaries`, and `test_boundaries`. Each is an integer array of `[start, end)` pairs with shape `[entities, 2]`, covering the corresponding stream without gaps. These prevent training windows, history residuals, or warm-up thresholds from crossing boundaries between independent SMAP entities. For concatenated SMAP channels, supplying correct boundaries is essential. The code **does not infer boundaries or construct splits**.

All four feature arrays must have the same 25-variable ordering. Split separation, missing-value handling, and any data preparation must have been completed externally. Scaling is fitted on `train` only inside the model runner.

### 3. Train and evaluate the SMAP main method

From the repository root, run:

```bash
bash scripts/SMAP.sh cuda
```

Or explicitly:

```bash
python main.py --config config/smap.yaml --data data/SMAP.npz \
  --output runs/SMAP --device cuda --seeds 42,43,44
```

Use `--device cpu` on systems without CUDA. The example above performs **three independent SMAP main-method runs**, following the paper's repeated-run reporting convention. For a single code-level check, you may use `--seeds 42`; this is not an ablation experiment.

Default experimental configuration: 100-point windows; training stride 25; evaluation stride 100; diffusion length 1,000; shared hidden dimension 64; 20 coarse probing nodes; 10 path intervals; at most 30,000 total time-variable normal signatures; covariance shrinkage 0.1; AdamW learning rate 1e-3; batch size 128; and gate admission margin 0.06. The exact settings are in `config/smap.yaml`.

### 4. Inspect model outputs

Each run creates its own folder, such as:

```text
runs/SMAP/
├── seed_42/
│   ├── checkpoint.pt
│   ├── configuration.json
│   ├── training_history.json
│   ├── normalization.npz
│   ├── normal_path_bank.npz
│   ├── validation_gate.json
│   ├── predictions.npz
│   └── metrics.json
├── seed_43/
├── seed_44/
└── summary.json
```

`summary.json` reports the mean and **sample** standard deviation of the completed SMAP runs. `predictions.npz` contains the test-point scores, predetermined thresholds and variable contributions. Labels are saved alongside predictions for final analysis only; the scoring and threshold code does not use them.

## Method and Evaluation Details

This implementation follows the manuscript's specified operations:

- **Variable-aware diffusion:** first-difference MAD from normal training data, a bounded per-variable coefficient with mean one, and the shared diffusion noise budget (Eqs. 7–12).
- **Intermediate paths:** coarse retention-based depth selection, deterministic Euler-style response updates, and **signed noise-prediction responses** `epsilon_theta(X_t,t,c)` (Eqs. 13–17).
- **Local statistics:** interval-weighted response energy, normalized response changes, a length `2K-1` signature, and a Mahalanobis distance using `Sigma_tilde = (1-eta) Sigma + eta * tr(Sigma)/m * I + epsilon I` (Eqs. 18–21).
- **Five calibrated components:** per-variable, per-statistic empirical upper-tail CDF (`V`, `C`, `D`, contextual residual `R`, historical residual `A`). The path and residual branches use the configured nonnegative, fixed weights (Eqs. 22–25).
- **Selective integration:** only synthetic proxy anomalies constructed from **normal validation** data participate in admitting a path weight. The manuscript denotes the selection function abstractly as `G_delta`. For reproducibility, this release explicitly tests `g` in `{0, 0.25, 0.5, 0.75, 1}`, admits path evidence only for a relative proxy-AP gain of at least `delta=0.06` and no negative proxy-family F1 change, and otherwise uses residual-only scoring. This is a **documented realization** of the paper's rejectable mechanism, not a claim that the manuscript uniquely defines that exact selection function.
- **Point-wise evaluation:** local Top-q across **all** 25 variables, median aggregation over overlapping windows, a threshold from the first 5% of each unlabeled test entity, and final Precision, Recall, F1, AUROC and AP without point adjustment. No test-label F1 maximization is used (Eqs. 27–29).

The normal-only training objective includes the shared noise regression loss and an auxiliary masked-context residual estimator. Its optimization does not access anomaly annotations. A historical residual is computed causally from prior observations within each entity.

## Reproducibility and Limitations

The supplied package contains **only SMAP code**, not the SMAP data or author-generated checkpoints. The exact result depends on the supplied external SMAP splits, preprocessing choices, hardware and random seeds. A successful unit test or an artificial-data smoke test establishes only software execution; it does **not** certify reproduction of the paper's SMAP score or full main table. This release does not include experimental result CSVs or fabricated metrics.

To check equations and array interfaces without processing the dataset:

```bash
python -m unittest discover -s tests -v
```

These are deterministic formula-level checks, not model comparisons or additional experiments.
