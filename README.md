# Facial-Research (Autoencoder + CCA on Simulated Faces)

This branch showcases a compact research pipeline for learning low‑dimensional structure from simulated 3D face geometry using interior Gaussian curvature as the signal and validating with PCA/CCA analyses.

What you’ll find here:
- A PyTorch autoencoder (CurvatureAutoencoder) with a small grid search runner and plotting utilities.
- Seed‑wise PCA/CCA scripts to quantify relationships between learned latents/PCs and "ground‑truth" simulated factors.
- Ready‑to‑use processed arrays under `processed/` so you can run everything without regenerating meshes.

No real‑data code is documented in this branch; this README focuses only on the simulation pipeline available here.

## Repo map (this branch)

```
facial_research/
├─ encoder_files/
│  ├─ curvature_autoencoder.py       # CurvatureAutoencoder (PyTorch) + loader helpers
│  ├─ encoder_experiments.py         # Grid search over latent dims, activations, etc. (curvature)
│  ├─ encoder_experiments_xyz.py     # XYZ baseline AE experiments (optional)
│  ├─ encoder_fitting.py             # Fit a single model configuration
│  ├─ plot_*.py                      # Quick plots for validation losses/latents
│  └─ encoder_data/                  # Results written here (by seed/experiment)
├─ CCA/
│  ├─ run_seedwise_cca_curvAE.py     # CCA: CurvAE latents vs. true factors
│  ├─ run_seedwise_cca_curvpca_*.py  # CCA: Curvature PCA projections
│  ├─ run_seedwise_cca_xyzAE.py      # CCA: XYZ AE latents (optional baseline)
│  ├─ run_seedwise_cca_xyzpca_*.py   # CCA: XYZ PCA projections (baseline)
│  ├─ aggregate_plot_cca_results.py  # Aggregate/plot per‑seed results
│  └─ cca_results/                   # CCA outputs land here
├─ processed/
│  ├─ 2000_interior_curvatures.npy   # (N×D_interior) curvature matrix for N=2000 faces
│  ├─ interior_mask.npy              # Boolean mask for interior vertices
│  └─ sim_2000_d4_*                  # Seed‑specific raw sim folders (reference)
├─ given_data/                       # Input artifacts used by simulation scripts
├─ Interior_curvature_calculation.py # Curvature helpers (used in earlier experiments)
├─ *.R                               # R scripts from initial exploration
└─ README.md                         # You are here
```

## Data used in this branch
- Curvatures: `processed/2000_interior_curvatures.npy` (float32, shape `(N, D_interior)`)
- Interior mask: `processed/interior_mask.npy` (bool, shape `(V,)`) used for reference/visualization
These are sufficient to train the autoencoder and run PCA/CCA without regenerating meshes.

## How it works (two phases)

1) Learn latents with a curvature autoencoder
- The model is an MLP encoder/decoder with a linear bottleneck. Architectures are chosen based on latent size.
- The grid search script uses deterministic seeds, a fixed train/val/test split, and early stopping on val loss.
- Results per experiment (val losses per fold, best epoch, config) are saved under `encoder_files/encoder_data/Curvature/SEED*/experiments/` and summarized to a CSV for ranking.

2) Validate with PCA/CCA
- PCA baselines on curvature (and optionally XYZ) offer a simple linear comparator.
- CCA is run seed‑wise to measure alignment between latent variables (or PCs) and the known simulated factors.
- The aggregator script builds ECDFs and summary plots to compare methods across seeds and settings.

## Quick start

Create a clean Python environment (3.10–3.11 recommended) and install minimal deps:
- numpy, pandas, scipy, matplotlib
- scikit‑learn
- torch (CPU or CUDA per your machine)

Run the curvature AE grid search:
```powershell
# From repo root
python .\encoder_files\encoder_experiments.py
```
This will write results to `encoder_files/encoder_data/Curvature/SEED{seed}/experiments/` and a combined CSV at `encoder_files/encoder_data/Curvature/experiment_metadata.csv`.

Run CCA on CurvAE latents (after models exist):
```powershell
python .\CCA\run_seedwise_cca_curvAE.py
```
Run PCA baselines and CCA:
```powershell
python .\CCA\run_seedwise_cca_curvpca_randstate=42_ncomp=20.py
python .\CCA\run_seedwise_cca_xyzpca_randstate=42_ncomp=20.py   # optional baseline
```
Aggregate/plot CCA results:
```powershell
python .\CCA\aggregate_plot_cca_results.py
```

## Results layout

- `encoder_files/encoder_data/Curvature/SEED*/experiments/*.pkl` — per‑config CV summaries
- `encoder_files/encoder_data/Curvature/experiment_metadata.csv` — sortable table of configs vs. metrics
- `CCA/cca_results/...` — per‑seed result folders by method; aggregator scans these to build plots

## Design choices (concise)
- Use interior curvatures to emphasize intrinsic shape over pose.
- Deterministic seeds and fixed test split for fair comparisons.
- Right‑skewed latent grid to focus resolution where most gains were observed.
- Symmetric decoders; ELU/LeakyReLU activations tested; Adam optimizer; ReduceLROnPlateau scheduler.

## Notes for reviewers
- This branch is self‑contained for simulation‑based experiments.
- If you’re evaluating research style: see the CCA aggregator outputs and the metadata CSV which make model selection and reporting straightforward.

---
If you have questions or want a guided walk‑through of the code paths (model build → training loop → CCA), open an issue on GitHub.
