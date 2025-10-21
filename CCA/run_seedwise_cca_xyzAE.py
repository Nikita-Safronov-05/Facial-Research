from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch

# Ensure repository root is on sys.path so we can import encoder_files when running as a script
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from encoder_files.curvature_autoencoder import build_model, encode_latent  # type: ignore
from cca_utils import set_seeds, per_feature_pvalues_via_cca

"""Run seedwise CCA: X (XYZ feature CSV) vs latent space of retrained XYZ autoencoder.

Inputs per seed (1..100 expected):
  processed/sim_2000_d4_<seed>/X.csv                (predictor feature matrix)
  processed/sim_2000_d4_<seed>/XYZ.csv (OR raw coordinate features if naming differs)  [Not needed directly; model encodes curvature-like input]
  processed/sim_2000_d4_<seed>/interior_curvatures.npy (Only if needed for compatibility; here we instead encode XYZ data itself)

However, for the XYZ autoencoder we must feed the same input type it was trained on. The retrained XYZ model checkpoint records its source_file; we will load X.csv (predictors) and the model input CSV indicated by the checkpoint directory (XYZ.csv) for latent encoding.

Outputs per seed:
  CCA/cca_results/latent_XYZ/seed_<seed>/X_vs_latent_pvalues_per_component.csv
  CCA/cca_results/latent_XYZ/seed_<seed>/X_vs_latent_summary.csv

Usage:
  python CCA/run_seedwise_cca_xyzAE.py

Determinism: seeds fixed via set_seeds(42).
"""

def main():
    set_seeds(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    ckpt_path = repo_root / "encoder_files" / "encoder_data" / "XYZ" / "retrained_best_autoencoder_full.pth"
    processed_root = repo_root / "processed"
    out_root = script_dir / "cca_results" / "latent_XYZ"
    out_root.mkdir(parents=True, exist_ok=True)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Build model (shared utility) and move to device
    model = build_model(ckpt_path, device)

    # Iterate over seeds
    for seed in range(1, 101):
        seed_dir = processed_root / f"sim_2000_d4_{seed}"
        X_path = seed_dir / "X.csv"
        xyz_path = seed_dir / "XYZ.csv"
        if not X_path.exists():
            print(f"[skip] Missing X.csv for seed {seed}")
            continue
        if not xyz_path.exists():
            print(f"[skip] Missing XYZ.csv for seed {seed}")
            continue

        print(f"[seed {seed}] Loading X and XYZ data...")
        X = pd.read_csv(X_path).values  # (N, p)
        XYZ = pd.read_csv(xyz_path).values  # (N, d_in) expected input dimension

        n = min(len(X), len(XYZ))
        X = X[:n]
        XYZ = XYZ[:n]

        # Encode XYZ to latent space
        Z = encode_latent(model, XYZ, device=device)

        # Run CCA: predictors X vs latent Z
        res = per_feature_pvalues_via_cca(X, Z, n_components=None, fdr_alpha=1e-5)

        feat_names = [f"x{j+1}" for j in range(X.shape[1])]
        comp_names = [f"comp_{k+1}" for k in range(res["pvals_per_component"].shape[1])]

        seed_out = out_root / f"seed_{seed:03d}"
        seed_out.mkdir(parents=True, exist_ok=True)

        # Save outputs
        pd.DataFrame(res["pvals_per_component"], index=feat_names, columns=comp_names).to_csv(
            seed_out / "X_vs_latent_pvalues_per_component.csv"
        )
        pd.DataFrame({
            "feature": feat_names,
            "min_pval": res["min_pval"],
            "fdr_q": res["fdr_q"],
            "reject": res["reject"],
        }).sort_values("min_pval").to_csv(seed_out / "X_vs_latent_summary.csv", index=False)

        print(f"[seed {seed}] Saved CCA results to {seed_out}")

    print("Done. All available seeds processed for XYZ latent CCA.")


if __name__ == "__main__":
    main()
