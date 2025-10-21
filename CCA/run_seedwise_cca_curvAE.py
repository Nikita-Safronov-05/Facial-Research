from pathlib import Path
import sys
import re
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


def _discover_checkpoints(repo_root: Path) -> list[Path]:
    """Return all retrained AE checkpoints under Curvature/Retrained_AEs."""
    ckpt_dir = repo_root / 'encoder_files' / 'encoder_data' / 'Curvature' / 'Retrained_AEs'
    if not ckpt_dir.exists():
        print(f"[warn] Checkpoint directory not found: {ckpt_dir}")
        return []
    return sorted(ckpt_dir.glob('retrained_AE_*.pth'))


def _latent_label_from_path(p: Path) -> str:
    """Return a folder name for the latent dimension, e.g., 'latent_ld20'.
    If no latent dimension is encoded in the filename, fall back to 'latent_misc'.
    """
    m = re.search(r"ld(\d+)", p.stem)
    if m:
        return f"latent_ld{m.group(1)}"
    return "latent_misc"


def main():
    set_seeds(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    processed_root = repo_root / "processed"

    # Discover all retrained AE checkpoints automatically
    ckpts = _discover_checkpoints(repo_root)
    if not ckpts:
        raise SystemExit("No checkpoints found in encoder_files/encoder_data/Curvature/Retrained_AEs/.")

    seed_list = list(range(1, 101))  # fixed range as requested (no parser)
    fdr_alpha = 1e-5
    out_base = script_dir / "cca_results" / "CurvAE"

    for ckpt_path in ckpts:
        if not ckpt_path.exists():
            print(f"[skip model] ckpt not found: {ckpt_path}")
            continue
        latent_label = _latent_label_from_path(ckpt_path)
        model_label = ckpt_path.stem
        # Save as: cca_results/<latent_label>/<model_label>/seed_xxx
        out_root = out_base / latent_label / model_label
        out_root.mkdir(parents=True, exist_ok=True)
        print(
            f"\n=== Running CCA ===\n  latent: {latent_label}\n  model:  {model_label}\n  ckpt:   {ckpt_path}\n  out:    {out_root}\n"
        )

        try:
            model = build_model(ckpt_path, device)
        except Exception as e:
            print(f"[skip model] failed to load {ckpt_path}: {e}")
            continue

        # Iterate over seed folders
        for seed in seed_list:
            seed_dir = processed_root / f"sim_2000_d4_{seed}"
            X_path = seed_dir / "X.csv"
            curv_path = seed_dir / "interior_curvatures.npy"
            if not (X_path.exists() and curv_path.exists()):
                print(f"[skip seed {seed}] Missing inputs: X={X_path.exists()} curv={curv_path.exists()}")
                continue

            print(f"[latent {latent_label}] [model {model_label}] [seed {seed}] Loading features and curvatures...")
            X = pd.read_csv(X_path).values  # (N, p)
            Y = np.load(curv_path)          # (N, D)

            n = min(len(X), len(Y))
            X = X[:n]
            Y = Y[:n]

            # Encode Y to latent space (N, L)
            Z = encode_latent(model, Y, device=device)

            # Run CCA: X vs latent Z
            res = per_feature_pvalues_via_cca(X, Z, n_components=None, fdr_alpha=fdr_alpha)

            # Save per-seed outputs
            feat_names = [f"x{j+1}" for j in range(X.shape[1])]
            comp_names = [f"comp_{k+1}" for k in range(res["pvals_per_component"].shape[1])]

            seed_out = out_root / f"seed_{seed:03d}"
            seed_out.mkdir(parents=True, exist_ok=True)

            pd.DataFrame(res["pvals_per_component"], index=feat_names, columns=comp_names).to_csv(seed_out / "X_vs_latent_pvalues_per_component.csv")
            pd.DataFrame({
                "feature": feat_names,
                "min_pval": res["min_pval"],
                "fdr_q": res["fdr_q"],
                "reject": res["reject"],
            }).sort_values("min_pval").to_csv(seed_out / "X_vs_latent_summary.csv", index=False)

            print(f"[latent {latent_label}] [model {model_label}] [seed {seed}] Saved CCA results to {seed_out}")

    print("\nDone. All requested models/seeds processed.")


if __name__ == "__main__":
    main()
