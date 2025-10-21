from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from cca_utils import set_seeds, per_feature_pvalues_via_cca


def main():
    set_seeds(42)
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    processed_root = repo_root / "processed"
    # We'll run PCA with multiple component counts and save each as its own model directory
    n_list = [5, 10, 15, 20, 30]

    for n_pca in n_list:
        out_root = script_dir / "cca_results" / "CurvPCA" / f"CurvPCA_n{n_pca}"
        out_root.mkdir(parents=True, exist_ok=True)
        print(f"\n=== Curvature PCA: n_components={n_pca} ===")
        for seed in range(1, 101):
            seed_dir = processed_root / f"sim_2000_d4_{seed}"
            X_path = seed_dir / "X.csv"
            curv_path = seed_dir / "interior_curvatures.npy"
            if not (seed_dir.is_dir() and X_path.exists() and curv_path.exists()):
                print(f"[skip] seed {seed:03d}: missing inputs")
                continue

            print(f"[n={n_pca}] [seed {seed:03d}] Loading X and interior_curvatures...")
            X = pd.read_csv(X_path, dtype=np.float32).values
            Y = np.load(curv_path).astype(np.float32, copy=False)

            n = min(len(X), len(Y))
            X = X[:n]
            Y = Y[:n]

            # PCA on curvature matrix to top n_pca components
            print(f"[n={n_pca}] [seed {seed:03d}] PCA on Curvatures (N={n}, D={Y.shape[1]}) -> {n_pca} comps")
            pca = PCA(n_components=n_pca, svd_solver="randomized", random_state=42)
            PCs = pca.fit_transform(Y)

            # CCA: X vs top PCs of curvature
            ncomp_cca = int(min(n_pca, X.shape[1], PCs.shape[1]))
            res = per_feature_pvalues_via_cca(X, PCs, n_components=ncomp_cca, fdr_alpha=1e-5)

            # Save outputs
            feat_names = [f"x{j+1}" for j in range(X.shape[1])]
            comp_names = [f"comp_{k+1}" for k in range(res["pvals_per_component"].shape[1])]

            seed_out = out_root / f"seed_{seed:03d}"
            seed_out.mkdir(parents=True, exist_ok=True)

            pd.DataFrame(res["pvals_per_component"], index=feat_names, columns=comp_names).to_csv(
                seed_out / "X_vs_CurvPCA_pvalues_per_component.csv"
            )
            pd.DataFrame(
                {
                    "feature": feat_names,
                    "min_pval": res["min_pval"],
                    "fdr_q": res["fdr_q"],
                    "reject": res["reject"],
                }
            ).sort_values("min_pval").to_csv(seed_out / "X_vs_CurvPCA_summary.csv", index=False)

            # Variance explained of PCA
            pd.DataFrame({
                "component": np.arange(1, len(pca.explained_variance_ratio_) + 1),
                "explained_variance_ratio": pca.explained_variance_ratio_,
            }).to_csv(seed_out / "CurvPCA_variance_explained.csv", index=False)

            print(f"[n={n_pca}] [seed {seed:03d}] Saved CCA results to {seed_out}")

    print("Done. All available seeds processed.")


if __name__ == "__main__":
    main()
