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
    n_list = [5, 10, 15, 20, 30]

    for n_pca in n_list:
        out_root = script_dir / "cca_results" / f"XYZPCA_n{n_pca}"
        out_root.mkdir(parents=True, exist_ok=True)
        print(f"\n=== XYZ PCA: n_components={n_pca} ===")
        for seed in range(1, 101):
            seed_dir = processed_root / f"sim_2000_d4_{seed}"
            X_path = seed_dir / "X.csv"
            xyz_path = seed_dir / "XYZ.csv"
            if not (seed_dir.is_dir() and X_path.exists() and xyz_path.exists()):
                print(f"[skip] seed {seed:03d}: missing inputs")
                continue

            print(f"[n={n_pca}] [seed {seed:03d}] Loading X and XYZ...")
            X = pd.read_csv(X_path, dtype=np.float32).values
            XYZ = pd.read_csv(xyz_path, header=None, dtype=np.float32).values

            n = min(len(X), len(XYZ))
            X = X[:n]
            XYZ = XYZ[:n]

            # PCA on XYZ to top n_pca components
            print(f"[n={n_pca}] [seed {seed:03d}] PCA on XYZ (N={n}, D={XYZ.shape[1]}) -> {n_pca} comps")
            pca = PCA(n_components=n_pca, svd_solver="randomized", random_state=42)
            PCs = pca.fit_transform(XYZ)

            # CCA: X vs top PCs
            ncomp_cca = int(min(n_pca, X.shape[1], PCs.shape[1]))
            res = per_feature_pvalues_via_cca(X, PCs, n_components=ncomp_cca, fdr_alpha=1e-5)

            feat_names = [f"x{j+1}" for j in range(X.shape[1])]
            comp_names = [f"comp_{k+1}" for k in range(res["pvals_per_component"].shape[1])]

            seed_out = out_root / f"seed_{seed:03d}"
            seed_out.mkdir(parents=True, exist_ok=True)

            pd.DataFrame(res["pvals_per_component"], index=feat_names, columns=comp_names).to_csv(
                seed_out / "X_vs_XYZPCA_pvalues_per_component.csv"
            )
            pd.DataFrame(
                {
                    "feature": feat_names,
                    "min_pval": res["min_pval"],
                    "fdr_q": res["fdr_q"],
                    "reject": res["reject"],
                }
            ).sort_values("min_pval").to_csv(seed_out / "X_vs_XYZPCA_summary.csv", index=False)

            pd.DataFrame({
                "component": np.arange(1, len(pca.explained_variance_ratio_) + 1),
                "explained_variance_ratio": pca.explained_variance_ratio_,
            }).to_csv(seed_out / "XYZPCA_variance_explained.csv", index=False)

            print(f"[n={n_pca}] [seed {seed:03d}] Saved CCA results to {seed_out}")

    print("Done. All available seeds processed.")


if __name__ == "__main__":
    main()
