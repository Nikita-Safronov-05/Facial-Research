from __future__ import annotations
import os
import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

# FDR threshold for significance
FDR_ALPHA = float(os.getenv("FDR_ALPHA", "1e-5"))

# Expected roots for each model; only used if they exist
DEFAULT_ROOTS = {
    "XYZ+PCA": Path("CCA/cca_results/XYZPCA"),
    "Curvature+PCA": Path("CCA/cca_results/CurvPCA"),
    "Curvature+AE": Path("CCA/cca_results/latent"),
    "XYZ+AE": Path("CCA/cca_results/latent_XYZ"),
}

SEED_PATTERN = re.compile(r"seed[_-](\d+)$", re.IGNORECASE)

# True features to highlight in plots (override with env TRUE_FEATURES="x1,x101,x201,x301")
_TF = os.getenv("TRUE_FEATURES", "x1,x101,x201,x301").strip()
TRUE_FEATURES = tuple([t.strip() for t in _TF.split(",") if t.strip()])


def find_models() -> dict[str, Path]:
    here = Path(__file__).resolve().parent
    models: dict[str, Path] = {}
    for name, rel in DEFAULT_ROOTS.items():
        p = (here.parent / rel) if rel.is_absolute() is False and rel.parts[0] != here.name else rel
        # If rel is relative like "encoder_files/...", resolve from repo root
        repo_root = here.parent
        p = (repo_root / rel) if not rel.is_absolute() else rel
        if p.exists():
            models[name] = p

    # Additionally, auto-discover Curvature AE runs saved under
    #   CCA/cca_results/latent_ldXX/<model_name>/seed_xxx
    # and register each <model_name> as its own entry so aggregation treats them separately.
    repo_root = here.parent
    cca_root = repo_root / "CCA" / "cca_results"
    if cca_root.exists():
        # Back-compat: top-level latent_ld*
        for latent_dir in sorted(cca_root.glob("latent_ld*")):
            if not latent_dir.is_dir():
                continue
            # Derive a short latent tag like 'ld20' for labels
            m = re.search(r"ld(\d+)", latent_dir.name)
            latent_tag = f"ld{m.group(1)}" if m else latent_dir.name
            for model_dir in sorted(latent_dir.iterdir()):
                if not model_dir.is_dir():
                    continue
                # Only consider if it contains seed folders
                has_seed = any((d.is_dir() and SEED_PATTERN.search(d.name)) for d in model_dir.iterdir())
                if not has_seed:
                    continue
                label = f"Curvature+AE {latent_tag} / {model_dir.name}"
                models[label] = model_dir

        # New structure: CCA/cca_results/CurvAE/latent_ld*/<model>/seed_*
        curvAE_root = cca_root / "CurvAE"
        if curvAE_root.exists():
            for latent_dir in sorted(curvAE_root.glob("latent_ld*")):
                if not latent_dir.is_dir():
                    continue
                m = re.search(r"ld(\d+)", latent_dir.name)
                latent_tag = f"ld{m.group(1)}" if m else latent_dir.name
                for model_dir in sorted(latent_dir.iterdir()):
                    if not model_dir.is_dir():
                        continue
                    has_seed = any((d.is_dir() and SEED_PATTERN.search(d.name)) for d in model_dir.iterdir())
                    if not has_seed:
                        continue
                    label = f"Curvature+AE {latent_tag} / {model_dir.name}"
                    models[label] = model_dir

        # Also auto-discover PCA variants
        # Back-compat: top-level CurvPCA_n*/XYZPCA_n*
        for pca_dir in sorted(cca_root.glob("CurvPCA_n*")):
            if pca_dir.is_dir():
                models[f"Curvature+PCA {pca_dir.name.split('_n')[-1]}"] = pca_dir
        for pca_dir in sorted(cca_root.glob("XYZPCA_n*")):
            if pca_dir.is_dir():
                models[f"XYZ+PCA {pca_dir.name.split('_n')[-1]}"] = pca_dir
        # New structure: under CurvPCA/ and XYZPCA/
        curvPCA_root = cca_root / "CurvPCA"
        if curvPCA_root.exists():
            for pca_dir in sorted(curvPCA_root.glob("CurvPCA_n*")):
                if pca_dir.is_dir():
                    models[f"Curvature+PCA {pca_dir.name.split('_n')[-1]}"] = pca_dir
        xyzPCA_root = cca_root / "XYZPCA"
        if xyzPCA_root.exists():
            for pca_dir in sorted(xyzPCA_root.glob("XYZPCA_n*")):
                if pca_dir.is_dir():
                    models[f"XYZ+PCA {pca_dir.name.split('_n')[-1]}"] = pca_dir

    if not models:
        raise FileNotFoundError("No CCA result roots found. Update DEFAULT_ROOTS paths or ensure outputs exist.")
    return models


def list_seed_dirs(root: Path) -> list[Path]:
    # Accept either seed_XXX subfolders or a flat root already at seed_XXX
    seeds = [d for d in root.iterdir() if d.is_dir() and SEED_PATTERN.search(d.name)]
    if not seeds and SEED_PATTERN.search(root.name):
        seeds = [root]
    return sorted(seeds, key=lambda p: int(SEED_PATTERN.search(p.name).group(1)))


def find_summary_csv(seed_dir: Path) -> Path | None:
    # Try common summary filename patterns
    patterns = ("*summary*.csv", "*_summary.csv", "*vs*summary*.csv")
    for pat in patterns:
        for csv in seed_dir.glob(pat):
            try:
                df = pd.read_csv(csv, nrows=3)
                cols = {c.lower() for c in df.columns}
                if "feature" in cols and ("fdr_q" in cols or "q" in cols or "q_value" in cols):
                    return csv
            except Exception:
                continue
    # One level deeper just in case
    nested = list(seed_dir.glob("**/*summary*.csv"))
    if nested:
        return nested[0]
    return None


def load_model_results(model_name: str, model_root: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    seeds = list_seed_dirs(model_root) or [model_root]
    for seed_dir in seeds:
        m = SEED_PATTERN.search(seed_dir.name)
        seed = int(m.group(1)) if m else None
        csv_path = find_summary_csv(seed_dir)
        if not csv_path:
            continue
        df = pd.read_csv(csv_path)
        # normalize
        if "fdr_q" not in df.columns:
            for alt in ("q", "q_value", "fdrq", "fdr"):
                if alt in df.columns:
                    df["fdr_q"] = df[alt]
                    break
        if "feature" not in df.columns or "fdr_q" not in df.columns:
            continue
        out = df[["feature", "fdr_q"]].copy()
        out["model"] = model_name
        out["seed"] = seed
        rows.append(out)
    if not rows:
        raise FileNotFoundError(f"No per-seed summary CSVs found under {model_root}")
    res = pd.concat(rows, ignore_index=True)
    # ranks and acceptance
    res["rank_in_seed"] = res.groupby(["model", "seed"])['fdr_q'].rank(method="average")
    res["accepted"] = res["fdr_q"] <= FDR_ALPHA
    return res


def aggregate(models: dict[str, Path]) -> pd.DataFrame:
    dfs = [load_model_results(name, root) for name, root in models.items()]
    return pd.concat(dfs, ignore_index=True)


def plot_counts_by_model(df: pd.DataFrame, outdir: Path) -> None:
    # Per-seed total significant features
    counts = (df.groupby(["model", "seed"])['accepted'].sum()
                .rename("n_significant").reset_index())
    # Per-seed number of true features among the significant
    if TRUE_FEATURES:
        true_mask = df["feature"].isin(TRUE_FEATURES)
        true_counts = (df[true_mask].groupby(["model", "seed"])['accepted'].sum()
                        .rename("n_true_significant").reset_index())
        counts = counts.merge(true_counts, on=["model", "seed"], how="left")
        counts["n_true_significant"] = counts["n_true_significant"].fillna(0).astype(int)
    else:
        counts["n_true_significant"] = 0

    counts.to_csv(outdir / "significant_counts_per_seed.csv", index=False)

    plt.figure(figsize=(10, 5))
    models = counts["model"].unique().tolist()
    data = [counts.loc[counts["model"] == m, "n_significant"].values for m in models]
    plt.boxplot(data, labels=models, showfliers=False)

    # Color jitter points by how many TRUE_FEATURES were significant
    max_true = max(int(counts["n_true_significant"].max()), 1)
    # Build a discrete colormap for 0..max_true
    base_cmap = plt.cm.get_cmap("viridis", max_true + 1)
    colors = base_cmap(np.arange(max_true + 1))
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, max_true + 1.5), cmap.N)

    handles = []
    labels = []
    for i, m in enumerate(models, start=1):
        sub = counts.loc[counts["model"] == m].sort_values("seed")
        y = sub["n_significant"].values
        c = sub["n_true_significant"].values
        x = np.random.normal(i, 0.05, size=len(y))
        sc = plt.scatter(x, y, c=c, cmap=cmap, norm=norm, s=18, alpha=0.8, edgecolor='k', linewidths=0.4)
        # Create legend entries only once for 0..max_true
        if i == 1:
            for val in range(0, max_true + 1):
                handles.append(plt.Line2D([0], [0], marker='o', color='w', label=str(val),
                                          markerfacecolor=cmap(val), markeredgecolor='k', markersize=6))
                labels.append(str(val))

    plt.ylabel(f"# features @ FDR <= {FDR_ALPHA:g}")
    plt.title("CCA: significant features per seed (point color = # true features significant)")
    # Legend for number of true features
    if max_true >= 0:
        plt.legend(handles, labels, title="# true features", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(outdir / "counts_boxplot_colored.png", dpi=200, bbox_inches='tight')
    plt.close()


def plot_ecdf(df: pd.DataFrame, outdir: Path) -> None:
    plt.figure(figsize=(9, 5))
    for model, sub in df.groupby("model"):
        q = np.clip(sub["fdr_q"].values, 1e-300, 1)
        q = np.sort(q)
        y = np.arange(1, len(q) + 1) / len(q)
        plt.plot(q, y, label=model)
    plt.xscale("log")
    plt.xlabel("FDR q-value (log scale)")
    plt.ylabel("ECDF")
    plt.title("CCA: aggregate FDR q-value distributions")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "fdr_ecdf.png", dpi=200)
    plt.close()


def plot_true_feature_ranks(df: pd.DataFrame, outdir: Path, true_feats=("x1", "x101", "x201", "x301")) -> None:
    sub = df[df["feature"].isin(true_feats)]
    if sub.empty:
        return
    med = (sub.groupby(["model", "feature"])['rank_in_seed']
             .median().rename("median_rank").reset_index())
    features = list(true_feats)
    models = med["model"].unique().tolist()
    width = 0.8 / max(1, len(models))
    x = np.arange(len(features))
    plt.figure(figsize=(9, 5))
    for i, m in enumerate(models):
        y = [med.loc[(med["model"] == m) & (med["feature"] == f), "median_rank"].min()
             if ((med["model"] == m) & (med["feature"] == f)).any() else np.nan
             for f in features]
        plt.bar(x + i * width, y, width=width, label=m)
    plt.xticks(x + (len(models) - 1) * width / 2, features)
    plt.ylabel("Median rank (lower is better)")
    plt.title("CCA: true feature rank across seeds")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "true_feature_median_ranks.png", dpi=200)
    plt.close()
    med.to_csv(outdir / "true_feature_median_ranks.csv", index=False)


def _produce_outputs(df: pd.DataFrame, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / "all_models_long.csv", index=False)
    plot_counts_by_model(df, outdir)
    plot_ecdf(df, outdir)
    plot_true_feature_ranks(df, outdir, true_feats=TRUE_FEATURES)


def main():
    parser = argparse.ArgumentParser(description="Aggregate and plot CCA results across models with optional XYZ+AE inclusion.")
    parser.add_argument(
        "--xyz-ae-mode",
        choices=["both", "with", "without"],
        default="both",
        help="Control inclusion of the 'XYZ+AE' model in plots: generate both variants, only with, or only without.",
    )
    args = parser.parse_args()

    models = find_models()
    df_all = aggregate(models)

    root_out = Path(__file__).resolve().parent / "cca_results"

    def has_rows(df: pd.DataFrame) -> bool:
        return df is not None and not df.empty and (df.shape[0] > 0)

    wrote_any = []

    if args.xyz_ae_mode in ("both", "with"):
        df_with = df_all.copy()
        outdir_with = root_out / "summary_with_XYZAE"
        if has_rows(df_with):
            _produce_outputs(df_with, outdir_with)
            wrote_any.append(outdir_with)
        else:
            print("[warn] No data rows found for 'with XYZ+AE' variant; skipping.")

    if args.xyz_ae_mode in ("both", "without"):
        df_without = df_all[df_all["model"] != "XYZ+AE"].copy()
        outdir_without = root_out / "summary_without_XYZAE"
        if has_rows(df_without):
            _produce_outputs(df_without, outdir_without)
            wrote_any.append(outdir_without)
        else:
            print("[warn] No data rows found for 'without XYZ+AE' variant; skipping.")

    if wrote_any:
        locs = ", ".join(str(p) for p in wrote_any)
        print(f"Done. Outputs written to: {locs}")
    else:
        raise SystemExit("No outputs generated; check input result folders and filters.")


if __name__ == "__main__":
    main()
