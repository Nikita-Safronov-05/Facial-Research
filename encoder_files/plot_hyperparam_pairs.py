"""Generate pairwise hyperparameter performance heatmaps.

Scans experiment pickle results under:
  encoder_files/encoder_data/Experiments/SEED*/experiments/*.pkl

Extracts the following hyperparameters per experiment:
  - latent_dim (int)
  - batch_size (int)
  - learning_rate (float, displayed in scientific notation)
  - patience (int)
  - activation (categorical: ELU, LeakyReLU, ...)
  - encoder_depth (len of encoder_layers)
  - decoder_depth (len of decoder_layers)
  - encoder_width_signature (string join of first two layer sizes for coarse width grouping)

Aggregates validation loss (prefers 'avg_best_val_loss' else derives from folds) by taking the median
across all remaining dimensions (seeds, architectures) for each pair of hyperparameters.

Outputs:
    - PNG grid of log-scale heatmaps: encoder_files/encoder_data/Experiments/hyperparam_pair_heatmaps_log.png
    - CSV of aggregated pair results: encoder_files/encoder_data/Experiments/hyperparam_pair_summary.csv

Heatmap rules:
  - For each pair (A,B), create a matrix with sorted unique values of A (rows) and B (columns)
  - Cells show median val_loss; color mapped with shared global vmin/vmax across all heatmaps
  - Missing combinations left blank (white hatch)
  - Numeric axes formatted nicely; learning_rate uses scientific notation labels
  - Activation treated as categorical (string ordering alphabetically)

This visualization helps identify weak regions of the grid to prune in future searches.

Run from repository root (PowerShell):
  python .\encoder_files\plot_hyperparam_pairs.py
"""
from __future__ import annotations
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

EXPERIMENTS_DIR = "XYZ"
SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR / "encoder_data" / EXPERIMENTS_DIR
SEED_PREFIX = "SEED"
PKL_SUBDIR = "experiments"
OUT_PNG_LOG = BASE_DIR / "hyperparam_pair_heatmaps_log.png"       # single log-scale output
OUT_CSV = BASE_DIR / "hyperparam_pair_summary.csv"

# ---------------- Loss Extraction -----------------

def _extract_loss(rec: Dict[str, Any]) -> float | None:
    if "avg_best_val_loss" in rec:
        return float(rec["avg_best_val_loss"])
    if "best_val_losses_per_fold" in rec:
        try:
            arr = np.asarray(rec["best_val_losses_per_fold"], dtype=float)
            if arr.size:
                return float(arr.mean())
        except Exception:
            pass
    # fallback using val_losses_per_fold -> min per fold
    vpf = rec.get("val_losses_per_fold")
    if isinstance(vpf, (list, tuple)) and vpf:
        mins = []
        for fold in vpf:
            try:
                a = np.asarray(fold, dtype=float)
                if a.size:
                    mins.append(a.min())
            except Exception:
                continue
        if mins:
            return float(np.mean(mins))
    return None

# ---------------- Data Collection -----------------

def collect_df(base: Path) -> pd.DataFrame:
    if not base.exists():
        raise FileNotFoundError(f"Base experiments dir not found: {base}")
    seed_dirs = [d for d in base.iterdir() if d.is_dir() and d.name.startswith(SEED_PREFIX)]
    rows: List[Dict[str, Any]] = []
    for sdir in seed_dirs:
        exp_dir = sdir / PKL_SUBDIR
        search_dirs = [exp_dir] if exp_dir.exists() else [sdir]
        for cdir in search_dirs:
            for pkl in cdir.glob("*.pkl"):
                try:
                    with open(pkl, "rb") as f:
                        rec = pickle.load(f)
                except Exception as e:
                    print(f"[WARN] Failed to load {pkl.name}: {e}")
                    continue
                loss = _extract_loss(rec)
                if loss is None or not np.isfinite(loss) or loss <= 0:
                    continue
                arch = rec.get("architecture", {}) or {}
                enc_layers = arch.get("encoder_layers", []) or []
                dec_layers = arch.get("decoder_layers", []) or []
                # coarse width signature: take first two layers (if available) to reduce cardinality
                enc_width_sig = "-".join(str(x) for x in enc_layers[:2]) if enc_layers else "NA"
                rows.append({
                    "latent_dim": rec.get("latent_dim"),
                    "batch_size": rec.get("batch_size"),
                    "learning_rate": rec.get("learning_rate"),
                    "patience": rec.get("patience"),
                    "activation": arch.get("activation", "UNK"),
                    "encoder_depth": len(enc_layers),
                    # decoder_depth intentionally removed (redundant with encoder_depth in this dataset)
                    "encoder_width_sig": enc_width_sig,
                    "val_loss": loss,
                })
    if not rows:
        raise RuntimeError("No experiment records found with valid loss.")
    df = pd.DataFrame(rows)
    # Clean types
    int_cols = ["latent_dim", "batch_size", "patience", "encoder_depth"]
    for c in int_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype('Int64')
    df["learning_rate"] = pd.to_numeric(df["learning_rate"], errors="coerce")
    return df.dropna(subset=["val_loss"])  # ensure losses present

# ---------------- Aggregation -----------------

def aggregate_pairwise(df: pd.DataFrame, agg_fn="min") -> pd.DataFrame:
    hyper_cols = [
        "latent_dim","batch_size","learning_rate","patience","activation",
        "encoder_depth","encoder_width_sig"
    ]
    # Ensure categorical ordering stable
    cat_orders: Dict[str, List[Any]] = {}
    for c in hyper_cols:
        # collect unique values in sorted order (numeric or lexical)
        vals = df[c].dropna().unique().tolist()
        try:
            vals_sorted = sorted(vals)
        except Exception:
            vals_sorted = vals
        cat_orders[c] = vals_sorted
    records: List[Dict[str, Any]] = []
    from itertools import combinations
    for a, b in combinations(hyper_cols, 2):
        grp = df.groupby([a,b])['val_loss']
        if agg_fn == 'mean':
            agg_series = grp.mean()
        elif agg_fn == 'min':
            agg_series = grp.min()
        elif agg_fn == 'median':
            agg_series = grp.median()  # retained for compatibility if explicitly requested
        else:
            raise ValueError("Unsupported agg_fn")
        for (va, vb), metric in agg_series.items():
            records.append({
                'hyper_a': a, 'val_a': va,
                'hyper_b': b, 'val_b': vb,
                'agg_loss': metric,
                'agg_fn': agg_fn,
                'n': int((df[(df[a]==va) & (df[b]==vb)]).shape[0])
            })
    return pd.DataFrame(records), cat_orders

# ---------------- Plotting -----------------

def _format_ticks(values: List[Any], name: str) -> List[str]:
    out = []
    for v in values:
        if name == 'learning_rate':
            try:
                out.append(f"{float(v):.0e}")
            except Exception:
                out.append(str(v))
        else:
            out.append(str(v))
    return out

def _compute_global_bounds(values: np.ndarray, clip_quantiles=(0.02, 0.98)) -> tuple[float,float]:
    """Robust global bounds with optional quantile clipping to avoid skew from outliers."""
    lo_q, hi_q = clip_quantiles
    if 0 <= lo_q < hi_q <= 1:
        vmin = float(np.quantile(values, lo_q))
        vmax = float(np.quantile(values, hi_q))
    else:
        vmin = float(np.min(values))
        vmax = float(np.max(values))
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        raise ValueError("Non-finite bounds computed for heatmaps")
    if vmin == vmax:  # widen slightly to avoid zero range
        vmax = vmin * 1.001 if vmin != 0 else 1e-12
    return vmin, vmax

def plot_pair_heatmaps(
    pairs_df: pd.DataFrame,
    cat_orders: Dict[str,List[Any]],
    out_path: Path,
    cmap='viridis_r',
    quantile_clip=(0.02, 0.98),
    log_scale: bool = True,
    annotate_threshold: int = 225,
) -> None:
    # Prepare data (log if requested for compression of dynamic range)
    plot_df = pairs_df.copy()
    if log_scale:
        plot_df['agg_loss_plot'] = np.log10(plot_df['agg_loss'])
        color_label = 'log10(Min Validation Loss)' if pairs_df['agg_fn'].iloc[0] == 'min' else 'log10(Median Validation Loss)'
    else:
        plot_df['agg_loss_plot'] = plot_df['agg_loss']
        color_label = 'Min Validation Loss' if pairs_df['agg_fn'].iloc[0] == 'min' else 'Median Validation Loss'

    # Global bounds with quantile clipping
    g_vmin, g_vmax = _compute_global_bounds(plot_df['agg_loss_plot'].values, clip_quantiles=quantile_clip)
    pairs = pairs_df[['hyper_a','hyper_b']].drop_duplicates().values.tolist()
    n_pairs = len(pairs)
    # Layout: approximate square
    n_cols = int(np.ceil(np.sqrt(n_pairs)))
    n_rows = int(np.ceil(n_pairs / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.6*n_cols, 3.6*n_rows), constrained_layout=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([[axes]])
    axes = axes.flatten()

    for idx, (ha, hb) in enumerate(pairs):
        ax = axes[idx]
        order_a = cat_orders[ha]
        order_b = cat_orders[hb]
        # build matrix
        mat = np.full((len(order_a), len(order_b)), np.nan)
        sub = plot_df[(plot_df.hyper_a==ha) & (plot_df.hyper_b==hb)]
        for _, r in sub.iterrows():
            ia = order_a.index(r.val_a) if r.val_a in order_a else None
            ib = order_b.index(r.val_b) if r.val_b in order_b else None
            if ia is not None and ib is not None:
                mat[ia, ib] = r.agg_loss_plot
        # Determine per-plot bounds if requested and there is sufficient dynamic range
        local_vmin, local_vmax = g_vmin, g_vmax
        finite_vals = mat[np.isfinite(mat)]
        if finite_vals.size > 0 and np.nanmax(finite_vals) > np.nanmin(finite_vals) * 1.05:  # >5% spread
            lvmin, lvmax = _compute_global_bounds(finite_vals, clip_quantiles=(0.0,1.0))
            local_vmin, local_vmax = lvmin, lvmax
        im = ax.imshow(mat, aspect='auto', origin='lower', cmap=cmap, vmin=local_vmin, vmax=local_vmax)
        # annotations (optional) only if small
        if mat.size <= annotate_threshold:  # threshold cells
            for i in range(len(order_a)):
                for j in range(len(order_b)):
                    val = mat[i,j]
                    if np.isfinite(val):
                        # convert back from log if log_scale for label clarity
                        label_val = (10**val) if log_scale else val
                        ax.text(j, i,
                                f"{label_val:.2e}" if label_val < 1e-2 else f"{label_val:.3f}",
                                ha='center', va='center', fontsize=6,
                                color='white' if (val - local_vmin)/(local_vmax-local_vmin+1e-12) > 0.5 else 'black')
        ax.set_title(f"{ha} vs {hb}")
        ax.set_yticks(range(len(order_a)))
        ax.set_yticklabels(_format_ticks(order_a, ha), fontsize=8)
        ax.set_xticks(range(len(order_b)))
        ax.set_xticklabels(_format_ticks(order_b, hb), fontsize=8, rotation=45, ha='right')
        # Outline missing cells subtly
        # (Optional enhancement: hatch could be added, kept simple here)
    # Remove unused axes
    for j in range(idx+1, len(axes)):
        axes[j].axis('off')
    # Shared colorbar
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=g_vmin, vmax=g_vmax), cmap=cmap), ax=axes.tolist(), shrink=0.6)
    cbar.set_label(color_label)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    title_metric = 'Min' if pairs_df['agg_fn'].iloc[0] == 'min' else 'Median'
    fig.suptitle(f'Pairwise Hyperparameter Performance ({title_metric} Validation Loss)' + (" [log10]" if log_scale else ""), fontsize=14)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved heatmap grid to {out_path}")

    # Removed alternate per-plot/global-only heatmap output per user request.

# ---------------- Main -----------------

def main():
    print(f"Collecting experiments from {BASE_DIR}")
    df = collect_df(BASE_DIR)
    print(f"Loaded {len(df)} experiment rows")
    pairs_df, cat_orders = aggregate_pairwise(df, agg_fn='min')
    pairs_df.to_csv(OUT_CSV, index=False)
    print(f"Pairwise summary saved to {OUT_CSV} (rows={len(pairs_df)})")
    # Only generate single log-scale output
    plot_pair_heatmaps(
        pairs_df,
        cat_orders,
        OUT_PNG_LOG,
        cmap='viridis_r',
        quantile_clip=(0.01,0.99),
        log_scale=True,
    )

if __name__ == '__main__':
    main()
