"""Plot latent dimension vs average validation loss across experiment seeds.

Looks for experiment result pickle files inside:
  encoder_files/encoder_data/Experiments/SEED*/experiments/*.pkl

Each pickle is expected to contain keys written by the grid search scripts, e.g.:
  - 'latent_dim'
  - 'architecture': {'activation': <ActivationName>, ...}
  - 'avg_best_val_loss' (preferred) OR fallback derivable from 'val_losses_per_fold'

The script:
  1. Recursively loads all .pkl result files
  2. Extracts latent_dim, activation name, and a scalar validation loss
  3. Produces a log-log scatter plot with color per activation
  4. Annotates the global loss range
  5. Writes plot to encoder_files/encoder_data/Experiments/latent_vs_val_loss.png

Environment variable overrides:
  EXPERIMENT_BASE : Override the default base directory path.

Run (PowerShell example from repo root):
    # Dry run (only list how many records would be plotted)
    python .\encoder_files\plot_latent_vs_val_loss.py --dry-run

    # Generate plot (explicit flag prevents accidental auto-run on import)
    python .\encoder_files\plot_latent_vs_val_loss.py --run

Optional: set EXPERIMENT_BASE if your layout differs
  $env:EXPERIMENT_BASE = "c:/path/to/encoder_files/encoder_data/Experiments"; \
    python .\encoder_files\plot_latent_vs_val_loss.py
"""
from __future__ import annotations
import os
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Any, Dict, List

# ---------------- Configuration (simplified to relative paths) ----------------
# Assume this script is run from repository root OR via python -m within encoder_files/.
# Use paths relative to the script location without environment overrides.
EXPERIMENTS_DIR = "XYZ" 
SCRIPT_DIR = Path(__file__).parent  # do not resolve to keep relative semantics simple
BASE_DIR = SCRIPT_DIR / "encoder_data" / EXPERIMENTS_DIR
SEED_PREFIX = "SEED"
PKL_SUBDIR = "experiments"  # inside each SEEDXX
PLOT_PATH = BASE_DIR / "latent_vs_val_loss.png"

# Activation color palette
COLOR_MAP = {
    # New palette: purple & teal/green (colorblind-friendly and aesthetically softer).
    # Purple:  #7b3294  (from Brewer PiYG extremes)
    # Teal:    #008837  (contrasting green)
    # Overlap tends toward muted dark neutral, preserving additive mixing cue.
    "ELU": "#7b3294",        # purple
    "LeakyReLU": "#008837",  # green
}

# Alternative palettes (uncomment to experiment):
# COLOR_MAP = {"ELU": "#e41a1c", "LeakyReLU": "#377eb8"}  # classic red/blue
# COLOR_MAP = {"ELU": "#984ea3", "LeakyReLU": "#4daf4a"}  # purple/green balanced
# COLOR_MAP = {"ELU": "#af8dc3", "LeakyReLU": "#7fbf7b"}  # softer pastel version

# ---------------- Helpers ----------------

def _extract_loss(rec: Dict[str, Any]) -> float | None:
    """Extract a representative validation loss from a result record."""
    if "avg_best_val_loss" in rec:
        return float(rec["avg_best_val_loss"])
    if "best_val_loss" in rec:
        return float(rec["best_val_loss"])
    # Fallback: derive from val_losses_per_fold -> min per fold then average
    vpf = rec.get("val_losses_per_fold")
    if isinstance(vpf, (list, tuple)) and vpf:
        mins = []
        for fold_losses in vpf:
            try:
                arr = np.asarray(fold_losses, dtype=float)
                if arr.size:
                    mins.append(arr.min())
            except Exception:
                continue
        if mins:
            return float(np.mean(mins))
    return None

# ---------------- Data Collection ----------------

def collect_results(base: Path) -> pd.DataFrame:
    if not base.exists():
        raise FileNotFoundError(f"Experiments base directory not found: {base}")
    seed_dirs = [d for d in base.iterdir() if d.is_dir() and d.name.startswith(SEED_PREFIX)]
    if not seed_dirs:
        raise FileNotFoundError(f"No seed directories (e.g. SEED39) under {base}")

    rows: List[Dict[str, Any]] = []
    for sdir in seed_dirs:
        exp_dir = sdir / PKL_SUBDIR
        if not exp_dir.exists():
            # allow legacy layout where pkls might be directly under seed dir
            candidate_dirs = [exp_dir, sdir]
        else:
            candidate_dirs = [exp_dir]
        for cdir in candidate_dirs:
            for pkl in cdir.glob("*.pkl"):
                try:
                    with open(pkl, "rb") as f:
                        rec = pickle.load(f)
                except Exception as e:
                    print(f"[WARN] Failed to load {pkl}: {e}")
                    continue
                latent_dim = rec.get("latent_dim")
                arch = rec.get("architecture", {}) or {}
                activation = arch.get("activation", "UNKNOWN")
                loss = _extract_loss(rec)
                if latent_dim is None or loss is None or not np.isfinite(loss) or loss <= 0:
                    print(f"[SKIP] Missing/invalid metrics in {pkl.name}")
                    continue
                rows.append({
                    "latent_dim": int(latent_dim),
                    "activation": str(activation),
                    "val_loss": float(loss),
                    "seed": sdir.name,
                    "file": pkl.name,
                })
    if not rows:
        raise RuntimeError("No valid experiment records collected.")
    return pd.DataFrame(rows)

# ---------------- Plotting ----------------

def make_compact_plot(
    df: pd.DataFrame,
    out_path: Path,
    marker_size: int = 70,
    jitter: float = 0.3,
    seed: int = 123,
    compact_pad: float = 0.08,
    fig_width: float = 11.0,
    fig_height: float = 6.5,
) -> None:
    # Filter to positive losses for log plotting
    df = df[df["val_loss"] > 0]
    if df.empty:
        raise RuntimeError("No positive validation losses to plot.")

    rng = np.random.default_rng(seed)
    plt.figure(figsize=(fig_width, fig_height))
    ordered_dims = sorted(df['latent_dim'].unique())
    dim_to_ix = {d: i for i, d in enumerate(ordered_dims)}
    base_positions = df['latent_dim'].map(dim_to_ix).astype(float)
    if jitter > 0:
        offsets = (rng.random(len(base_positions)) - 0.5) * jitter
        base_positions = base_positions + offsets

    # Additive color mixing approach:
    #  1. Use two equally prominent colors (blue/orange) so overlap visually blends toward brown/gray.
    #  2. Concatenate both activations, shuffle rows, and plot in a single pass with low alpha.
    #     This avoids systematic dominance by one activation.
    #  3. Keep identical marker size & alpha across groups.
    alpha_val = 0.22  # small bump to compensate slightly darker palette
    size_val = marker_size
    work = df.copy()
    work['x_pos'] = base_positions
    # Random global shuffle for unbiased layering
    rng_layer = np.random.default_rng(seed + 999)
    perm = rng_layer.permutation(len(work))
    work = work.iloc[perm]
    colors = work['activation'].map(lambda a: COLOR_MAP.get(a, '#555555')).values
    plt.scatter(
        work['x_pos'].values,
        work['val_loss'].values,
        s=size_val,
        c=colors,
        alpha=alpha_val,
        edgecolors='none'
    )
    # Create legend manually to reflect chosen palette
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0],[0], marker='o', color='none', label='LeakyReLU', markerfacecolor=COLOR_MAP['LeakyReLU'], markersize=8, alpha=0.7),
        Line2D([0],[0], marker='o', color='none', label='ELU', markerfacecolor=COLOR_MAP['ELU'], markersize=8, alpha=0.7),
    ]
    plt.legend(handles=legend_handles, title="Activation Function")

    plt.yscale('log')
    plt.xticks(range(len(ordered_dims)), [str(d) for d in ordered_dims], rotation=0)
    plt.xlabel('Latent Dimension')
    plt.xlim(-compact_pad * len(ordered_dims), len(ordered_dims) - 1 + compact_pad * len(ordered_dims))
    plt.ylabel("Average Validation Loss (log scale)")
    plt.title("Latent Dimension vs Average Validation Loss")
    # Legend already added above

    loss_min = df["val_loss"].min()
    loss_max = df["val_loss"].max()
    plt.figtext(0.01, 0.01, f"Loss range: {loss_min:.2e} - {loss_max:.2e}", fontsize=10, ha="left", va="bottom")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved plot to: {out_path}")
    print(f"Records plotted: {len(df)} from {df['seed'].nunique()} seeds")

# ---------------- Main ----------------

def main():
    print(f"Scanning experiment results under (relative): {BASE_DIR}")
    df = collect_results(BASE_DIR)
    print(f"Discovered {len(df)} valid records across {df['seed'].nunique()} seeds")
    out_path = PLOT_PATH
    make_compact_plot(df, out_path)
    # Also write a summary CSV for quick inspection
    summary_csv = BASE_DIR / "latent_val_loss_summary.csv"
    summary = df.groupby(['latent_dim','activation'])['val_loss'].agg(['count','min','median','max']).reset_index()
    summary.to_csv(summary_csv, index=False)
    print(f"Summary written to: {summary_csv}")

if __name__ == "__main__":
    main()
