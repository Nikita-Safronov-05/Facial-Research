import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Minimal settings
CSV_PATH = "cca_results/XYZPCA/seed_001/X_vs_XYZPCA_summary.csv"
THRESHOLD = 1e-5  # FDR q threshold
EPS = 1e-18     # to avoid -log10(0)

# Load data
df = pd.read_csv(CSV_PATH)

# Extract feature indices (assumes names like 'x123')
df["feature_index"] = df["feature"].astype(str).str.extract(r"x(\d+)").astype(int)

# Values and acceptance
q = df["fdr_q"].astype(float).to_numpy()
accepted = q <= THRESHOLD

# -log10 transform for visibility
y = -np.log10(np.clip(q, EPS, 1.0))
thresh_y = -np.log10(max(THRESHOLD, EPS))

# Plot
plt.figure(figsize=(12, 6))
plt.scatter(df["feature_index"], y, c=np.where(accepted, "red", "blue"), s=14, alpha=0.8)

# Highlight original injected signals: 1, 101, 201, 301
highlight_idxs = [1, 101, 201, 301]
mask = df["feature_index"].isin(highlight_idxs)
plt.scatter(
	df.loc[mask, "feature_index"],
	y[mask],
	c="limegreen",
	s=60,
	edgecolors="black",
	linewidths=0.8,
	zorder=3,
	label="Original signals (1,101,201,301)",
)
plt.axhline(thresh_y, color="gray", linestyle="--", linewidth=1.0)
plt.xlabel("Feature Index")
plt.ylabel("-log10(FDR q)")
plt.title(f"Feature significance (-log10 q); threshold={THRESHOLD:g} | accepted={accepted.sum()}/{len(df)}")
plt.legend(frameon=False)
plt.tight_layout()
plt.show()
