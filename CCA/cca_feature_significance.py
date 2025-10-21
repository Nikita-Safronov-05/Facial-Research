import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler
from encoder_files.curvature_autoencoder import CurvatureAutoencoder


def set_seeds(seed: int = 42):
    import random, os
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"


def build_model(ckpt_path: Path, device: torch.device) -> CurvatureAutoencoder:
    ckpt = torch.load(ckpt_path, map_location=device)
    model = CurvatureAutoencoder(
        input_dim=ckpt["input_dim"],
        latent_dim=ckpt["latent_dim"],
        encoder_layers=ckpt["encoder_layers"],
        decoder_layers=ckpt["decoder_layers"],
        activation=getattr(nn, ckpt["activation_name"]) if isinstance(ckpt["activation_name"], str) else ckpt["activation_name"],
    )
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device)
    model.eval()
    return model


def reconstruct(model: CurvatureAutoencoder, X: np.ndarray, batch_size: int = 256, device: str | torch.device = "cpu") -> np.ndarray:
    device = torch.device(device)
    model = model.to(device)
    outs = []
    with torch.no_grad():
        for i in range(0, X.shape[0], batch_size):
            xb = torch.from_numpy(X[i : i + batch_size]).float().to(device)
            xhat = model(xb)
            outs.append(xhat.cpu().numpy())
    return np.vstack(outs)


def pvals_from_corr(r: np.ndarray, n: int) -> np.ndarray:
    from scipy import stats
    r = np.clip(r, -0.999999, 0.999999)
    t = r * np.sqrt((n - 2) / (1 - r ** 2))
    p = 2 * (1 - stats.t.cdf(np.abs(t), df=n - 2))
    return p


def bh_fdr(pvals: np.ndarray, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    m = pvals.size
    order = np.argsort(pvals)
    ranked = pvals[order]
    factors = np.arange(1, m + 1) / m
    q_ordered = np.minimum.accumulate((ranked / factors)[::-1])[::-1]
    q_ordered = np.clip(q_ordered, 0.0, 1.0)
    q = np.empty_like(q_ordered)
    q[order] = q_ordered
    reject = q <= alpha
    return q, reject


def per_feature_pvalues_via_cca(X: np.ndarray, Y: np.ndarray, n_components: int | None = None):
    # Standardize blocks
    Xs = StandardScaler().fit_transform(X)
    Ys = StandardScaler().fit_transform(Y)
    n, px = Xs.shape
    _, py = Ys.shape

    if n_components is None:
        n_components = min(px, py, 5)

    cca = CCA(n_components=n_components, max_iter=1000)
    U, V = cca.fit_transform(Xs, Ys)

    # Score each feature j by its correlation with canonical variates of Y
    pmat = np.zeros((px, n_components))
    rmat = np.zeros((px, n_components))
    for k in range(n_components):
        v = (V[:, k] - V[:, k].mean()) / (V[:, k].std() + 1e-12)
        for j in range(px):
            xj = Xs[:, j]
            xj = (xj - xj.mean()) / (xj.std() + 1e-12)
            r = float(np.corrcoef(xj, v)[0, 1])
            rmat[j, k] = r
        pmat[:, k] = pvals_from_corr(rmat[:, k], n)

    # Aggregate per feature via min p across components, then FDR
    min_p = pmat.min(axis=1)
    q, reject = bh_fdr(min_p, alpha=1e-5)
    return {
        "pvals_per_component": pmat,
        "corrs_per_component": rmat,
        "min_pval": min_p,
        "fdr_q": q,
        "reject": reject,
    }


def main():

    # Inputs
    ckpt_path = Path("encoder_data/retrained_best_autoencoder_full.pth")
    curv_np = Path("../processed/2000_interior_curvatures.npy")
    sim_X_csv = Path("../given_data/sim_X_2000.csv")  # Original 1000 features X

    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)
    if not curv_np.exists():
        raise FileNotFoundError(curv_np)
    if not sim_X_csv.exists():
        raise FileNotFoundError(sim_X_csv)

    # Reproducibility and device
    set_seeds(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    Y_orig = np.load(curv_np)  # (N, D)
    X_feats = pd.read_csv(sim_X_csv).values     # (N, 1000)

    n = min(Y_orig.shape[0], X_feats.shape[0])
    Y_orig = Y_orig[:n]
    X_feats = X_feats[:n]

    # Build model and reconstruct faces
    model = build_model(ckpt_path, device=device)
    Y_recon = reconstruct(model, Y_orig, batch_size=256, device=device)  # (N, D)

    # 1) CCA: features X vs original curvatures Y_orig
    res_orig = per_feature_pvalues_via_cca(X_feats, Y_orig)

    # 2) CCA: features X vs reconstructed curvatures Y_recon
    res_recon = per_feature_pvalues_via_cca(X_feats, Y_recon)

    # Save outputs
    os.makedirs("encoder_data/cca", exist_ok=True)
    feat_names = [f"x{j+1}" for j in range(X_feats.shape[1])]
    comp_names = [f"comp_{k+1}" for k in range(res_orig["pvals_per_component"].shape[1])]

    # Per-component
    pd.DataFrame(res_orig["pvals_per_component"], index=feat_names, columns=comp_names).to_csv(
        "encoder_data/cca/features_vs_Yorig_pvalues_per_component.csv"
    )
    pd.DataFrame(res_recon["pvals_per_component"], index=feat_names, columns=comp_names).to_csv(
        "encoder_data/cca/features_vs_Yrecon_pvalues_per_component.csv"
    )

    # Summaries
    pd.DataFrame({
        "feature": feat_names,
        "min_pval": res_orig["min_pval"],
        "fdr_q": res_orig["fdr_q"],
        "reject_fdr_0.05": res_orig["reject"],
    }).sort_values("min_pval").to_csv("encoder_data/cca/features_vs_Yorig_summary.csv", index=False)

    pd.DataFrame({
        "feature": feat_names,
        "min_pval": res_recon["min_pval"],
        "fdr_q": res_recon["fdr_q"],
        "reject_fdr_0.05": res_recon["reject"],
    }).sort_values("min_pval").to_csv("encoder_data/cca/features_vs_Yrecon_summary.csv", index=False)

    print("CCA feature significance complete.")
    print("Saved results to encoder_data/cca/")


if __name__ == "__main__":
    main()
