import os
import random
from pathlib import Path
from typing import Tuple, Dict
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler


def set_seeds(seed: int = 42) -> None:
    import torch
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


def pvals_from_corr(r: np.ndarray, n: int) -> np.ndarray:
    from scipy import stats
    r = np.clip(r, -0.999999, 0.999999)
    t = r * np.sqrt((n - 2) / (1 - r ** 2))
    p = 2 * (1 - stats.t.cdf(np.abs(t), df=n - 2))
    return p


def bh_fdr(pvals: np.ndarray, alpha: float = 1e-5) -> Tuple[np.ndarray, np.ndarray]:
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


def per_feature_pvalues_via_cca(
    X: np.ndarray,
    Y: np.ndarray,
    n_components: int | None = None,
    fdr_alpha: float = 1e-5,
) -> Dict[str, np.ndarray]:
    """Run CCA and compute per-feature p-values vs Y's canonical variates.

    Returns dict with pvals_per_component, corrs_per_component, min_pval, fdr_q, reject.
    """
    # Standardize blocks
    Xs = StandardScaler().fit_transform(X)
    Ys = StandardScaler().fit_transform(Y)
    n, px = Xs.shape
    _, py = Ys.shape

    if n_components is None:
        n_components = min(px, py)

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
    q, reject = bh_fdr(min_p, alpha=fdr_alpha)
    return {
        "pvals_per_component": pmat,
        "corrs_per_component": rmat,
        "min_pval": min_p,
        "fdr_q": q,
        "reject": reject,
    }
