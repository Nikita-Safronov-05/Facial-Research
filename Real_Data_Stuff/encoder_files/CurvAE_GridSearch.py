from __future__ import annotations
import os
import pickle
from pathlib import Path
from datetime import datetime
import time
import platform
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, TensorDataset
from sklearn.model_selection import train_test_split, KFold
from concurrent.futures import ThreadPoolExecutor, as_completed

# Local import safety for direct run
import sys
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from CurvAE import CurvAE

# ------------------------------
# 1. Load Real-Data Curvatures
# ------------------------------
ROOT = HERE.parent  # Real_Data_Stuff/
OUT = ROOT / "outputs"
CURV_PATH = OUT / "interior_curvatures.npy"
MASK_PATH = OUT / "interior_mask.npy"

if not CURV_PATH.exists():
    raise FileNotFoundError(f"Not found: {CURV_PATH}. Run run_compute_curvatures.py first.")

X = np.load(CURV_PATH)  # shape: (N, D_interior)
X_tensor = torch.tensor(X, dtype=torch.float32)
full_indices = np.arange(X.shape[0])
full_dataset = TensorDataset(X_tensor)
input_dim = X.shape[1]

# ------------------------------
# 2. Deterministic base seed for split
# ------------------------------
BASE_SPLIT_SEED = 1
random.seed(BASE_SPLIT_SEED)
np.random.seed(BASE_SPLIT_SEED)
torch.manual_seed(BASE_SPLIT_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(BASE_SPLIT_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
try:
    torch.use_deterministic_algorithms(True)
except Exception:
    pass

SAVE_ROOT = HERE / "encoder_data" / "Curvature"
SAVE_ROOT.mkdir(parents=True, exist_ok=True)

# Fixed held-out test split; reused across all experiments
TEST_SIZE = 0.1
trainval_idx, test_idx = train_test_split(full_indices, test_size=TEST_SIZE,
                                         random_state=BASE_SPLIT_SEED, shuffle=True)
trainval_dataset = Subset(full_dataset, trainval_idx)

with open(SAVE_ROOT / "test_indices.pkl", "wb") as f:
    pickle.dump(test_idx, f)

print(f"Total: {len(full_dataset)}, Train+Val: {len(trainval_dataset)}, Test: {len(test_idx)}")

# Device summary print so we can verify CUDA usage
def _device_summary_print():
    cuda_avail = torch.cuda.is_available()
    dev = torch.device("cuda" if cuda_avail else "cpu")
    cuda_ver = getattr(torch.version, "cuda", None)
    torch_ver = torch.__version__
    msg = [f"[device] using={dev}, cuda_available={cuda_avail}, torch={torch_ver}, cuda_version={cuda_ver}"]
    if cuda_avail:
        try:
            idx = torch.cuda.current_device()
        except Exception:
            idx = 0
        try:
            name = torch.cuda.get_device_name(idx)
        except Exception:
            name = None
        try:
            count = torch.cuda.device_count()
        except Exception:
            count = None
        msg.append(f"gpu_index={idx}, gpu_name={name}, gpu_count={count}")
    print(" ".join(m for m in msg if m))

_device_summary_print()

# ------------------------------
# 3. Architectures compatible with latent dims (<500 kept)
# ------------------------------

def get_compatible_encoders(latent_dim: int) -> list[list[int]]:
    """Return encoder layer sizes based on latent-dim range.
    We keep the same families we used for dims < 500 in sim experiments.
    """
    encs: list[list[int]] = []
    if 10 <= latent_dim <= 20:
        encs.extend([
            [1024, 256, 64],
            [2048, 512, 128],
            [1024, 512, 256, 64],
        ])
    elif 30 <= latent_dim <= 120:
        encs.extend([
            [2048, 512],
            [4096, 1024],
            [2048, 1024, 256],
            [4096, 2048, 512],
        ])
    elif 200 <= latent_dim <= 400:
        encs.extend([
            [4096, 2048],
            [3400, 1700],
            [4096, 2048, 1024],
        ])
    else:
        # Fallback: a conservative 2-layer if outside specified bands
        encs.append([2048, 1024])
    return encs


def arch_repr(encoder_layers: list[int], decoder_layers: list[int], activation: type[nn.Module]):
    return {
        "encoder_layers": encoder_layers,
        "decoder_layers": decoder_layers,
        "activation": activation.__name__ if hasattr(activation, "__name__") else str(activation),
    }

# ------------------------------
# 4. Training function
# ------------------------------

def train_autoencoder(model: nn.Module, train_loader, val_loader,
                      patience: int = 5, max_epochs: int = 200, lr: float = 1e-3,
                      device: torch.device | None = None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
    crit = nn.MSELoss()

    best_val = float("inf")
    patience_ctr = 0
    best_epoch = 0
    train_losses: list[float] = []
    val_losses: list[float] = []

    for epoch in range(max_epochs):
        model.train()
        tr_loss = 0.0
        n_tr = 0
        for (xb,) in train_loader:
            xb = xb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), xb)
            loss.backward()
            opt.step()
            bs = xb.size(0)
            tr_loss += loss.item() * bs
            n_tr += bs
        tr_loss /= max(1, n_tr)

        model.eval()
        va_loss = 0.0
        n_va = 0
        with torch.no_grad():
            for (xb,) in val_loader:
                xb = xb.to(device)
                loss = crit(model(xb), xb)
                bs = xb.size(0)
                va_loss += loss.item() * bs
                n_va += bs
        va_loss /= max(1, n_va)

        train_losses.append(float(tr_loss))
        val_losses.append(float(va_loss))
        sched.step(va_loss)

        if va_loss < best_val:
            best_val = va_loss
            patience_ctr = 0
            best_epoch = epoch + 1
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                break

    return float(best_val), int(best_epoch), train_losses, val_losses


# ------------------------------
# 5. Single fold worker
# ------------------------------

def train_single_fold(cfg):
    (fold_idx, train_idx, val_idx, dataset, input_dim, latent_dim,
     enc_layers, dec_layers, activation, batch_size, patience,
     max_epochs, lr, seed) = cfg

    # Seed determinism per experiment
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass

    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size, shuffle=False)

    model = CurvAE(input_dim, latent_dim, enc_layers, dec_layers, activation)
    return train_autoencoder(model, train_loader, val_loader,
                             patience=patience, max_epochs=max_epochs, lr=lr)


# ------------------------------
# 6. K-fold runner
# ------------------------------

def run_kfold_cv(dataset, input_dim: int, latent_dim: int,
                 encoder_layers: list[int], decoder_layers: list[int], activation: type[nn.Module],
                 batch_size: int, patience: int, max_epochs: int, lr: float,
                 k_folds: int, seed: int):
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
    idxs = np.arange(len(dataset))
    fold_splits = list(kf.split(idxs))

    fold_results = {}
    with ThreadPoolExecutor(max_workers=1) as ex:  # single worker = deterministic, low overhead
        futures = {}
        for fold_idx, (tr, va) in enumerate(fold_splits):
            cfg = (fold_idx, tr, va, dataset, input_dim, latent_dim,
                   encoder_layers, decoder_layers, activation, batch_size,
                   patience, max_epochs, lr, seed)
            futures[ex.submit(train_single_fold, cfg)] = fold_idx
        for fut in as_completed(futures):
            fidx = futures[fut]
            best_val, best_epoch, tr_losses, va_losses = fut.result()
            fold_results[fidx] = {
                "val_loss": best_val,
                "best_epoch": best_epoch,
                "train_losses": tr_losses,
                "val_losses": va_losses,
            }
            print(f"  Fold {fidx+1} best val loss: {best_val:.6f}")

    vals = [fold_results[i]["val_loss"] for i in sorted(fold_results.keys())]
    epochs = [fold_results[i]["best_epoch"] for i in sorted(fold_results.keys())]

    return {
        "latent_dim": latent_dim,
        "patience": patience,
        "architecture": arch_repr(encoder_layers, decoder_layers, activation),
        "k_folds": k_folds,
        "optimizer": "Adam",
        "learning_rate": lr,
        "seed": seed,
        "batch_size": batch_size,
        "best_val_losses_per_fold": vals,
        "avg_best_val_loss": float(np.mean(vals)),
        "std_best_val_loss": float(np.std(vals)),
        "best_epochs_per_fold": epochs,
        "avg_best_epoch": float(np.mean(epochs)),
    }


# ------------------------------
# 7. Grid spec (from sim-data learnings)
# ------------------------------
# Right-skewed latent dims (≈10 values, 10..400)
LATENT_DIMS = [10, 15, 20, 30, 40, 60, 80, 120, 200, 400]
# Learning rates, batch sizes, patience, activations
LEARNING_RATES = [1e-4, 3e-4, 1e-3]
BATCH_SIZES = [16, 32]
PATIENCES = [5, 8]
ACTIVATIONS = [nn.ELU, nn.LeakyReLU]
# Seeds consistent with earlier experiments
SEEDS = [39, 40, 41]
# Default CV
K_FOLDS = 3
MAX_EPOCHS = 400


def run_main_grid_search(latent_dims=None, seeds=None, max_epochs=MAX_EPOCHS, k_folds=K_FOLDS,
                         learning_rates=LEARNING_RATES, batch_sizes=BATCH_SIZES,
                         patiences=PATIENCES, activations=ACTIVATIONS):
    latent_dims = LATENT_DIMS if latent_dims is None else latent_dims
    seeds = SEEDS if seeds is None else seeds

    meta_rows = []
    exp_counter = 0

    for ld in latent_dims:
        encoders = get_compatible_encoders(ld)
        print(f"Latent {ld}: {len(encoders)} compatible encoder depths")
        for enc_layers in encoders:
            dec_layers = list(enc_layers)[::-1]
            for act in activations:
                for pat in patiences:
                    for bs in batch_sizes:
                        for lr in learning_rates:
                            for seed in seeds:
                                # Ensure seed-specific dir exists
                                seed_dir = SAVE_ROOT / f"SEED{seed}" / "experiments"
                                seed_dir.mkdir(parents=True, exist_ok=True)

                                # Print per-run device info to confirm CUDA usage
                                dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                                print(f"[train] device={dev}, cuda={torch.cuda.is_available()}, ld={ld}, bs={bs}, lr={lr}, pat={pat}, act={act.__name__}, seed={seed}")

                                started_at = datetime.utcnow().isoformat() + "Z"
                                t0 = time.time()

                                res = run_kfold_cv(
                                    trainval_dataset,
                                    input_dim=input_dim,
                                    latent_dim=ld,
                                    encoder_layers=enc_layers,
                                    decoder_layers=dec_layers,
                                    activation=act,
                                    batch_size=bs,
                                    patience=pat,
                                    max_epochs=max_epochs,
                                    lr=lr,
                                    k_folds=k_folds,
                                    seed=seed,
                                )

                                finished_at = datetime.utcnow().isoformat() + "Z"
                                train_seconds = float(time.time() - t0)

                                model_id = (
                                    f"ld{ld}-pat{pat}-bs{bs}-lr{lr:.0e}-seed{seed}"
                                    f"-enc{'-'.join(map(str, enc_layers))}"
                                    f"-dec{'-'.join(map(str, dec_layers))}"
                                    f"-act{act.__name__}"
                                )
                                out_pkl = seed_dir / f"{model_id}.pkl"
                                with open(out_pkl, "wb") as f:
                                    pickle.dump(res, f)

                                # Collect system / device metadata
                                cuda_ver = getattr(torch.version, "cuda", None)
                                cudnn_ver = getattr(torch.backends.cudnn, "version", lambda: None)()
                                torch_ver = torch.__version__
                                cuda_avail = torch.cuda.is_available()
                                if cuda_avail:
                                    try:
                                        gpu_idx = torch.cuda.current_device()
                                    except Exception:
                                        gpu_idx = 0
                                    try:
                                        gpu_name = torch.cuda.get_device_name(gpu_idx)
                                    except Exception:
                                        gpu_name = None
                                    try:
                                        gpu_cc = torch.cuda.get_device_capability(gpu_idx)
                                    except Exception:
                                        gpu_cc = None
                                    try:
                                        props = torch.cuda.get_device_properties(gpu_idx)
                                        gpu_mem = getattr(props, 'total_memory', None)
                                    except Exception:
                                        gpu_mem = None
                                else:
                                    gpu_idx = None
                                    gpu_name = None
                                    gpu_cc = None
                                    gpu_mem = None

                                sys_info = {
                                    "platform": platform.platform(),
                                    "python_version": platform.python_version(),
                                }

                                meta = {
                                    "model_id": model_id,
                                    "latent_dim": ld,
                                    "patience": pat,
                                    "encoder_layers": enc_layers,
                                    "decoder_layers": dec_layers,
                                    "activation": act.__name__,
                                    "batch_size": bs,
                                    "learning_rate": lr,
                                    "avg_best_val_loss": res["avg_best_val_loss"],
                                    "std_best_val_loss": res["std_best_val_loss"],
                                    "seed": seed,
                                    "result_file": str(out_pkl),
                                    # Timing
                                    "started_at_utc": started_at,
                                    "finished_at_utc": finished_at,
                                    "train_seconds": train_seconds,
                                    # Dataset / split
                                    "input_dim": int(input_dim),
                                    "n_samples": int(len(trainval_dataset) + len(test_idx)),
                                    "test_size": TEST_SIZE,
                                    "k_folds": k_folds,
                                    "base_split_seed": BASE_SPLIT_SEED,
                                    # Device / runtime
                                    "torch_version": torch_ver,
                                    "cuda_version": cuda_ver,
                                    "cudnn_version": cudnn_ver,
                                    "cuda_available": cuda_avail,
                                    "gpu_index": gpu_idx,
                                    "gpu_name": gpu_name,
                                    "gpu_compute_capability": gpu_cc,
                                    "gpu_total_memory_bytes": gpu_mem,
                                    **sys_info,
                                }
                                meta_rows.append(meta)
                                exp_counter += 1
                                print(f"Stored experiment {exp_counter}: {out_pkl}")

    meta_csv = SAVE_ROOT / "experiment_metadata.csv"
    pd.DataFrame(meta_rows).to_csv(meta_csv, index=False)
    print(f"[done] Wrote metadata: {meta_csv}")
    return meta_rows


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Grid search CurvAE on real-data curvatures.")
    ap.add_argument("--quick", action="store_true", help="Tiny run for sanity-check (3 configs)")
    ap.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    ap.add_argument("--k-folds", type=int, default=K_FOLDS)
    args = ap.parse_args()

    if args.quick:
        dims = [10, 40]
        seeds = [39]
        lrs = [3e-4]
        bss = [16]
        pats = [5]
        acts = [nn.ELU]
    else:
        dims = LATENT_DIMS
        seeds = SEEDS
        lrs = LEARNING_RATES
        bss = BATCH_SIZES
        pats = PATIENCES
        acts = ACTIVATIONS

    run_main_grid_search(latent_dims=dims, seeds=seeds, max_epochs=args.max_epochs,
                         k_folds=args.k_folds, learning_rates=lrs, batch_sizes=bss,
                         patiences=pats, activations=acts)


if __name__ == "__main__":
    main()
