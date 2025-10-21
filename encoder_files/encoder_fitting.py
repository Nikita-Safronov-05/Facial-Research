import os
import ast
import argparse
import random
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

import os
import ast
import argparse
import random
import pickle
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

import matplotlib
matplotlib.use('Agg')  # headless
import matplotlib.pyplot as plt


# ----------------------------- CLI ARGUMENTS -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Retrain best autoencoder configurations for selected latent dims on Curvature or XYZ data."
    )
    parser.add_argument("--mode", choices=["Curvature", "XYZ"], default="Curvature",
                        help="Which data representation to use: Curvature or XYZ")
    parser.add_argument("--test-size", type=float, default=0.1,
                        help="Fraction for test split")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Optional cap on rows for XYZ CSV (debugging)")
    parser.add_argument("--epoch-multiplier", type=float, default=1.1,
                        help="Multiply the cross-val best epoch mean to set final training epochs")
    parser.add_argument("--eval-frequency", type=int, default=1,
                        help=(
                            ">0 evaluate test loss every N epochs; 0 only at the end; <0 disable test eval altogether"
                        ))
    parser.add_argument("--latent-dims", type=str, default="5,10,15,20,30",
                        help="Comma-separated list of latent dims to retrain")
    return parser.parse_args()


args = parse_args()


# ------------------------- Reproducibility setup -------------------------
seed = 41
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
try:
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
except Exception:
    pass


# ------------------------------- Data load -------------------------------
def load_data_matrix(mode: str, max_rows: Optional[int]) -> Tuple[np.ndarray, str]:
    if mode == "Curvature":
        path = os.path.join("..", "processed", "2000_interior_curvatures_good.npy")
        if not os.path.exists(path):
            # fallback to original name if _good not present
            path = os.path.join("..", "processed", "2000_interior_curvatures.npy")
        if not os.path.exists(path):
            raise SystemExit(f"Curvature npy not found: {path}")
        arr = np.load(path)
        # ensure 2D [n_samples, features]
        if arr.ndim > 2:
            arr = arr.reshape(arr.shape[0], -1)
        elif arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr.astype(np.float32), path
    else:
        path = os.path.join("..", "processed", "sim_2000_d4_1", "XYZ.csv")
        if not os.path.exists(path):
            raise SystemExit(f"Hardcoded XYZ CSV not found: {path}")
        print(f"Loading XYZ data from {path} ...")
        df = pd.read_csv(path)
        if max_rows is not None:
            df = df.head(max_rows)
        mat = df.to_numpy(dtype=np.float32)
        print(f"Loaded XYZ matrix shape: {mat.shape}")
        return mat, path


data_matrix, source_file = load_data_matrix(args.mode, args.max_rows)
input_dim = int(np.prod(data_matrix.shape[1:]))


# ----------------------------- Model builders ----------------------------
def activation_from_name(name) -> type[nn.Module]:
    # normalize common names
    key = str(name)
    if hasattr(name, "__name__"):
        key = name.__name__
    key = key.lower()
    if "relu" in key and "leaky" not in key:
        return nn.ReLU
    if "leakyrelu" in key or ("relu" in key and "leaky" in key):
        return nn.LeakyReLU
    if "gelu" in key:
        return nn.GELU
    if "tanh" in key:
        return nn.Tanh
    if "sigmoid" in key:
        return nn.Sigmoid
    # default
    return nn.ReLU


class MLP(nn.Module):
    def __init__(self, dims: List[int], activation: type[nn.Module], last_activation: bool = False):
        super().__init__()
        layers: List[nn.Module] = []
        for i in range(len(dims) - 1):
            in_f, out_f = dims[i], dims[i + 1]
            layers.append(nn.Linear(in_f, out_f))
            if i < len(dims) - 2 or last_activation:
                layers.append(activation())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class AutoEncoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, enc_layers: List[int], dec_layers: List[int], act: type[nn.Module]):
        super().__init__()
        enc_dims = [input_dim] + list(enc_layers) + [latent_dim]
        dec_dims = [latent_dim] + list(dec_layers) + [input_dim]
        self.encoder = MLP(enc_dims, act, last_activation=False)
        self.decoder = MLP(dec_dims, act, last_activation=False)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat


# --------------------------- Experiments loading --------------------------
def canonical_activation_name(act_val) -> str:
    if hasattr(act_val, "__name__"):
        return act_val.__name__
    if isinstance(act_val, str):
        return act_val
    return str(act_val)


def load_all_experiments(mode: str) -> pd.DataFrame:
    print("Loading all experiment results from pkl files...")
    seed_dirs = ["SEED39", "SEED40", "SEED41", "SEED42"]
    all_rows: List[dict] = []
    for seed_dir in seed_dirs:
        exp_dir = Path(f"encoder_data/{mode}/{seed_dir}/experiments")
        if not exp_dir.exists():
            print(f"Directory {exp_dir} does not exist, skipping...")
            continue
        pkl_files = list(exp_dir.glob("*.pkl"))
        print(f"{seed_dir}: found {len(pkl_files)} files")
        for pkl_file in pkl_files:
            try:
                with open(pkl_file, "rb") as f:
                    result = pickle.load(f)
            except Exception as e:
                print(f"Error loading {pkl_file}: {e}")
                continue
            if not all(k in result for k in ["latent_dim", "architecture", "batch_size", "avg_best_val_loss"]):
                print(f"Skipping {pkl_file.name}: missing fields")
                continue
            arch = result["architecture"]
            act_name = canonical_activation_name(arch.get("activation", "ReLU"))
            all_rows.append({
                "model_id": pkl_file.stem,
                "latent_dim": result["latent_dim"],
                "patience": result.get("patience"),
                "encoder_layers": arch.get("encoder_layers"),
                "decoder_layers": arch.get("decoder_layers"),
                "activation": act_name,
                "batch_size": result["batch_size"],
                "learning_rate": result.get("learning_rate"),
                "avg_best_val_loss": result["avg_best_val_loss"],
                "std_best_val_loss": result.get("std_best_val_loss", 0.0),
                "avg_best_epoch": result.get("avg_best_epoch", 0),
                "seed": result.get("seed"),
                "result_file": str(pkl_file),
            })
    df = pd.DataFrame(all_rows)
    if df.empty:
        print("No experiments found.")
        return df
    df["encoder_layers_str"] = df["encoder_layers"].apply(lambda x: str(x) if x is not None else "")
    df["decoder_layers_str"] = df["decoder_layers"].apply(lambda x: str(x) if x is not None else "")
    return df


exp_df = load_all_experiments(args.mode)
if exp_df.empty:
    print("No experiments to aggregate; exiting.")
    raise SystemExit(1)

group_cols = [
    "latent_dim", "patience", "encoder_layers_str", "decoder_layers_str",
    "activation", "batch_size", "learning_rate"
]
agg_df = (
    exp_df.groupby(group_cols)
          .agg(
              avg_val_loss_mean=("avg_best_val_loss", "mean"),
              avg_val_loss_std=("avg_best_val_loss", "std"),
              avg_best_epoch_mean=("avg_best_epoch", "mean"),
              n_seeds=("seed", "count"),
              result_files=("result_file", list),
              seeds=("seed", list),
          )
          .reset_index()
)


# --------------------------- Training utilities ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_model(ld: int, row: pd.Series, X: np.ndarray) -> str:
    # parse architecture fields
    try:
        enc_layers = ast.literal_eval(row["encoder_layers_str"]) if row["encoder_layers_str"] else []
    except Exception:
        enc_layers = []
    try:
        dec_layers = ast.literal_eval(row["decoder_layers_str"]) if row["decoder_layers_str"] else []
    except Exception:
        dec_layers = []
    act_cls = activation_from_name(row["activation"])  # returns class, to be instantiated later

    batch_size = int(row["batch_size"]) if not pd.isna(row["batch_size"]) else 64
    learning_rate = float(row["learning_rate"]) if not pd.isna(row["learning_rate"]) else 1e-3
    patience = int(row["patience"]) if not pd.isna(row["patience"]) else None
    best_epoch_mean = float(row["avg_best_epoch_mean"]) if not pd.isna(row["avg_best_epoch_mean"]) else 50.0

    # determine epochs
    max_epochs = max(1, int(np.ceil(best_epoch_mean * float(args.epoch_multiplier))))

    # dataset & split
    X = X.astype(np.float32)
    if X.ndim > 2:
        X = X.reshape(X.shape[0], -1)
    tensor = torch.from_numpy(X)
    idx_all = np.arange(X.shape[0])
    train_idx, test_idx = train_test_split(idx_all, test_size=args.test_size, random_state=seed, shuffle=True)
    ds_train = TensorDataset(tensor[train_idx])
    ds_test = TensorDataset(tensor[test_idx])
    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False)

    # model
    model = AutoEncoder(input_dim=input_dim, latent_dim=ld, enc_layers=enc_layers, dec_layers=dec_layers, act=act_cls).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss(reduction='mean')

    train_losses: List[float] = []
    test_losses_epoch: List[float] = []

    for epoch in range(max_epochs):
        model.train()
        total = 0.0
        n = 0
        for (xb,) in train_loader:
            xb = xb.to(device)
            optimizer.zero_grad(set_to_none=True)
            recon = model(xb)
            loss = criterion(recon, xb)
            loss.backward()
            optimizer.step()
            total += loss.item() * xb.size(0)
            n += xb.size(0)
        train_loss = total / max(n, 1)
        train_losses.append(float(train_loss))

        # periodic test eval
        did_eval = False
        if args.eval_frequency > 0 and ((epoch + 1) % args.eval_frequency == 0 or epoch == max_epochs - 1):
            model.eval()
            tot_t = 0.0
            n_t = 0
            with torch.no_grad():
                for (xb,) in test_loader:
                    xb = xb.to(device)
                    l = criterion(model(xb), xb)
                    tot_t += l.item() * xb.size(0)
                    n_t += xb.size(0)
            test_losses_epoch.append(tot_t / max(n_t, 1))
            did_eval = True

        if epoch % 10 == 0 or epoch == max_epochs - 1:
            if did_eval and len(test_losses_epoch) > 0:
                print(f"[ld={ld}] Epoch {epoch+1}/{max_epochs}: train={train_loss:.6f} test={test_losses_epoch[-1]:.6f}")
            else:
                print(f"[ld={ld}] Epoch {epoch+1}/{max_epochs}: train={train_loss:.6f}")

    final_train_loss = float(train_losses[-1])

    # final test eval
    final_test_loss = None
    if args.eval_frequency >= 0:
        model.eval()
        tot = 0.0
        nte = 0
        with torch.no_grad():
            for (xb,) in test_loader:
                xb = xb.to(device)
                l = criterion(model(xb), xb)
                tot += l.item() * xb.size(0)
                nte += xb.size(0)
        final_test_loss = float(tot / max(nte, 1))
        print(f"[ld={ld}] Final test loss: {final_test_loss:.6f}")
    else:
        print(f"[ld={ld}] Test evaluation skipped (eval_frequency < 0)")

    # plot curve
    save_root = os.path.join("encoder_data", args.mode, "Retrained_AEs")
    os.makedirs(save_root, exist_ok=True)
    epochs_axis = list(range(1, max_epochs + 1))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs_axis, train_losses, label="Train Loss", color="tab:blue")
    if len(test_losses_epoch) > 0:
        if args.eval_frequency > 0:
            eval_epochs = list(range(args.eval_frequency, max_epochs + 1, args.eval_frequency))
            if eval_epochs[-1] != max_epochs:
                eval_epochs.append(max_epochs)
            eval_epochs = sorted(set(eval_epochs))
        else:
            eval_epochs = [max_epochs]
        ax.plot(eval_epochs, test_losses_epoch, label="Test Loss", color="tab:orange", marker="o", linewidth=1)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Reconstruction MSE")
    ax.set_yscale("log")
    ax.set_title(f"Train vs Test Loss (Mode={args.mode}, LD={ld}, eval_freq={args.eval_frequency})")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    curve_path = os.path.join(save_root, f"train_test_loss_curve_ld{ld}.png")
    fig.savefig(curve_path, dpi=200)
    plt.close(fig)
    print(f"Saved train/test loss curve to {curve_path}")

    # save checkpoint
    ckpt = {
        "input_dim": input_dim,
        "data_mode": args.mode,
        "source_file": os.path.abspath(source_file),
        "latent_dim": ld,
        "encoder_layers": enc_layers,
        "decoder_layers": dec_layers,
        "activation_name": canonical_activation_name(row["activation"]),
        "state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "seed": seed,
        "batch_size": batch_size,
        "learning_rate": float(learning_rate),
        "patience": patience,
        "epochs_trained": int(max_epochs),
        "final_train_loss": final_train_loss,
        "test_loss": (final_test_loss if final_test_loss is not None else None),
        "trainval_indices": list(map(int, train_idx.tolist())),
        "test_indices": list(map(int, test_idx.tolist())),
        "best_config_summary": {
            "avg_val_loss_mean": float(row["avg_val_loss_mean"]),
            "avg_best_epoch_mean": float(row["avg_best_epoch_mean"]),
            "n_seeds": int(row["n_seeds"]),
        },
    }
    ckpt_path = os.path.join(save_root, f"retrained_AE_ld{ld}.pth")
    torch.save(ckpt, ckpt_path)
    print(f"Saved retrained model to {ckpt_path}")

    # compatibility copy for LD=20
    if ld == 20:
        compat_path = os.path.join("encoder_data", args.mode, "retrained_best_autoencoder_full.pth")
        torch.save(ckpt, compat_path)
        print(f"Saved LD=20 compatibility copy to {compat_path}")

    return ckpt_path


# --------------------------- Main retraining loop -------------------------
print("\n" + "="*80)
print("RETRAINING BEST CONFIGS")
print("="*80)

saved_ckpts: List[str] = []
latent_dims = [int(x.strip()) for x in args.latent_dims.split(",") if x.strip()]
for ld in latent_dims:
    cand = agg_df[agg_df["latent_dim"] == ld]
    if cand.empty:
        print(f"No experiments found for latent_dim={ld}; skipping.")
        continue
    # choose best row: lowest mean val loss, then lowest std, then more seeds (desc), then shorter epoch mean
    cand = cand.sort_values(by=["avg_val_loss_mean", "avg_val_loss_std", "n_seeds", "avg_best_epoch_mean"],
                            ascending=[True, True, False, True])
    best_row_ld = cand.iloc[0]
    print(f"Selected best config for LD={ld}: act={best_row_ld['activation']}, bs={best_row_ld['batch_size']}, lr={best_row_ld['learning_rate']}, enc={best_row_ld['encoder_layers_str']}, dec={best_row_ld['decoder_layers_str']}, seeds={best_row_ld['n_seeds']}")
    try:
        ckpt_path = train_one_model(ld, best_row_ld, data_matrix)
        saved_ckpts.append(ckpt_path)
    except Exception as e:
        print(f"Error retraining LD={ld}: {e}")

print("\n" + "="*80)
print("RETRAINING SUMMARY")
print("="*80)
if saved_ckpts:
    for p in saved_ckpts:
        print(f"- {p}")
else:
    print("No models retrained. Check that experiments exist for the requested latent dims.")
print("="*80 + "\n")