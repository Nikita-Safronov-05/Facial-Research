import os
import pickle
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, TensorDataset
from sklearn.model_selection import train_test_split, KFold
import matplotlib.pyplot as plt

# ------------------------------
# 1. Load Data
# ------------------------------
curvatures = np.load("../processed/interior_curvatures_good.npy")  # shape: (N, D)
interior_mask = np.load("../processed/interior_mask.npy")  # shape: (7160,)
n_samples, input_dim = curvatures.shape

X_tensor = torch.tensor(curvatures, dtype=torch.float32)
full_indices = np.arange(n_samples)
full_dataset = TensorDataset(X_tensor)

# --------------------------------
# 1.1. Set Random Seed for Reproducibility
# --------------------------------
seed = 41  # Always the same for reproducibility
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
os.makedirs(f"encoder_data/SEED{seed}/experiments", exist_ok=True)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

# ------------------------------
# 2. Fixed Test Split (for all experiments)
# ------------------------------
TEST_SIZE = 0.1  # 10% for test, adjust if needed

trainval_indices, test_indices = train_test_split(
    full_indices, test_size=TEST_SIZE, random_state=seed, shuffle=True
)
trainval_dataset = Subset(full_dataset, trainval_indices)
test_dataset = Subset(full_dataset, test_indices)

print(f"Total: {n_samples}, Train+Val: {len(trainval_dataset)}, Test: {len(test_dataset)}")

with open(f"encoder_data/SEED{seed}/test_indices.pkl", "wb") as f:
    pickle.dump(test_indices, f)

# ------------------------------
# 3. Model Constructor with Configurable Architecture
# ------------------------------
from curvature_autoencoder import CurvatureAutoencoder

def get_arch_repr(encoder_layers, decoder_layers, activation):
    return {
        "encoder_layers": encoder_layers,
        "decoder_layers": decoder_layers,
        "activation": activation.__name__ if hasattr(activation, "__name__") else str(activation)
    }

# ------------------------------
# 4. Train Function for 1 Split
# ------------------------------
def train_autoencoder(model, train_loader, val_loader, patience=5, max_epochs=100, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    patience_counter = 0
    best_epoch = 0

    for epoch in range(max_epochs):
        model.train()
        epoch_train_loss = 0
        for batch in train_loader:
            x_batch = batch[0].to(device)
            optimizer.zero_grad()
            loss = criterion(model(x_batch), x_batch)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item() * x_batch.size(0)
        epoch_train_loss /= len(train_loader.dataset)

        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                x_val = batch[0].to(device)
                loss = criterion(model(x_val), x_val)
                epoch_val_loss += loss.item() * x_val.size(0)
        epoch_val_loss /= len(val_loader.dataset)

        scheduler.step(epoch_val_loss)

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            best_epoch = epoch + 1
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    return best_val_loss, best_epoch

# ------------------------------
# 5. K-Fold Cross-Validation (on trainval, test set untouched)
# ------------------------------
def run_kfold_cv(
    dataset, 
    input_dim,
    latent_dim, 
    encoder_layers,
    decoder_layers,
    activation,
    batch_size=64,
    patience=5, 
    max_epochs=100, 
    k_folds=5, 
    random_seed=42
):
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=random_seed)
    indices = np.arange(len(dataset))
    fold_indices = []
    final_val_losses = []
    best_epochs = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(indices)):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        os.makedirs(f"encoder_data/SEED{seed}/experiments", exist_ok=True)
        print(f"Fold {fold+1}/{k_folds} - {len(train_idx)} train, {len(val_idx)} val")
        fold_indices.append({'train': train_idx.tolist(), 'val': val_idx.tolist()})
        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size, shuffle=False)

        model = CurvatureAutoencoder(input_dim, latent_dim, encoder_layers, decoder_layers, activation)
        best_val_loss, best_epoch = train_autoencoder(
            model, train_loader, val_loader, patience=patience, max_epochs=max_epochs
        )

        final_val_losses.append(float(best_val_loss))
        best_epochs.append(best_epoch)
        print(f"  Fold {fold+1} best val loss: {best_val_loss:.6f}")

    arch_repr = get_arch_repr(encoder_layers, decoder_layers, activation)
    results_entry = {
        "latent_dim": latent_dim,
        "patience": patience,
        "architecture": arch_repr,
        "k_folds": k_folds,
        "optimizer": "Adam",
        "learning_rate": 0.001,
        "seed": random_seed,
        "batch_size": batch_size,
        "fold_indices": fold_indices,
        "best_val_losses_per_fold": final_val_losses,
        "avg_best_val_loss": float(np.mean(final_val_losses)),
        "std_best_val_loss": float(np.std(final_val_losses)),
        "best_epochs_per_fold": best_epochs,
        "avg_best_epoch": float(np.mean(best_epochs))
    }

    return results_entry

# -----------------------------------
# 6. SEARCH GRID (customize as needed)
# -----------------------------------
metadata_list = []
latent_dims = [50, 200, 400, 600, 800, 1000, 2000, 3000]  # Latent dimensions to test
patiences = [5, 8, 10]
encoders = [[1024, 256], [2048, 256, 128]]
# decoders = [[256, 1024], [128, 256, 2048]]
activations = [nn.ReLU, nn.LeakyReLU, nn.ELU, nn.SiLU]
batch_sizes = [16, 32] 

exp_num = 0
for encoder_layers in encoders:
    for act_fn in activations:
        for latent_dim in latent_dims:
            for patience in patiences:
                for batch_size in batch_sizes:
                    for seed in [39, 40, 41, 42]:
                        result = run_kfold_cv(
                            trainval_dataset,
                            input_dim=input_dim,
                            latent_dim=latent_dim,
                            encoder_layers=encoder_layers,
                            decoder_layers=encoder_layers[::-1], #Symmetric decoder,
                            activation=act_fn,
                            patience=patience,
                            max_epochs=100,
                            batch_size=batch_size,
                            k_folds=5,
                            random_seed=seed,
                        )

                        # Construct unique filename for this experiment
                        model_id = (
                            f"ld{latent_dim}-pat{patience}-bs{batch_size}-seed{seed}"
                            f"-enc{'-'.join(map(str, encoder_layers))}"
                            f"-dec{'-'.join(map(str, encoder_layers[::-1]))}"
                            f"-act{act_fn.__name__}"
                        )
                        filename = os.path.join(f'encoder_data/SEED{seed}/experiments', f"{model_id}.pkl")

                        # Write the per-experiment result
                        with open(filename, "wb") as f:
                            pickle.dump(result, f)
                            
                        # Add metadata entry (flat, for easy CSV/JSON export)
                        meta = {
                            "model_id": model_id,
                            "latent_dim": latent_dim,
                            "patience": patience,
                            "encoder_layers": encoder_layers,
                            "decoder_layers": encoder_layers[::-1],  # Symmetric decoder
                            "activation": act_fn.__name__,
                            "batch_size": batch_size,
                            "avg_best_val_loss": result["avg_best_val_loss"],
                            "std_best_val_loss": result["std_best_val_loss"],
                            "seed": seed,
                            "result_file": filename
                        }
                        metadata_list.append(meta)
                        exp_num += 1
                        print(f"Stored experiment {exp_num}: {filename}")

# Save metadata as CSV for easy searching/filtering
meta_csv = f"encoder_data/SEED{seed}/experiment_metadata.csv"
pd.DataFrame(metadata_list).to_csv(meta_csv, index=False)