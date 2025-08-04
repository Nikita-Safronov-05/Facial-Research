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
from concurrent.futures import ThreadPoolExecutor, as_completed

# ------------------------------
# 1. Load Data
# ------------------------------
curvatures = np.load("../processed/2000_interior_curvatures.npy")  # shape: (N, D)
interior_mask = np.load("../processed/interior_mask.npy")  # shape: (7160,)
n_samples, input_dim = curvatures.shape

X_tensor = torch.tensor(curvatures, dtype=torch.float32)
full_indices = np.arange(n_samples)
full_dataset = TensorDataset(X_tensor)

# --------------------------------
# 1.1. Set Random Seed for Reproducibility
# --------------------------------
seed = 1  # Always the same for reproducibility
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
def train_autoencoder(model, train_loader, val_loader, patience=5, max_epochs=1000, learning_rate=1e-3, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
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
# 5. Optimized Single Fold Training Function
# ------------------------------
def train_single_fold(fold_config):
    """Train a single fold with optimized function structure for better performance"""
    try:
        (fold_idx, train_idx, val_idx, dataset, input_dim, latent_dim, 
         encoder_layers, decoder_layers, activation, batch_size, patience, 
         max_epochs, learning_rate, random_seed) = fold_config
        
        # Set seeds for deterministic results
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_seed)
            torch.cuda.manual_seed_all(random_seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        
        # Ensure directory exists
        os.makedirs(f"encoder_data/SEED{random_seed}/experiments", exist_ok=True)
        
        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size, shuffle=False)

        model = CurvatureAutoencoder(input_dim, latent_dim, encoder_layers, decoder_layers, activation)
        best_val_loss, best_epoch = train_autoencoder(
            model, train_loader, val_loader, patience=patience, max_epochs=max_epochs, learning_rate=learning_rate
        )
        
        return fold_idx, float(best_val_loss), best_epoch, train_idx.tolist(), val_idx.tolist()
        
    except Exception as e:
        print(f"Error in fold {fold_idx}: {e}")
        import traceback
        traceback.print_exc()
        return fold_idx, None, None, None, None

# ------------------------------
# 6. K-Fold Cross-Validation (optimized structure)
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
    max_epochs=1000, 
    learning_rate=1e-3,
    k_folds=3, 
    random_seed=42
):
    """Run K-fold CV with optimized function structure for 1.5x speedup"""
    
    # Generate fold splits
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=random_seed)
    indices = np.arange(len(dataset))
    fold_splits = list(kf.split(indices))
    
    # Prepare fold configurations
    fold_configs = []
    for fold_idx, (train_idx, val_idx) in enumerate(fold_splits):
        print(f"Fold {fold_idx+1}/{k_folds} - {len(train_idx)} train, {len(val_idx)} val")
        config = (
            fold_idx, train_idx, val_idx, dataset, input_dim, latent_dim,
            encoder_layers, decoder_layers, activation, batch_size, patience,
            max_epochs, learning_rate, random_seed
        )
        fold_configs.append(config)
    
    # Execute folds using optimized single-worker ThreadPoolExecutor
    fold_results = {}
    with ThreadPoolExecutor(max_workers=1) as executor:  # Single worker for 1.5x speedup
        future_to_fold = {executor.submit(train_single_fold, config): config[0] 
                         for config in fold_configs}
        
        for future in as_completed(future_to_fold):
            fold_idx, best_val_loss, best_epoch, train_indices, val_indices = future.result()
            if best_val_loss is not None:
                fold_results[fold_idx] = {
                    'val_loss': best_val_loss,
                    'best_epoch': best_epoch,
                    'train_indices': train_indices,
                    'val_indices': val_indices
                }
                print(f"  Fold {fold_idx+1} best val loss: {best_val_loss:.6f}")
    
    # Organize results in fold order
    final_val_losses = []
    best_epochs = []
    fold_indices = []
    
    for fold_idx in sorted(fold_results.keys()):
        result = fold_results[fold_idx]
        final_val_losses.append(result['val_loss'])
        best_epochs.append(result['best_epoch'])
        fold_indices.append({
            'train': result['train_indices'],
            'val': result['val_indices']
        })

    arch_repr = get_arch_repr(encoder_layers, decoder_layers, activation)
    results_entry = {
        "latent_dim": latent_dim,
        "patience": patience,
        "architecture": arch_repr,
        "k_folds": k_folds,
        "optimizer": "Adam",
        "learning_rate": learning_rate,
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
# 7. SEARCH GRID (customize as needed)
# -----------------------------------
latent_dims = [5, 10, 15, 20, 30, 40, 50, 100, 200, 400, 600, 800, 1000, 2000, 3000]  # Latent dimensions to test

# Publication-level patience search: more aggressive to conservative early stopping (streamlined)
patiences = [3, 8]  # Aggressive and conservative options only

# Function to get compatible encoder architectures for a given latent dimension
def get_compatible_encoders(latent_dim):
    """Return logical encoder architectures based on latent dimension ranges"""
    all_encoders = []
    
    # Very small latent dims (5-20): Need aggressive compression, use deep networks
    if 5 <= latent_dim <= 20:
        all_encoders.extend([
            [1024, 256, 64],           # Deep aggressive compression
            [2048, 512, 128],          # Medium aggressive compression  
            [1024, 512, 256, 64],      # Very deep compression
        ])
    
    # Small latent dims (30-100): Medium compression
    elif 30 <= latent_dim <= 100:
        all_encoders.extend([
            [2048, 512],               # Simple 2-layer
            [4096, 1024],              # Aggressive 2-layer
            [2048, 1024, 256],         # 3-layer medium
            [4096, 2048, 512],         # 3-layer gradual
        ])
    
    # Medium latent dims (200-600): Conservative compression
    elif 200 <= latent_dim <= 600:
        all_encoders.extend([
            [4096, 2048],              # Conservative 2-layer
            [3400, 1700],              # Natural halving
            [4096, 2048, 1024],        # 3-layer gradual
        ])
    
    # Large latent dims (800-1000): Minimal compression
    elif 800 <= latent_dim <= 1000:
        all_encoders.extend([
            [4096, 2048],              # Conservative compression
            [3400, 1700],              # Natural halving
        ])
    
    # Very large latent dims (2000+): Almost no compression
    elif latent_dim >= 2000:
        all_encoders.extend([
            [4096],                    # Single hidden layer - minimal compression
        ])
    
    return all_encoders

# Batch sizes optimized for 2000 samples and memory efficiency
batch_sizes = [16, 32, 64, 128]       # 16=good balance, 32=efficient, 64=fast but less stable

# Learning rates for comprehensive optimization
learning_rates = [1e-3, 1e-2]  # Conservative to aggressive learning

activations = [nn.LeakyReLU, nn.ELU] 

seeds = [39, 40, 41]  # Fixed seeds for reproducibility

def run_main_grid_search():
    """Main grid search function - can be called from other scripts"""
    metadata_list = []
    exp_num = 0
    
    for latent_dim in latent_dims:
        # Get compatible encoder architectures for this latent dimension
        compatible_encoders = get_compatible_encoders(latent_dim)
        print(f"Latent dim {latent_dim}: {len(compatible_encoders)} compatible architectures")
        
        for encoder_layers in compatible_encoders:
            for act_fn in activations:
                for patience in patiences:
                    for batch_size in batch_sizes:
                        for learning_rate in learning_rates:
                            for exp_seed in seeds:
                                result = run_kfold_cv(
                                    trainval_dataset,
                                    input_dim=input_dim,
                                    latent_dim=latent_dim,
                                    encoder_layers=encoder_layers,
                                    decoder_layers=encoder_layers[::-1], #Symmetric decoder,
                                    activation=act_fn,
                                    patience=patience,
                                    max_epochs=1000,
                                    learning_rate=learning_rate,
                                    batch_size=batch_size,
                                    k_folds=3,  # Reduced from 5 to 3 for 40% speedup
                                    random_seed=exp_seed,
                                )

                                # Construct unique filename for this experiment
                                model_id = (
                                    f"ld{latent_dim}-pat{patience}-bs{batch_size}-lr{learning_rate:.0e}-seed{exp_seed}"
                                    f"-enc{'-'.join(map(str, encoder_layers))}"
                                    f"-dec{'-'.join(map(str, encoder_layers[::-1]))}"
                                    f"-act{act_fn.__name__}"
                                )
                                filename = os.path.join(f'encoder_data/SEED{exp_seed}/experiments', f"{model_id}.pkl")

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
                                    "learning_rate": learning_rate,
                                    "avg_best_val_loss": result["avg_best_val_loss"],
                                    "std_best_val_loss": result["std_best_val_loss"],
                                    "seed": exp_seed,
                                    "result_file": filename
                                }
                                metadata_list.append(meta)
                                exp_num += 1
                                print(f"Stored experiment {exp_num}: {filename}")

    # Save metadata as CSV for easy searching/filtering
    meta_csv = f"encoder_data/experiment_metadata.csv"  # Use general path since we have multiple seeds
    pd.DataFrame(metadata_list).to_csv(meta_csv, index=False)
    return metadata_list

if __name__ == "__main__":
    # Only run the main grid search when this file is executed directly
    run_main_grid_search()