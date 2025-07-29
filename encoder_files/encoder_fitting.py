import pandas as pd
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split, KFold
import torch.nn as nn
import torch.optim as optim
import random
import os

# --------------- 1. Load metadata, select best model ---------------
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
curvatures = np.load("../processed/interior_curvatures_good.npy")
metadata = 'encoder_data/experiment_metadata.csv'
df = pd.read_csv(metadata)

# Columns to use as the unique config key
group_cols = ['latent_dim', 'patience', 'encoder_layers', 'decoder_layers', 'activation', 'batch_size']

# Aggregation: mean and std for validation loss, mean for best epoch (add more columns as needed)
agg_df = df.groupby(group_cols).agg(
    avg_val_loss_mean = ('avg_best_val_loss', 'mean'),
    avg_val_loss_std = ('avg_best_val_loss', 'std'),
    n_seeds = ('seed', 'count'),
    result_files = ('result_file', list),
    seeds = ('seed', list)
).reset_index()
print(agg_df.shape, agg_df.head())

best_row = agg_df.sort_values('avg_val_loss_mean').iloc[0]
model_file = best_row['result_files'][0]

# --------------- 2. Load best experiment's config and avg_best_epoch ---------------
with open(model_file, 'rb') as f:
    result = pickle.load(f)

latent_dim = result["latent_dim"]
encoder_layers = result["architecture"]["encoder_layers"]
decoder_layers = result["architecture"]["decoder_layers"]
activation_name = result["architecture"]["activation"]
activation = getattr(nn, activation_name)
batch_size = result["batch_size"]

n_samples, input_dim = curvatures.shape

avg_best_epoch = int(round(result['avg_best_epoch']))  # Safeguard default
print(f"Will train {avg_best_epoch} epochs.")

# --------------- 3. Prepare data (use full trainval split) ---------------
X_tensor = torch.tensor(curvatures, dtype=torch.float32)
full_dataset = torch.utils.data.TensorDataset(X_tensor)

# --------- 4. Load previously saved trainval_indices ----------
TEST_SIZE = 0.1  # 10% for test, adjust if needed
full_indices = np.arange(n_samples)
all_indices = np.arange(n_samples)
trainval_indices, test_indices = train_test_split(
    full_indices, test_size=TEST_SIZE, random_state=seed, shuffle=True
)

trainval_dataset = Subset(full_dataset, trainval_indices)
test_dataset = Subset(full_dataset, test_indices)
trainval_loader = DataLoader(trainval_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# --------------- 5. Build the model ---------------
from curvature_autoencoder import CurvatureAutoencoder

model = CurvatureAutoencoder(input_dim, latent_dim, encoder_layers, decoder_layers, activation)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# --------------- 5. Train on whole trainval set ---------------
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()
train_losses = []

for epoch in range(avg_best_epoch):
    model.train()
    running_loss = 0.0
    for batch in trainval_loader:
        x = batch[0].to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), x)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * x.size(0)
    epoch_loss = running_loss / len(full_dataset)
    train_losses.append(epoch_loss)
    print(f"Epoch {epoch+1}/{avg_best_epoch}: train loss = {epoch_loss:.6f}")

print("Retraining complete.")

# ---- 4. Evaluate on the test set ----
model.eval()
test_loss = 0.0
n_test = 0
with torch.no_grad():
    for batch in test_loader:
        x = batch[0].to(device)
        loss = criterion(model(x), x)
        test_loss += loss.item() * x.size(0)
        n_test += x.size(0)
test_loss /= n_test
print(f"\nGeneralization (test) reconstruction loss: {test_loss:.6f}\n")

# --------------- 6. Save retrained model ---------------
torch.save({
    'input_dim': input_dim,
    'latent_dim': latent_dim,
    'encoder_layers': encoder_layers,
    'decoder_layers': decoder_layers,
    'activation_name': activation.__name__,
    'state_dict': model.state_dict()
}, 'encoder_data/retrained_best_autoencoder_full.pth')