import torch
import torch.nn as nn
from curvature_autoencoder import CurvatureAutoencoder
import trimesh
import numpy as np
import matplotlib.pyplot as plt

ACTIVATION_MAP = {
    'ReLU': nn.ReLU,
    'LeakyReLU': nn.LeakyReLU,
    'ELU': nn.ELU,
    'SiLU': nn.SiLU,
}

def load_autoencoder(filepath):
    """
    Loads full autoencoder config and weights from a .pth file saved as a dict.
    Returns the model, and its hyperparameters as a dict.
    """
    checkpoint = torch.load(filepath)
    
    input_dim = checkpoint['input_dim']
    latent_dim = checkpoint['latent_dim']
    encoder_layers = checkpoint['encoder_layers']
    decoder_layers = checkpoint['decoder_layers']
    activation_name = checkpoint['activation_name']
    activation = ACTIVATION_MAP[activation_name]
    state_dict = checkpoint['state_dict']

    model = CurvatureAutoencoder(
        input_dim=input_dim,
        latent_dim=latent_dim,
        encoder_layers=encoder_layers,
        decoder_layers=decoder_layers,
        activation=activation
    )
    model.load_state_dict(state_dict)
    model.eval()

    # Optionally return config for documentation
    config = {
        'input_dim': input_dim,
        'latent_dim': latent_dim,
        'encoder_layers': encoder_layers,
        'decoder_layers': decoder_layers,
        'activation': activation,
        'activation_name': activation_name
    }

    return model, config

def visualize_curvature_on_face(idx, curvature_vector, title="Curvature"):
    filename = f"../5pc_simfaces/simface_5pc_{idx+1}.obj"
    mesh = trimesh.load(filename)

    # Color vertices
    face_colors = np.zeros((len(mesh.vertices), 3))
    face_colors[interior_mask] = plt.cm.viridis(curvature_vector)[:, :3]  # RGB
    mesh.visual.vertex_colors = (face_colors * 255).astype(np.uint8)

    print(f"Showing: {title}")
    mesh.show()

def render_curvature_to_image(idx, curvature_vector):
    filename = f"../5pc_simfaces/simface_5pc_{idx+1}.obj"
    mesh = trimesh.load(filename)

    face_colors = np.zeros((len(mesh.vertices), 3))
    face_colors[interior_mask] = plt.cm.viridis(curvature_vector)[:, :3]
    mesh.visual.vertex_colors = (face_colors * 255).astype(np.uint8)

    scene = mesh.scene()
    png = scene.save_image(resolution=(512, 512), visible=True)
    return plt.imread(trimesh.util.wrap_as_stream(png))

def render_side_by_side(original_img, reconstructed_img):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(original_img)
    plt.axis("off")
    plt.title("Original Curvature")

    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed_img)
    plt.axis("off")
    plt.title("Reconstructed Curvature")

    plt.tight_layout()
    plt.show()

def visualize_difference_on_face(idx, original, reconstructed, title="Curvature Difference"):
    filename = f"../5pc_simfaces/simface_5pc_{idx+1}.obj"
    mesh = trimesh.load(filename)

    # Compute absolute difference for interior vertices
    diff = np.abs(original - reconstructed)

    # Normalize for colormap
    diff_norm = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8)

    # Apply color to interior vertices only
    face_colors = np.zeros((len(mesh.vertices), 3))
    face_colors[interior_mask] = plt.cm.plasma(diff_norm)[:, :3]
    mesh.visual.vertex_colors = (face_colors * 255).astype(np.uint8)

    print(f"Showing: {title}")
    mesh.show()

# Load the best model
model_path = 'encoder_data/retrained_best_autoencoder_full.pth'
model, config = load_autoencoder(model_path)
print(f"Loaded model config: {config}")

# Load the curvature data
curvatures = np.load("../processed/interior_curvatures_good.npy")  # shape: (N, D)
interior_mask = np.load("../processed/interior_mask.npy")  # shape: (7160,)
n_samples, input_dim = curvatures.shape

X_tensor = torch.tensor(curvatures, dtype=torch.float32)
full_indices = np.arange(n_samples)
print(torch.cuda.is_available(), torch.cuda.device_count())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

model.eval()
with torch.no_grad():
    latent_codes = model.encoder(X_tensor.to(device)).cpu().numpy()
    X_reconstructed = model(X_tensor.to(device)).cpu().numpy()

for example_idx in range(3):
    original_img = render_curvature_to_image(example_idx, curvatures[example_idx])
    reconstructed_img = render_curvature_to_image(example_idx, X_reconstructed[example_idx])
    render_side_by_side(original_img, reconstructed_img)
    visualize_difference_on_face(example_idx, curvatures[example_idx], X_reconstructed[example_idx])