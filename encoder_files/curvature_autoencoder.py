from pathlib import Path
from typing import Union
import numpy as np
import torch
import torch.nn as nn

class CurvatureAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, encoder_layers, decoder_layers, activation):
        """
        input_dim: int, dimension of input features
        latent_dim: int, size of latent (bottleneck) layer
        encoder_layers: list of int, sizes of encoder hidden layers
        decoder_layers: list of int, sizes of decoder hidden layers
        activation: torch.nn.Module class, e.g. nn.ReLU, nn.ELU (pass the class, not an instance)
        """
        super().__init__()
        # Encoder
        encoder = []
        last_dim = input_dim
        for h in encoder_layers:
            encoder.append(nn.Linear(last_dim, h))
            encoder.append(activation())
            last_dim = h
        encoder.append(nn.Linear(last_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder)

        # Decoder
        decoder = []
        last_dim = latent_dim
        for h in decoder_layers:
            decoder.append(nn.Linear(last_dim, h))
            decoder.append(activation())
            last_dim = h
        decoder.append(nn.Linear(last_dim, input_dim))
        self.decoder = nn.Sequential(*decoder)

    def forward(self, x):
        """
        x: input tensor of shape (N, input_dim)
        Returns: reconstruction of x
        """
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon


def build_model(ckpt_path: Union[str, Path], device: torch.device) -> CurvatureAutoencoder:
    """Load a CurvatureAutoencoder from a checkpoint."""
    ckpt_path = Path(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device)
    activation = getattr(nn, ckpt["activation_name"]) if isinstance(ckpt.get("activation_name"), str) else ckpt.get("activation_name", nn.ReLU)
    model = CurvatureAutoencoder(
        input_dim=ckpt["input_dim"],
        latent_dim=ckpt["latent_dim"],
        encoder_layers=ckpt["encoder_layers"],
        decoder_layers=ckpt["decoder_layers"],
        activation=activation,
    )
    state_dict = ckpt.get("state_dict", ckpt)

    # First attempt: direct load
    try:
        model.load_state_dict(state_dict)
    except Exception:
        # Remap keys to handle older/newer naming (e.g., 'encoder.net.0.*' -> 'encoder.0.*', 'decoder.net.*' -> 'decoder.*')
        remapped = {}
        for k, v in state_dict.items():
            nk = k
            if nk.startswith("module."):
                nk = nk[len("module."):]
            if nk.startswith("encoder.net."):
                nk = nk.replace("encoder.net.", "encoder.", 1)
            if nk.startswith("decoder.net."):
                nk = nk.replace("decoder.net.", "decoder.", 1)
            remapped[nk] = v
        try:
            model.load_state_dict(remapped)
        except Exception:
            # Last resort: non-strict load to allow minor mismatches
            missing, unexpected = model.load_state_dict(remapped, strict=False)
            # Optional: print a small notice to help debugging
            if missing:
                print(f"[warn] Missing keys when loading {ckpt_path.name}: {sorted(missing)[:6]}{'...' if len(missing)>6 else ''}")
            if unexpected:
                print(f"[warn] Unexpected keys when loading {ckpt_path.name}: {sorted(unexpected)[:6]}{'...' if len(unexpected)>6 else ''}")
    model = model.to(device)
    model.eval()
    return model


def encode_latent(
    model: CurvatureAutoencoder,
    Y: np.ndarray,
    batch_size: int = 256,
    device: Union[str, torch.device] = "cpu",
) -> np.ndarray:
    """Encode curvature vectors Y to latent codes using the model's encoder."""
    device = torch.device(device)
    model = model.to(device)
    zs = []
    with torch.no_grad():
        for i in range(0, Y.shape[0], batch_size):
            yb = torch.from_numpy(Y[i : i + batch_size]).float().to(device)
            zb = model.encoder(yb)
            zs.append(zb.cpu().numpy())
    return np.vstack(zs)