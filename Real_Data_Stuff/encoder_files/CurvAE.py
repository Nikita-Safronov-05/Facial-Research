from __future__ import annotations
from pathlib import Path
from typing import Union
import numpy as np
import torch
import torch.nn as nn


class CurvAE(nn.Module):
    """Simple fully-connected autoencoder for curvature vectors.

    Encoder/decoder hidden layer sizes are configurable, with a linear bottleneck.
    Activation is applied after each hidden Linear.
    """

    def __init__(self, input_dim: int, latent_dim: int,
                 encoder_layers: list[int], decoder_layers: list[int],
                 activation: type[nn.Module] = nn.ELU):
        super().__init__()
        # Encoder
        enc = []
        d = input_dim
        for h in encoder_layers:
            enc.append(nn.Linear(d, h))
            enc.append(activation())
            d = h
        enc.append(nn.Linear(d, latent_dim))
        self.encoder = nn.Sequential(*enc)

        # Decoder
        dec = []
        d = latent_dim
        for h in decoder_layers:
            dec.append(nn.Linear(d, h))
            dec.append(activation())
            d = h
        dec.append(nn.Linear(d, input_dim))
        self.decoder = nn.Sequential(*dec)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)


def build_model(ckpt_path: Union[str, Path], device: Union[str, torch.device] = "cpu") -> CurvAE:
    """Load a CurvAE checkpoint with robust key remapping support."""
    device = torch.device(device)
    ckpt_path = Path(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device)

    act_name = ckpt.get("activation_name", "ELU")
    activation = getattr(nn, act_name, nn.ELU)

    model = CurvAE(
        input_dim=int(ckpt["input_dim"]),
        latent_dim=int(ckpt["latent_dim"]),
        encoder_layers=list(ckpt["encoder_layers"]),
        decoder_layers=list(ckpt["decoder_layers"]),
        activation=activation,
    )

    state = ckpt.get("state_dict", ckpt)

    # direct load
    try:
        model.load_state_dict(state)
    except Exception:
        # remap old/new style keys + DataParallel prefixes
        remapped = {}
        for k, v in state.items():
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
            missing, unexpected = model.load_state_dict(remapped, strict=False)
            if missing:
                print(f"[warn] Missing keys: {sorted(missing)[:6]}{'...' if len(missing)>6 else ''}")
            if unexpected:
                print(f"[warn] Unexpected keys: {sorted(unexpected)[:6]}{'...' if len(unexpected)>6 else ''}")

    model = model.to(device)
    model.eval()
    return model


def encode_latent(model: CurvAE, Y: np.ndarray, batch_size: int = 256,
                  device: Union[str, torch.device] = "cpu") -> np.ndarray:
    device = torch.device(device)
    model = model.to(device)
    outs = []
    with torch.no_grad():
        for i in range(0, Y.shape[0], batch_size):
            xb = torch.from_numpy(Y[i:i+batch_size]).float().to(device)
            zb = model.encoder(xb)
            outs.append(zb.cpu().numpy())
    return np.vstack(outs)
