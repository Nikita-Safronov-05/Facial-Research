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