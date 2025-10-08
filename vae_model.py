import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_shape, latent_dim):
        super(VAE, self).__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        c, d, h, w = input_shape

        # --- Encoder blocks ---
        def down_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True)
            )

        self.encoder = nn.Sequential(
            down_block(c, 16),
            down_block(16, 32),
            down_block(32, 64),
            down_block(64, 128)
        )

        enc_output_shape = (128, d // 16, h // 16, w // 16)
        self.enc_output_dim = np.prod(enc_output_shape)

        self.fc_mu = nn.Linear(self.enc_output_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.enc_output_dim, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, self.enc_output_dim)

        # --- Decoder blocks (Upsample + Conv3D) ---
        def up_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
                nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True)
            )

        self.decoder = nn.Sequential(
            up_block(128, 64),
            up_block(64, 32),
            up_block(32, 16),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            nn.Conv3d(16, c, kernel_size=3, padding=1)  # No activation here
        )

    def encode(self, x):
        x = self.encoder(x)
        x = x.reshape(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.fc_decode(z)
        x = x.view(z.size(0), 128, self.input_shape[1] // 16, self.input_shape[2] // 16, self.input_shape[3] // 16)
        x = self.decoder(x)
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar
