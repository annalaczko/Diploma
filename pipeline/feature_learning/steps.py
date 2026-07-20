"""Concrete feature learning steps."""

from typing import Any
import numpy as np
import torch
import torch.nn as nn

from pipeline.feature_learning.base import FeatureLearningStep


class PretrainedFeatureLearning(FeatureLearningStep):
    def __init__(self, backbone: str = "resnet50", normalize: bool = True):
        super().__init__()
        self.backbone = backbone
        self.normalize = normalize

    def run(self, data: Any) -> np.ndarray:
        # TODO: extract features from pretrained backbone
        features = np.zeros((len(data), 512))
        return self.l2_normalize(features) if self.normalize else features


class VAEFeatureLearning(FeatureLearningStep):
    def __init__(self, latent_dim: int = 128, epochs: int = 100, normalize: bool = True):
        super().__init__()
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.normalize = normalize

    def run(self, data: Any) -> np.ndarray:
        # TODO: train VAE, return latent vectors
        features = np.zeros((len(data), self.latent_dim))
        return self.l2_normalize(features) if self.normalize else features


class PotteryVAE(nn.Module):
    def __init__(self, latent_dim=128, dropout_p=0.25):
        super().__init__()
        # Encoder: 256x256 -> csökkenő spatial dim, növekvő csatornaszám
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # 128x128
            nn.ELU(), nn.Dropout2d(dropout_p),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 64x64
            nn.ELU(), nn.Dropout2d(dropout_p),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 32x32
            nn.ELU(), nn.Dropout2d(dropout_p),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # 16x16
            nn.ELU(), nn.Dropout2d(dropout_p),
        )
        self.flatten_dim = 256 * 16 * 16
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)

        self.decoder_input = nn.Linear(latent_dim, self.flatten_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ELU(), nn.Dropout2d(dropout_p),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ELU(), nn.Dropout2d(dropout_p),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ELU(), nn.Dropout2d(dropout_p),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
        )

    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder_input(z)
        h = h.view(h.size(0), 256, 16, 16)
        return self.decoder(h)  # logits, sigmoid loss-ban alkalmazva

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar, z
