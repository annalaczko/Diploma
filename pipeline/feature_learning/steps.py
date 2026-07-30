"""Concrete feature learning steps."""

import math

import math

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import dataloader
from pathlib import Path
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms

from pipeline.feature_learning.base import FeatureLearningStep

class VAE( nn.Module):
    def __init__(self, f=4, n=3, s=1, k=128):
        nn.Module.__init__(self)
        self.encoder = Encoder(f, n, s, k)
        self.decoder = Decoder(f, n, s, k)

    def forward(self, x):
        mu, sigma = self.encoder(x)
        std = torch.exp(0.5 * sigma)
        eps = torch.randn_like(std)
        z = mu + eps * std
        z = self.decoder.fc(z)
        z = z.view(-1, 128, 32, 32)
        x_reconstructed = self.decoder(z)
        return x_reconstructed, mu, sigma
    
    def train_model(self, folder, epochs=10, lr=1e-6, batch_size=32):
        dataset = ImageFolderDataset(folder)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = optim.Adam(self.parameters(), lr=lr)

        self.train()
        for epoch in range(epochs):
            train_loss = 0.0
            for i, inputs in enumerate(loader):
                optimizer.zero_grad()
                outputs, mu, sigma = self(inputs)
                beta = self._cyclical_beta(epoch + 1, epochs, M=10)
                loss = self._loss_function(outputs, inputs, mu, sigma, beta=beta)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            print(f'[epoch {epoch+1}] loss: {train_loss / len(loader):.3f}, beta: {beta:.3f}')

    def extract_features(self, folder, batch_size=32):
        dataset = ImageFolderDataset(folder)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

        self.eval()
        all_features = []
        with torch.no_grad():
            for inputs in loader:
                mu, sigma = self.encoder(inputs)
                z = torch.cat([mu, sigma], dim=1)
                all_features.append(z)
        return torch.cat(all_features, dim=0)

    @staticmethod
    def _loss_function(recon_x, x, mu, logvar, beta=1.0):
        BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + beta * KLD

    @staticmethod
    def _cyclical_beta(epoch, T, M, R=0.5):
        """
        epoch: current epoch (1-indexed, matches 'e' in the paper)
        T: total number of epochs
        M: number of cycles
        R: proportion of each cycle spent annealing (fraction that ramps 0->1)
        """
        period = T / M
        tau = ((epoch - 1) % math.ceil(period)) / period
        if tau <= R:
            return tau / R          # linear ramp 0 -> 1
        else:
            return 1.0              # fixed stage, beta = 1


class Encoder(nn.Module):
    def __init__(self, f=4, n=3, s=1, k=128, p=1):
        super(Encoder, self).__init__()

        #used multiple times
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(p=0.25)

        #block0
        self.conv0 = nn.Conv2d(1, 2**f, kernel_size=n, stride=s, padding=p)

        #block1
        self.conv1_0 = nn.Conv2d(2**f, 2**(f+1), kernel_size=n, stride=s+1, padding=p)
        self.conv1_1 = nn.Conv2d(2**(f+1), 2**(f+1), kernel_size=n, stride=s, padding=p)

        #block2
        self.conv2_0 = nn.Conv2d(2**(f+1), 2**(f+2), kernel_size=n, stride=s+1, padding=p)
        self.conv2_1 = nn.Conv2d(2**(f+2), 2**(f+2), kernel_size=n, stride=s, padding=p)

        #block3
        self.conv3_0 = nn.Conv2d(2**(f+2), 2**(f+3), kernel_size=n, stride=s+1, padding=p)
        self.conv3_1 = nn.Conv2d(2**(f+3), 2**(f+3), kernel_size=n, stride=s, padding=p)

        #block4
        flatten_size = 131072

        self.fc_mu = nn.Linear(flatten_size, k)
        self.fc_sigma = nn.Linear(flatten_size, k)

    def forward(self, x):
        #block0
        x = self.conv0(x)
            
        #block1
        x = self.elu(self.conv1_0(x))
        x = self.elu(self.conv1_1(x))
        x = self.dropout(x)

        #block2
        x = self.elu(self.conv2_0(x))
        x = self.elu(self.conv2_1(x))
        x = self.dropout(x)

        #block3
        x = self.elu(self.conv3_0(x))
        x = self.elu(self.conv3_1(x))
        x = self.dropout(x)

        #block4
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        sigma = self.fc_sigma(x)

        return mu, sigma

class Decoder(nn.Module):
    def __init__(self, f=4, n=3, s=1, k=128, p=1):
        super(Decoder, self).__init__()

        #used multiple times
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(p=0.25)

        #block0
        self.fc = nn.Linear(k, 2**(f+3) * 32 * 32)
        self.conv0 = nn.ConvTranspose2d(k, 2**(f+3), kernel_size=n, stride=s, padding=p)

        #block1
        self.conv1_0 = nn.ConvTranspose2d(2**(f+3), 2**(f+3), kernel_size=n, stride=s, padding=p)
        self.conv1_1 = nn.ConvTranspose2d(2**(f+3), 2**(f+2), kernel_size=n, stride=s+1, padding=p, output_padding=1)

        #block2
        self.conv2_0 = nn.ConvTranspose2d(2**(f+2), 2**(f+2), kernel_size=n, stride=s, padding=p)
        self.conv2_1 = nn.ConvTranspose2d(2**(f+2), 2**(f+1), kernel_size=n, stride=s+1, padding=p, output_padding=1)

        #block3
        self.conv3_0 = nn.ConvTranspose2d(2**(f+1), 2**(f+1), kernel_size=n, stride=s, padding=p)
        self.conv3_1 = nn.ConvTranspose2d(2**(f+1), 2**(f), kernel_size=n, stride=s+1, padding=p, output_padding=1)

        #block4
        self.conv4 = nn.ConvTranspose2d(2**f, 1, kernel_size=n, stride=s, padding=p)

    def forward(self, x):
        #block0
        x = self.conv0(x)
            
        #block1
        x = self.elu(self.conv1_0(x))
        x = self.elu(self.conv1_1(x))
        x = self.dropout(x)

        #block2
        x = self.elu(self.conv2_0(x))
        x = self.elu(self.conv2_1(x))
        x = self.dropout(x)

        #block3
        x = self.elu(self.conv3_0(x))
        x = self.elu(self.conv3_1(x))
        x = self.dropout(x)

        #block4
        x = self.conv4(x)

        return torch.sigmoid(x)

def resize_with_padding(img, target_size=256, fill=255):
    img = img.copy()
    img.thumbnail((target_size, target_size))
    delta_w = target_size - img.size[0]
    delta_h = target_size - img.size[1]
    padding = (delta_w // 2, delta_h // 2, delta_w - delta_w // 2, delta_h - delta_h // 2)
    return ImageOps.expand(img, padding, fill=fill)


class ImageFolderDataset(Dataset):
    def __init__(self, folder, target_size=256):
        self.paths = sorted(Path(folder).glob("*.jpg"))
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Lambda(lambda img: resize_with_padding(img, target_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx])
        return self.transform(img)