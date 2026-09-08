"""Concrete feature learning steps built around a convolutional VAE.

    VAE - unsupervised: trains an Encoder/Decoder pair to reconstruct
    pottery-sherd images, and can extract the encoder's latent means as a
    feature vector per image.

The supervised counterpart, `VAEClassifier` (which wraps a trained `VAE`'s
encoder with a classifier head), lives in
`pipeline.feature_learning.vae_classifier` - a separate file since it's a
distinct model/training loop that only depends on `VAE` through its
`encoder` attribute.

Encoder/Decoder are the underlying conv-net architecture; image loading
(path discovery + tensor conversion) is delegated to
`pipeline.preprocessing.dataset`, shared with the preprocessing steps.
"""

import math
from typing import Literal
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from PIL import ImageOps

from pipeline.feature_learning.featurelearning_step import FeatureLearningStep
from pipeline.preprocessing.dataset import ImageDataset, collect_image_paths


class VAE(nn.Module, FeatureLearningStep):
    """Convolutional Variational Autoencoder used as an unsupervised
    feature learner: trained to reconstruct grayscale pottery-sherd
    images, then its encoder's latent means (`mu`) are used as a
    fixed-length feature vector per image for downstream clustering.
    """

    def __init__(
        self,
        input_path: str,
        f=4,
        n=3,
        s=1,
        k=128,
        output_path=None,
        name=None,
        delete_input_data=False,
        save_results=False,
        run_id="default_run",
    ):
        """
        Args:
            input_path: folder of images to train on / extract features
                from (searched recursively for .jpg/.jpeg/.png).
            f: base channel-width exponent for the Encoder/Decoder; block i
                uses 2**(f+i) channels, so larger f means a wider network.
            n: convolution kernel size, used throughout the Encoder/Decoder.
            s: base stride; each block's first conv downsamples (Encoder)
                or upsamples (Decoder) with stride s+1, its second conv
                uses stride s (see Encoder/Decoder for the exact layout).
            k: latent dimensionality (the feature vector size `mu`/`sigma`
                end up with, and what downstream clustering operates on).
            output_path: where extracted features / checkpoints are saved.
                Defaults to the Step-computed run-scoped directory.
        """
        nn.Module.__init__(self)
        FeatureLearningStep.__init__(
            self,
            input_path=input_path or "",
            output_path=output_path,
            name=name,
            delete_input_data=delete_input_data,
            save_results=save_results,
            run_id=run_id,
        )
        self.encoder = Encoder(f, n, s, k)
        self.decoder = Decoder(f, n, s, k)

    def run(
        self,
        mode: Literal["extract_features", "train_model"] = "extract_features",
        *,
        epochs=10,
        lr=1e-6,
        batch_size=32
    ) -> pd.DataFrame:
        """Either train the VAE from scratch, or just embed `input_path`'s
        images with the current (e.g. previously loaded) weights.

        Args:
            mode: 'train_model' trains for `epochs` epochs and then
                extracts features from the trained model; 'extract_features'
                (default) skips training and just runs inference - use
                this once you already have trained weights loaded.
            epochs: training epochs, only used when mode='train_model'.
            lr: Adam learning rate, only used when mode='train_model'.
            batch_size: batch size for both training and feature extraction.

        Returns:
            A DataFrame with one 'filename' and 'features' (latent vector)
            row per image, as produced by `extract_features`.
        """
        if mode not in {"extract_features", "train_model"}:
            raise ValueError(
                "mode must be either 'extract_features' or 'train_model'"
            )

        folder = self.input_path
        if not folder:
            raise ValueError("input_path is required to run the VAE feature step")

        if mode == "train_model":
            return self.train_model(
                folder,
                epochs=epochs,
                lr=lr,
                batch_size=batch_size,
            )

        return self.extract_features(folder, batch_size=batch_size)

    def forward(self, x):
        """Encode `x`, sample a latent `z` via the reparameterization 
        trick (so gradients can flow through the sampling step), and
        decode it back into a reconstructed image.

        Returns:
            (x_reconstructed, mu, sigma): the reconstruction plus the
            encoder's mean and log-variance, both needed by
            `_loss_function` to compute the KL-divergence term.
        """
        mu, sigma = self.encoder(x)
        std = torch.exp(0.5 * sigma)
        eps = torch.randn_like(std)
        z = mu + eps * std
        z = self.decoder.fc(z)
        z = z.view(-1, 128, 32, 32)
        x_reconstructed = self.decoder(z)
        return x_reconstructed, mu, sigma

    def train_model(self, folder, epochs=10, lr=1e-6, batch_size=32):
        """Train the VAE (encoder + decoder) to reconstruct images from
        `folder`, using a cyclically annealed KL weight (`beta`, see
        `_cyclical_beta`) to avoid posterior collapse, then extract and
        return features from the trained model.

        Args:
            folder: image folder to train on (passed to `ImageDataset`).
            epochs: number of training epochs.
            lr: Adam learning rate.
            batch_size: training batch size.

        Returns:
            The DataFrame from `extract_features` (filenames + latent
            vectors), run once training is complete.
        """
        device = next(self.parameters()).device
        # target_size=None: this pipeline's Normalizer step already makes every
        # image 256x256, so resizing here would just be repeated wasted work
        # (every image, every epoch); a wrong-sized image will fail loudly when
        # DataLoader tries to batch it, rather than being silently resized.
        dataset = ImageDataset(collect_image_paths(folder), grayscale=True, target_size=None)
        print(f"Dataset size: {dataset.__len__()}")
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True,
            num_workers=16, pin_memory=True, persistent_workers=True
        )
        optimizer = optim.Adam(self.parameters(), lr=lr)

        self.train()
        for epoch in range(epochs):
            train_loss = 0.0
            beta = self._cyclical_beta(epoch+1, epochs, 4)
            for i, inputs in enumerate(loader):
                inputs = inputs.to(device, non_blocking=True)
                optimizer.zero_grad()
                outputs, mu, sigma = self(inputs)
                loss = self._loss_function(outputs, inputs, mu, sigma, beta)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            print(f'[epoch {epoch+1}] loss: {train_loss / len(loader):.3f} | beta: {beta:.3f}')

        return self.extract_features(batch_size=batch_size)

    def extract_features(self, batch_size=32) -> pd.DataFrame:
        """Run the encoder over every image in `self.input_path` in eval
        mode and collect both the latent mean (`mu`) and log-variance
        (`sigma`) as that image's feature vectors - not a sampled `z`, so
        features are deterministic and reproducible across calls.

        Args:
            batch_size: inference batch size.

        Returns:
            A DataFrame with one 'filename', 'features' (`mu` as a numpy
            array) and 'sigma' (log-variance as a numpy array) row per
            image; also written to `<output_path>/features.csv`.
            `sigma` reflects the encoder's uncertainty per latent
            dimension for that image - e.g. a high value on a dimension
            means the encoder isn't confident about that image's value
            there, which can be useful to inspect even though downstream
            steps here (clustering, the classifier) only use `mu`.
        """
        device = next(self.parameters()).device
        # See train_model's ImageDataset call for why target_size=None here.
        dataset = ImageDataset(collect_image_paths(self.input_path), grayscale=True, target_size=None)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False,
            num_workers=16, pin_memory=True, persistent_workers=True
        )

        filenames = [Path(p).name for p in dataset.paths]
        self.eval()
        all_mu = []
        all_sigma = []
        with torch.no_grad():
            for inputs in loader:
                inputs = inputs.to(device, non_blocking=True)
                mu, sigma = self.encoder(inputs)
                all_mu.append(mu.cpu())
                all_sigma.append(sigma.cpu())

        mu_features = torch.cat(all_mu, dim=0).numpy()
        sigma_features = torch.cat(all_sigma, dim=0).numpy()

        result = pd.DataFrame({
            "filename": filenames,
            "features": list(mu_features),
            "sigma": list(sigma_features),
        })
        output_file = Path(self.output_path) / "features.csv"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_file, index=False)
        return result

    @staticmethod
    def _loss_function(recon_x, x, mu, logvar, beta=1.0):
        """Standard beta-VAE loss: pixel-wise reconstruction error (BCE,
        since pixels are in [0, 1]) plus a KL-divergence term pulling the
        latent distribution toward a standard normal, weighted by `beta`.
        A higher `beta` trades off reconstruction quality for a latent
        space that's more disentangled/regularized (better for downstream
        clustering, worse for pixel-perfect reconstructions).
        """
        BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + beta * KLD

    @staticmethod
    def _cyclical_beta(epoch, T, M, R=0.5):
        """Cyclical KL-annealing schedule (Fu et al., "Cyclical Annealing
        Schedule", 2019): ramps beta linearly from 0 to 1 over the first
        `R` fraction of each cycle, then holds it at 1 for the rest of the
        cycle, repeating for `M` cycles across training. This avoids
        posterior collapse (the decoder ignoring the latent code, which a
        constant high beta from epoch 1 tends to cause) while still
        reaching full KL regularization within each cycle.

        Args:
            epoch: current epoch (1-indexed, matches 'e' in the paper).
            T: total number of epochs.
            M: number of cycles.
            R: proportion of each cycle spent annealing (fraction that ramps 0->1).

        Returns:
            The beta weight to use for this epoch's loss, in [0, 1].
        """
        period = T / M
        tau = ((epoch - 1) % math.ceil(period)) / period
        if tau <= R:
            return tau / R          # linear ramp 0 -> 1
        else:
            return 1.0              # fixed stage, beta = 1


class Encoder(nn.Module):
    """Downsamples a 256x256 grayscale image to a `k`-dimensional latent
    distribution (mu, sigma), via 4 blocks that each roughly halve the
    spatial resolution while doubling the channel count.

    Expects a 1x256x256 input (batch dim aside); with the default f=4,
    n=3, s=1, the four blocks bring it to 128x32x32 before flattening to
    the 131072-dim vector that `fc_mu`/`fc_sigma` project down to `k`
    (that flatten size is hardcoded to match those exact defaults - it
    must be recomputed by hand if f/n/s/the input resolution change).
    """

    def __init__(self, f=4, n=3, s=1, k=128, p=1):
        """
        Args:
            f: base channel-width exponent; block i uses 2**(f+i) channels.
            n: conv kernel size (used for every conv in every block).
            s: base stride; each block's first conv uses stride s+1 (the
                one that actually downsamples), its second conv uses
                stride s (typically 1, so it refines features at the new
                resolution without downsampling further).
            k: output latent dimensionality (size of mu/sigma).
            p: convolution padding, applied throughout.
        """
        super(Encoder, self).__init__()

        # Shared across blocks
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(p=0.25)

        # block0: 1 -> 2**f channels, no downsampling yet
        self.conv0 = nn.Conv2d(1, 2**f, kernel_size=n, stride=s, padding=p)

        # block1: downsample (stride s+1) then refine (stride s)
        self.conv1_0 = nn.Conv2d(2**f, 2**(f+1), kernel_size=n, stride=s+1, padding=p)
        self.conv1_1 = nn.Conv2d(2**(f+1), 2**(f+1), kernel_size=n, stride=s, padding=p)

        # block2: downsample again, double channels again
        self.conv2_0 = nn.Conv2d(2**(f+1), 2**(f+2), kernel_size=n, stride=s+1, padding=p)
        self.conv2_1 = nn.Conv2d(2**(f+2), 2**(f+2), kernel_size=n, stride=s, padding=p)

        # block3: downsample a third time
        self.conv3_0 = nn.Conv2d(2**(f+2), 2**(f+3), kernel_size=n, stride=s+1, padding=p)
        self.conv3_1 = nn.Conv2d(2**(f+3), 2**(f+3), kernel_size=n, stride=s, padding=p)

        # block4: flatten the conv output and project to the latent distribution.
        # Hardcoded for the default f=4/n=3/s=1 architecture (128 channels x 32 x 32).
        flatten_size = 131072

        self.fc_mu = nn.Linear(flatten_size, k)
        self.fc_sigma = nn.Linear(flatten_size, k)

    def forward(self, x):
        """Returns (mu, sigma) - sigma is a log-variance, not a raw
        variance (see VAE.forward, which does `exp(0.5 * sigma)`)."""
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
    """Mirror image of `Encoder`: projects a `k`-dim latent vector back up
    to a 1x256x256 reconstruction through the same 4 blocks in reverse,
    using transposed convolutions to upsample instead of downsample.

    Note the `fc` layer here is applied by the caller (`VAE.forward`),
    not inside this module's own `forward` - `forward` expects an
    already-reshaped (batch, 2**(f+3), 32, 32) feature map as input.
    """

    def __init__(self, f=4, n=3, s=1, k=128, p=1):
        """See `Encoder.__init__` for what f/n/s/k/p control - the same
        architecture, mirrored."""
        super(Decoder, self).__init__()

        # Shared across blocks
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(p=0.25)

        # Projects the k-dim latent vector up to the flattened conv shape;
        # applied externally (see VAE.forward) before conv0 runs.
        self.fc = nn.Linear(k, 2**(f+3) * 32 * 32)
        self.conv0 = nn.ConvTranspose2d(k, 2**(f+3), kernel_size=n, stride=s, padding=p)

        # block1: upsample (stride s+1, mirrors Encoder's block3 downsample)
        self.conv1_0 = nn.ConvTranspose2d(2**(f+3), 2**(f+3), kernel_size=n, stride=s, padding=p)
        self.conv1_1 = nn.ConvTranspose2d(2**(f+3), 2**(f+2), kernel_size=n, stride=s+1, padding=p, output_padding=1)

        # block2: upsample again (mirrors Encoder's block2)
        self.conv2_0 = nn.ConvTranspose2d(2**(f+2), 2**(f+2), kernel_size=n, stride=s, padding=p)
        self.conv2_1 = nn.ConvTranspose2d(2**(f+2), 2**(f+1), kernel_size=n, stride=s+1, padding=p, output_padding=1)

        # block3: upsample a third time (mirrors Encoder's block1)
        self.conv3_0 = nn.ConvTranspose2d(2**(f+1), 2**(f+1), kernel_size=n, stride=s, padding=p)
        self.conv3_1 = nn.ConvTranspose2d(2**(f+1), 2**(f), kernel_size=n, stride=s+1, padding=p, output_padding=1)

        # block4: back down to a single output channel; sigmoid'd in forward()
        # to land in [0, 1], matching the BCE reconstruction loss.
        self.conv4 = nn.ConvTranspose2d(2**f, 1, kernel_size=n, stride=s, padding=p)

    def forward(self, x):
        """Expects x already reshaped to (batch, 2**(f+3), 32, 32) (see
        `Decoder.fc` and `VAE.forward`). Returns a (batch, 1, 256, 256)
        image with pixel values in [0, 1]."""
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
    """Resize `img` to fit within a `target_size` x `target_size` box
    (preserving aspect ratio via `thumbnail`), then pad it to exactly
    that size with `fill`, centering the resized image.

    Standalone alternative to `Normalizer.pad_to_square` used directly in
    notebooks/scripts that want padding without instantiating a full
    pipeline Step - same goal (aspect-ratio-preserving square canvas),
    different implementation (resize-then-pad here vs pad-then-resize in
    Normalizer).

    Args:
        img: a PIL Image.
        target_size: final width/height in pixels.
        fill: fill value for the padding (0-255 for a grayscale/RGB image).

    Returns:
        A new PIL Image of exactly (target_size, target_size).
    """
    img = img.copy()
    img.thumbnail((target_size, target_size))
    delta_w = target_size - img.size[0]
    delta_h = target_size - img.size[1]
    padding = (delta_w // 2, delta_h // 2, delta_w - delta_w // 2, delta_h - delta_h // 2)
    return ImageOps.expand(img, padding, fill=fill)