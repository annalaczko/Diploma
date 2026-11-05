"""VAE: a variational autoencoder used as an unsupervised feature learner.

How it works, in short:
    - The Encoder compresses a 256x256 drawing into `k` numbers: the mean
      (`mu`) and log-variance (`sigma`) of a Gaussian in "latent space".
    - During training, a random point `z` is drawn from that Gaussian and
      the Decoder tries to redraw the original image from it.
    - The loss is reconstruction error + beta * KL divergence. The KL term
      keeps the latent space smooth, so similar sherds get similar codes.
    - After training, `mu` of each image is its feature vector, which is
      what clustering uses.

`VAEClassifier` (vae_classifier.py) puts a classifier head on a trained
VAE's encoder.
"""

import math
from pathlib import Path
from typing import Literal

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from pipeline.feature_learning.featurelearning_step import FeatureLearningStep
from pipeline.feature_learning.early_stopping import EarlyStopping
from pipeline.feature_learning.reconstruction_losses import get_reconstruction_loss
from pipeline.preprocessing.dataset import ImageDataset


class VAE(nn.Module, FeatureLearningStep):
    """Convolutional Variational Autoencoder used as an unsupervised
    feature learner: trained to reconstruct grayscale pottery-sherd
    images, then its encoder's latent means (`mu`) are used as a
    fixed-length feature vector per image for downstream clustering.
    """

    def __init__(
        self,
        input_path: str | None = None,
        f=4,
        n=3,
        s=1,
        k=128,
        recon_loss="bce",
        beta_max=1.0,
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
            recon_loss: reconstruction term of the loss - 'bce' (default),
                'dice', 'iou' or 'bce_dice' (see
                `pipeline.feature_learning.reconstruction_losses`), or any
                callable `loss(recon_x, x) -> scalar`.
            beta_max: KL weight reached on the flat stage of each beta
                cycle (see `train_model`); 1.0 is the plain VAE, > 1 a
                beta-VAE. Used by every training run unless `run` /
                `train_model` is given its own `beta_max`.
            output_path: where extracted features / checkpoints are saved.
                Defaults to the Step-computed run-scoped directory.
        """
        nn.Module.__init__(self)
        # Architecture settings, logged to MLflow with each training run.
        self.hparams = {"f": f, "n": n, "s": s, "k": k,
                        "recon_loss": getattr(recon_loss, "__name__", recon_loss), "beta_max": beta_max}
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
        self.recon_loss = get_reconstruction_loss(recon_loss)
        self.beta_max = beta_max

    def run(
        self,
        mode: Literal["extract_features", "train_model"] = "extract_features",
        *,
        epochs=10,
        lr=1e-6,
        batch_size=32,
        early_stopping_patience=15,
        early_stopping_min_delta=0.0,
        early_stopping_smoothing_window=10,
        lr_scheduler_factor=0.5,
        lr_scheduler_patience=5,
        min_lr=1e-8,
        beta_max=None,
    ) -> pd.DataFrame:
        """mode='train_model': train (see `train_model` for the arguments),
        then extract features. mode='extract_features' (default): only
        extract features with the current weights (e.g. a loaded
        checkpoint); only `batch_size` is used then.

        Returns:
            The DataFrame from `extract_features`.
        """
        if mode not in {"extract_features", "train_model"}:
            raise ValueError(
                "mode must be either 'extract_features' or 'train_model'"
            )

        folder = self.input_path

        if mode == "train_model":
            return self.train_model(
                folder,
                epochs=epochs,
                lr=lr,
                batch_size=batch_size,
                early_stopping_patience=early_stopping_patience,
                early_stopping_min_delta=early_stopping_min_delta,
                early_stopping_smoothing_window=early_stopping_smoothing_window,
                lr_scheduler_factor=lr_scheduler_factor,
                lr_scheduler_patience=lr_scheduler_patience,
                min_lr=min_lr,
                beta_max=beta_max,
            )

        return self.extract_features(batch_size=batch_size)

    def forward(self, x):
        """Encode `x`, draw a random latent `z` from N(mu, exp(sigma)) and
        decode it back into an image.

        Returns:
            (x_reconstructed, mu, sigma): the reconstruction and the
            encoder's mean and log-variance (needed for the KL term).
        """
        mu, sigma = self.encoder(x)
        # Reparameterization trick: z = mu + noise * std, so the gradient
        # can flow back through mu and sigma despite the random draw.
        std = torch.exp(0.5 * sigma)
        eps = torch.randn_like(std)
        z = mu + eps * std
        z = self.decoder.fc(z)
        z = z.view(-1, 128, 32, 32)
        x_reconstructed = self.decoder(z)
        return x_reconstructed, mu, sigma

    def train_model(
        self,
        folder,
        epochs=10,
        lr=1e-6,
        batch_size=32,
        early_stopping_patience=15,
        early_stopping_min_delta=0.0,
        early_stopping_smoothing_window=10,
        lr_scheduler_factor=0.5,
        lr_scheduler_patience=5,
        min_lr=1e-8,
        beta_max=None,
    ):
        """Train the encoder and decoder to reconstruct the images, then
        extract and return the features.

        The KL weight `beta` follows a cyclical schedule (`_cyclical_beta`):
        in each of 4 cycles it rises from 0 to `beta_max`, then stays there.
        Starting each cycle low lets the decoder learn to use the latent
        code before the KL term pushes it toward the prior. `beta_max > 1`
        (a beta-VAE) makes latent information more expensive, so the code
        keeps the overall shape rather than pixel details (line thickness,
        small breaks), which is what clustering needs.

        Early stopping (`EarlyStopping`, on a moving average of the training
        loss; there is no validation set) and the learning-rate scheduler
        (`ReduceLROnPlateau`) only look at the epochs where beta == beta_max.
        Only there is the loss comparable between epochs: during the ramp
        it changes with beta, and right after a cycle restart the KL term
        jumps up for a few epochs. The best weights are restored at the end.

        Args:
            folder: image folder to train on (ignored in the in-memory mode).
            epochs: maximum number of epochs.
            lr: Adam learning rate.
            batch_size: training (and extraction) batch size.
            early_stopping_patience: stop after this many monitored epochs
                without improvement of the smoothed loss.
            early_stopping_min_delta: smallest decrease that counts.
            early_stopping_smoothing_window: epochs in the moving average.
            lr_scheduler_factor: multiply the lr by this on a plateau.
            lr_scheduler_patience: plateau length (epochs) before that.
            min_lr: the lr never goes below this.
            beta_max: KL weight on the flat stage of each beta cycle; 1.0 is
                the plain VAE, > 1 a beta-VAE. None uses the constructor's.

        Returns:
            The DataFrame from `extract_features`.
        """
        if beta_max is None:
            beta_max = self.beta_max
        self._log_training_start(**{k: v for k, v in locals().items() if k not in ("self", "folder")})
        device = next(self.parameters()).device
        # target_size=None: the Normalizer already made every image 256x256
        # (an image of another size makes batching fail instead of being
        # silently resized). train_transform: on-the-fly augmentation in
        # the in-memory mode (see set_images).
        dataset = ImageDataset(self._collect_inputs(folder), grayscale=True, target_size=None,
                               transform=self.train_transform)
        print(f"Dataset size: {len(dataset)}")
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True,
            num_workers=16, pin_memory=True, persistent_workers=True
        )
        optimizer = optim.Adam(self.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=lr_scheduler_factor,
            patience=lr_scheduler_patience,
            min_lr=min_lr,
        )
        early_stopping = EarlyStopping(
            patience=early_stopping_patience,
            min_delta=early_stopping_min_delta,
            smoothing_window=early_stopping_smoothing_window,
            mode="min",
        )

        self.train()
        for epoch in range(epochs):
            train_loss = 0.0
            train_elbo = 0.0
            train_recon = 0.0
            train_kld = 0.0
            cycle_frac = self._cyclical_beta(epoch+1, epochs, 4)
            beta = beta_max * cycle_frac
            for i, inputs in enumerate(loader):
                inputs = inputs.to(device, non_blocking=True)
                optimizer.zero_grad()
                outputs, mu, sigma = self(inputs)
                recon, kld = self._loss_terms(outputs, inputs, mu, sigma)
                loss = recon + beta * kld
                if not torch.isfinite(loss):
                    raise RuntimeError(
                        f"Non-finite VAE loss at epoch {epoch + 1}, batch {i} "
                        f"(recon={recon.item()}, kld={kld.item()}) - try a lower lr."
                    )
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                train_elbo += (recon + kld).item()
                train_recon += recon.item()
                train_kld += kld.item()

            average_loss = train_loss / len(loader)
            average_elbo = train_elbo / len(loader)
            # Only the flat stage (beta == beta_max) is monitored, see the docstring.
            if cycle_frac >= 1.0:
                scheduler.step(average_loss)
                smoothed_loss, should_stop = early_stopping.step(average_loss, epoch + 1, model=self)
                smoothed_str = f'{smoothed_loss:.3f}'
            else:
                smoothed_loss = None
                should_stop = False
                smoothed_str = '- (ramp, not monitored)'
            self._log_epoch(
                epoch + 1,
                loss=average_loss,
                elbo=average_elbo,
                recon=train_recon / len(loader),
                kld=train_kld / len(loader),
                smoothed_loss=smoothed_loss,
                beta=beta,
                lr=optimizer.param_groups[0]["lr"],
            )
            print(
                f'[epoch {epoch+1}] loss: {average_loss:.3f} | '
                f'elbo: {average_elbo:.3f} | smoothed loss: {smoothed_str} | '
                f'beta: {beta:.3f} | lr: {optimizer.param_groups[0]["lr"]:.2e}'
            )

            if should_stop:
                print(
                    f"Early stopping at epoch {epoch + 1} "
                    f"(no improvement in smoothed loss for {early_stopping_patience} flat-stage epochs, "
                    f"best smoothed={early_stopping.best_smoothed:.3f} @ epoch {early_stopping.best_epoch})"
                )
                break

        self._log_training_end(early_stopping, epoch + 1)
        early_stopping.restore_best(self)

        return self.extract_features(batch_size=batch_size)

    def extract_features(self, batch_size=32) -> pd.DataFrame:
        """Encode every image (eval mode: no dropout, no random draw) and
        return its latent mean `mu` as the features, plus the log-variance
        `sigma` (the encoder's uncertainty per dimension; only stored,
        clustering uses `mu`).

        Returns:
            A DataFrame with 'filename', 'features' (mu) and 'sigma' per
            image, also written to `<output_path>/features.csv`.
        """
        device = next(self.parameters()).device
        # See train_model's ImageDataset call for why target_size=None here.
        dataset = ImageDataset(self._collect_inputs(), grayscale=True, target_size=None)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False,
            num_workers=16, pin_memory=True, persistent_workers=True
        )

        filenames = dataset.names
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
        self._log_features(output_file)
        return result

    def _loss_terms(self, recon_x, x, mu, logvar):
        """The two parts of the VAE loss, unweighted: the reconstruction
        error (`self.recon_loss`) and the KL divergence of N(mu, exp(logvar))
        from the standard normal. The training loss is recon + beta * KLD;
        recon + KLD is the ELBO, also tracked.
        """
        recon = self.recon_loss(recon_x, x)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon, KLD

    @staticmethod
    def _cyclical_beta(epoch, T, M, R=0.5):
        """Cyclical KL annealing (Fu et al., 2019): training is split into
        `M` cycles; in each, beta rises linearly from 0 to 1 over the first
        `R` fraction, then stays at 1. This prevents "posterior collapse"
        (the decoder learning to ignore the latent code).

        Args:
            epoch: current epoch (1-indexed, matches 'e' in the paper).
            T: total number of epochs.
            M: number of cycles.
            R: proportion of each cycle spent annealing (fraction that ramps 0->1).

        Returns:
            The fraction of `beta_max` to use for this epoch's loss, in [0, 1].
        """
        period = T / M
        tau = ((epoch - 1) % math.ceil(period)) / period
        if tau <= R:
            return tau / R          # linear ramp 0 -> 1
        else:
            return 1.0              # fixed stage, beta = 1


class Encoder(nn.Module):
    """Compresses a 1x256x256 image into a `k`-dimensional Gaussian
    (mu, sigma). Each block halves the resolution and doubles the channels:
    256 -> 128 -> 64 -> 32 px, ending at 128x32x32 with the default
    f=4/n=3/s=1. That is flattened to 131072 numbers, which `fc_mu` and
    `fc_sigma` project down to `k`. The 131072 is hardcoded: recompute it
    if f/n/s or the input size change. SimCLR and BYOL use the same layout.
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
        # Clamp the log-variance: unbounded, exp(sigma) in the KL term can
        # overflow to inf, which turns gradients - and then weights - into NaN.
        sigma = torch.clamp(self.fc_sigma(x), min=-10.0, max=10.0)

        return mu, sigma

class Decoder(nn.Module):
    """The Encoder in reverse: transposed convolutions grow a latent
    vector back into a 1x256x256 image.

    Its `fc` layer (latent vector -> 128x32x32) is applied by the caller
    (`VAE.forward`), so `forward` gets the already reshaped feature map.
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
        # Input is fc's output reshaped to 2**(f+3) channels (see VAE.forward),
        # not the k-dim latent - so in_channels must not depend on k.
        self.conv0 = nn.ConvTranspose2d(2**(f+3), 2**(f+3), kernel_size=n, stride=s, padding=p)

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