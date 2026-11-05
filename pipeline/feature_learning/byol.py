"""BYOL (Bootstrap Your Own Latent, Grill et al., 2020): self-supervised
feature learning without negative pairs.

Like SimCLR, BYOL learns from two random views of each image, but it never
compares an image with other images. So the loss does not need large
batches, which suits a small dataset. It uses two copies of the network:
  - the "online" network (encoder + projector + predictor), trained by
    gradient descent as usual;
  - the "target" network (encoder + projector), never trained directly:
    its weights slowly follow the online network's (an exponential moving
    average, see `tau`).
The online network learns to predict the target network's output for the
other view of the same image.

It reuses SimCLR's encoder (with BatchNorm) and dataset (two views per
image). Without negatives, BatchNorm is what stops both networks from
settling on the useless solution "always output the same vector".
"""

import copy
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms

from pipeline.feature_learning.featurelearning_step import FeatureLearningStep
from pipeline.feature_learning.early_stopping import EarlyStopping
from pipeline.feature_learning.simclr import SimCLREncoder
from pipeline.preprocessing.augmentation import build_augmentation_transform
from pipeline.preprocessing.dataset import ImageDataset, TwoViewDataset


class BYOL(nn.Module, FeatureLearningStep):
    """Self-supervised feature learner (see the module docstring). The
    features are the online encoder's output `h`; the projector and
    predictor only serve the loss.
    """

    def __init__(
        self,
        input_path: str | None = None,
        f=4,
        n=3,
        s=1,
        k=128,
        hidden_dim=256,
        projection_dim=64,
        tau=0.99,
        output_path=None,
        name=None,
        delete_input_data=False,
        save_results=False,
        run_id="default_run",
    ):
        """
        Args:
            input_path: folder of images to train on / extract features
                from (searched recursively).
            f, n, s, k: encoder layout, as in `vae.Encoder` (use the same
                values as the VAE/SimCLR for a fair comparison).
            hidden_dim: hidden layer size of the projector/predictor.
            projection_dim: output size of the projector/predictor.
            tau: after every step, `target = tau * target + (1 - tau) *
                online`. Close to 1, the target changes slowly, so it is a
                stable thing to predict. (The paper raises tau toward 1
                during training; a fixed value is a simplification.)
        """
        nn.Module.__init__(self)
        # Architecture settings, logged to MLflow with each training run.
        self.hparams = {"f": f, "n": n, "s": s, "k": k, "hidden_dim": hidden_dim, "projection_dim": projection_dim, "tau": tau}
        FeatureLearningStep.__init__(
            self,
            input_path=input_path or "",
            output_path=output_path,
            name=name,
            delete_input_data=delete_input_data,
            save_results=save_results,
            run_id=run_id,
        )
        self.tau = tau

        self.online_encoder = SimCLREncoder(f, n, s, k)
        self.online_projector = self._make_mlp(k, hidden_dim, projection_dim)
        self.online_predictor = self._make_mlp(projection_dim, hidden_dim, projection_dim)

        # The target network starts as a copy of the online one and is
        # afterwards only moved by _update_target_network, not by gradients.
        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.target_projector = copy.deepcopy(self.online_projector)
        for parameter in self.target_encoder.parameters():
            parameter.requires_grad = False
        for parameter in self.target_projector.parameters():
            parameter.requires_grad = False

    @staticmethod
    def _make_mlp(input_dim, hidden_dim, output_dim):
        """The projector/predictor: one hidden layer with BatchNorm."""
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, view1, view2):
        """The training loss for a batch of view pairs (unlike VAE/SimCLR,
        `forward` returns the loss, since it needs both views at once):
        the online network predicts the target's output for the other view,
        in both directions, averaged over the batch.
        """
        online_p1 = self.online_predictor(self.online_projector(self.online_encoder(view1)))
        online_p2 = self.online_predictor(self.online_projector(self.online_encoder(view2)))

        # No gradients through the target network (it is only moved by EMA).
        with torch.no_grad():
            target_z1 = self.target_projector(self.target_encoder(view1))
            target_z2 = self.target_projector(self.target_encoder(view2))

        # Symmetric: predict view2's target from view1's online output,
        # and vice versa, so both views get used both ways.
        loss = self._byol_loss(online_p1, target_z2) + self._byol_loss(online_p2, target_z1)
        return loss.mean()

    @torch.no_grad()
    def _update_target_network(self):
        """Move the target weights a step toward the online weights:
        target = tau * target + (1 - tau) * online. Called after every
        optimizer step."""
        pairs = [(self.online_encoder, self.target_encoder), (self.online_projector, self.target_projector)]
        for online, target in pairs:
            for online_param, target_param in zip(online.parameters(), target.parameters()):
                target_param.data.mul_(self.tau).add_(online_param.data, alpha=1 - self.tau)

    def run(
        self,
        mode: str = "extract_features",
        *,
        epochs=100,
        lr=1e-3,
        batch_size=64,
        view_transform=None,
        early_stopping_patience=15,
        early_stopping_min_delta=0.0,
        early_stopping_smoothing_window=10,
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
                view_transform=view_transform,
                early_stopping_patience=early_stopping_patience,
                early_stopping_min_delta=early_stopping_min_delta,
                early_stopping_smoothing_window=early_stopping_smoothing_window,
            )

        return self.extract_features(batch_size=batch_size)

    def train_model(
        self,
        folder,
        epochs=100,
        lr=1e-3,
        batch_size=64,
        view_transform=None,
        early_stopping_patience=15,
        early_stopping_min_delta=0.0,
        early_stopping_smoothing_window=10,
    ):
        """Train the online network, moving the target network after every
        step, then extract and return the features. Early stopping watches
        a moving average of the training loss (there is no validation set)
        and restores the best epoch's weights.

        Args:
            folder: image folder to train on (ignored in the in-memory mode).
            epochs: maximum number of epochs.
            lr: Adam learning rate (only the online network is optimized).
            batch_size: training batch size; no negatives are needed, so
                choose it for memory and speed.
            view_transform: PIL -> tensor transform making each view; see
                `SimCLR.train_model`.
            early_stopping_patience: stop after this many epochs without
                improvement of the smoothed loss.
            early_stopping_min_delta: smallest decrease that counts.
            early_stopping_smoothing_window: epochs in the moving average.

        Returns:
            The DataFrame from `extract_features`.
        """
        self._log_training_start(**{k: v for k, v in locals().items() if k not in ("self", "folder")})
        device = next(self.parameters()).device

        if view_transform is None:
            view_transform = transforms.Compose([
                build_augmentation_transform(),
                transforms.ToTensor(),
            ])
        if self.train_transform is not None:
            print(
                f"Warning: {self.name} makes its own augmented views (view_transform); "
                "the pipeline's train_transform (Augmentor) is ignored."
            )
        dataset = TwoViewDataset(self._collect_inputs(folder), view_transform)
        print(f"Dataset size: {len(dataset)}")
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True, drop_last=True,
            num_workers=16, pin_memory=True, persistent_workers=True,
        )
        # Only the online network is optimized (see _update_target_network).
        optimizer = optim.Adam(
            list(self.online_encoder.parameters())
            + list(self.online_projector.parameters())
            + list(self.online_predictor.parameters()),
            lr=lr,
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
            for view1, view2 in loader:
                view1 = view1.to(device, non_blocking=True)
                view2 = view2.to(device, non_blocking=True)

                optimizer.zero_grad()
                loss = self(view1, view2)
                loss.backward()
                optimizer.step()
                self._update_target_network()
                train_loss += loss.item()

            average_loss = train_loss / len(loader)
            smoothed_loss, should_stop = early_stopping.step(average_loss, epoch + 1, model=self)
            self._log_epoch(epoch + 1, loss=average_loss, smoothed_loss=smoothed_loss,
                            lr=optimizer.param_groups[0]["lr"])
            print(
                f"[epoch {epoch + 1}/{epochs}] loss: {average_loss:.4f} | "
                f"smoothed: {smoothed_loss:.4f}"
            )

            if should_stop:
                print(
                    f"Early stopping at epoch {epoch + 1} "
                    f"(no improvement in smoothed loss for {early_stopping_patience} epochs, "
                    f"best smoothed={early_stopping.best_smoothed:.4f} @ epoch {early_stopping.best_epoch})"
                )
                break

        self._log_training_end(early_stopping, epoch + 1)
        early_stopping.restore_best(self)

        return self.extract_features(batch_size=batch_size)

    def extract_features(self, batch_size=32) -> pd.DataFrame:
        """Encode every image with the online encoder (eval mode) and return
        its output `h` as the features: a DataFrame with 'filename' and
        'features', also written to `<output_path>/features.csv`."""
        device = next(self.parameters()).device
        # target_size=None: the Normalizer already made all images one size.
        dataset = ImageDataset(self._collect_inputs(), grayscale=True, target_size=None)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False,
            num_workers=16, pin_memory=True, persistent_workers=True,
        )

        filenames = dataset.names
        self.eval()
        all_features = []
        with torch.no_grad():
            for inputs in loader:
                inputs = inputs.to(device, non_blocking=True)
                h = self.online_encoder(inputs)
                all_features.append(h.cpu())

        features = torch.cat(all_features, dim=0).numpy()

        result = pd.DataFrame({
            "filename": filenames,
            "features": list(features),
        })
        output_file = Path(self.output_path) / "features.csv"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_file, index=False)
        self._log_features(output_file)
        return result

    @staticmethod
    def _byol_loss(pred, target):
        """Per-sample loss: squared distance of the length-normalized
        vectors, which equals 2 - 2 * cosine_similarity (0 = same
        direction, 4 = opposite). Returns an (N,) tensor.
        """
        pred = F.normalize(pred, dim=1)
        target = F.normalize(target, dim=1)
        return 2 - 2 * (pred * target).sum(dim=1)
