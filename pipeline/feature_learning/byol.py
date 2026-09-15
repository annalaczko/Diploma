"""BYOL (Bootstrap Your Own Latent): self-supervised feature learning
without negative pairs, as a second alternative to the VAE's
reconstruction-based objective (alongside `pipeline.feature_learning.simclr.SimCLR`).

Like SimCLR, BYOL (Grill et al., "Bootstrap Your Own Latent: A New
Approach to Self-Supervised Learning", 2020) trains the encoder to relate
two differently-augmented views of the same image - but unlike SimCLR, it
never contrasts against *other* images at all. There's no NT-Xent, no
negative pairs, and therefore no dependence on batch size for the loss to
have enough negatives to be meaningful (SimCLR's `run()` docstring notes
this exact limitation) - a real practical advantage on a dataset this
size, where large batches are impractical.

Instead, BYOL keeps two copies of the network:
  - the "online" network (encoder + projector + predictor), trained by
    gradient descent as usual.
  - the "target" network (encoder + projector only, no predictor),
    *never* trained by gradient descent - its weights are an exponential
    moving average (EMA) of the online network's weights.
The online network's predicted projection of one view is trained to
match the target network's projection of the *other* view (stop-gradient
on the target side), and vice versa.

Reuses `pipeline.feature_learning.simclr.SimCLREncoder` (with BatchNorm)
and `SimCLRDataset` (two augmented views per image) rather than
duplicating them - BYOL needs both just as much as SimCLR does. If
anything, BYOL is *more* dependent on BatchNorm than SimCLR: BYOL is the
paper that first showed removing BatchNorm from this kind of
architecture collapses training to a trivial constant solution, since
without negative pairs, BatchNorm's cross-batch normalization is the main
remaining mechanism stopping the online/target networks from converging
on "predict the same constant regardless of input" as a trivially
perfect (and useless) solution.
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
from pipeline.feature_learning.simclr import SimCLREncoder, SimCLRDataset
from pipeline.preprocessing.augmentation import build_augmentation_transform
from pipeline.preprocessing.dataset import ImageDataset, collect_image_paths


class BYOL(nn.Module, FeatureLearningStep):
    """Self-supervised feature learner with an online/target network pair
    and no negative pairs. See the module docstring for the overall idea.

    Once trained, downstream features are taken from the *online
    encoder's* embedding (`h`) - same convention as `SimCLR` and `VAE`
    (never from a projection/prediction head, which exist only to shape
    the training loss).
    """

    def __init__(
        self,
        input_path: str,
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
            f, n, s, k: `SimCLREncoder` architecture hyperparameters -
                match the VAE's/SimCLR's own values here to keep a fair,
                architecture-controlled comparison across all three.
            hidden_dim: hidden layer size of the projector/predictor MLPs.
            projection_dim: output size of the projector/predictor; a
                smaller space than `k`, same reasoning as SimCLR's
                `projection_dim`.
            tau: EMA momentum for updating the target network's weights
                from the online network's, each optimizer step:
                `target = tau * target + (1 - tau) * online`. Close to 1
                (0.99 here) means the target changes slowly/smoothly,
                which is what keeps it a stable prediction target for the
                online network. The original paper ramps this up to 1.0
                over training via a cosine schedule; a fixed value is a
                simpler approximation that's fine at this project's scale.
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
        self.tau = tau

        self.online_encoder = SimCLREncoder(f, n, s, k)
        self.online_projector = self._make_mlp(k, hidden_dim, projection_dim)
        self.online_predictor = self._make_mlp(projection_dim, hidden_dim, projection_dim)

        # Target network: an exact copy of the online encoder+projector at
        # init, updated only via EMA afterward - never by its own gradient.
        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.target_projector = copy.deepcopy(self.online_projector)
        for parameter in self.target_encoder.parameters():
            parameter.requires_grad = False
        for parameter in self.target_projector.parameters():
            parameter.requires_grad = False

    @staticmethod
    def _make_mlp(input_dim, hidden_dim, output_dim):
        """The projector/predictor MLP shape BYOL uses throughout: one
        hidden layer with BatchNorm (see the module docstring for why
        BatchNorm matters even more here than in SimCLR) before the
        output projection.
        """
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, view1, view2):
        """Computes BYOL's symmetric loss directly from a pair of views -
        unlike `VAE`/`SimCLR`'s `forward`, which returns an embedding for
        a single input. BYOL's online-predicts-target computation only
        makes sense given two views at once, so returning the scalar loss
        here (instead of, say, just the online embedding) avoids
        duplicating this logic between `forward` and the training loop.

        Returns:
            A scalar loss tensor (already averaged over the batch and
            both prediction directions).
        """
        online_p1 = self.online_predictor(self.online_projector(self.online_encoder(view1)))
        online_p2 = self.online_predictor(self.online_projector(self.online_encoder(view2)))

        # Stop-gradient on the target side: the target network is only
        # ever updated via EMA (see _update_target_network), never by
        # backpropagating through these calls.
        with torch.no_grad():
            target_z1 = self.target_projector(self.target_encoder(view1))
            target_z2 = self.target_projector(self.target_encoder(view2))

        # Symmetric: predict view2's target from view1's online output,
        # and vice versa, so both views get used both ways.
        loss = self._byol_loss(online_p1, target_z2) + self._byol_loss(online_p2, target_z1)
        return loss.mean()

    @torch.no_grad()
    def _update_target_network(self):
        """Push the target network's weights toward the online network's
        current weights by `tau`. Called once per training step, after
        the optimizer has updated the online network - never the other
        way around, and never via gradient descent on the target side.
        """
        for online_param, target_param in zip(
            self.online_encoder.parameters(), self.target_encoder.parameters()
        ):
            target_param.data.mul_(self.tau).add_(online_param.data, alpha=1 - self.tau)
        for online_param, target_param in zip(
            self.online_projector.parameters(), self.target_projector.parameters()
        ):
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
        """Either train the online/target networks from scratch, or just
        embed `input_path`'s images with the current (e.g. previously
        loaded) online encoder weights.

        Args:
            mode: 'train_model' trains for `epochs` epochs and then
                extracts features from the trained online encoder;
                'extract_features' (default) skips training and just runs
                inference - use this once you already have trained
                weights loaded.
            epochs: maximum training epochs, only used when
                mode='train_model' (may stop earlier via early stopping).
            lr: Adam learning rate, only used when mode='train_model'.
            batch_size: batch size for both training and feature
                extraction. Unlike SimCLR, BYOL has no negative-pair pool,
                so this doesn't need to be large for the loss itself to
                make sense - pick it based on memory/throughput, not on
                needing "enough negatives".
            view_transform: PIL-to-tensor transform used to generate each
                of the two augmented views per image; only used when
                mode='train_model'. See `train_model`'s docstring - same
                mechanism as `SimCLR.run`'s `view_transform`.
            early_stopping_patience: stop after this many consecutive
                epochs with no improvement in the smoothed training loss.
            early_stopping_min_delta: minimum decrease in smoothed
                training loss to count as an improvement.
            early_stopping_smoothing_window: number of most recent epochs
                averaged together for the early-stopping signal, so a
                single noisy epoch can't trigger a stop on its own.

        Returns:
            A DataFrame with one 'filename' and 'features' (`h` as a
            numpy array) row per image, as produced by `extract_features`.
        """
        if mode not in {"extract_features", "train_model"}:
            raise ValueError(
                "mode must be either 'extract_features' or 'train_model'"
            )

        folder = self.input_path
        if not folder:
            raise ValueError("input_path is required to run the BYOL feature step")

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
        """Train the online network (encoder+projector+predictor) to
        predict the target network's projection of the other augmented
        view, updating the target network via EMA after every step, then
        extract and return features from the trained online encoder.

        There's no validation split here (this is unsupervised training,
        not classification), so early stopping instead watches a smoothed
        moving average of the *training* loss - see `EarlyStopping` - and
        stops once it's stopped improving for `early_stopping_patience`
        epochs, restoring the best-epoch weights before returning.

        Args:
            folder: image folder to train on.
            epochs: maximum number of training epochs (may stop earlier
                via early stopping).
            lr: Adam learning rate. Only the *online* network's
                parameters are optimized - the target network is never
                touched by the optimizer, only by `_update_target_network`.
            batch_size: training batch size (see `run`'s note on why this
                is much less constrained than SimCLR's).
            view_transform: PIL-to-tensor transform used to generate the
                two augmented views of each image. If not given, builds
                one from `build_augmentation_transform()`'s defaults. To
                reuse an existing, already-tuned `Augmentor`'s exact
                settings instead, pass its `.transform` composed with
                `transforms.ToTensor()` - see `SimCLR.train_model`'s
                docstring for a worked example (identical mechanism here).
            early_stopping_patience: stop after this many consecutive
                epochs with no improvement in the smoothed training loss.
            early_stopping_min_delta: minimum decrease in smoothed
                training loss to count as an improvement.
            early_stopping_smoothing_window: number of most recent epochs
                averaged together for the early-stopping signal, so a
                single noisy epoch can't trigger a stop on its own.

        Returns:
            The DataFrame from `extract_features` (filenames + latent
            vectors), run once training is complete.
        """
        device = next(self.parameters()).device

        if view_transform is None:
            view_transform = transforms.Compose([
                build_augmentation_transform(),
                transforms.ToTensor(),
            ])
        dataset = SimCLRDataset(collect_image_paths(folder), view_transform)
        print(f"Dataset size: {len(dataset)}")
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True, drop_last=True,
            num_workers=16, pin_memory=True, persistent_workers=True,
        )
        # Only the online network is trained by gradient descent - the
        # target network's weights only ever move via EMA (see
        # _update_target_network), so it's deliberately excluded here.
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

        early_stopping.restore_best(self)

        return self.extract_features(batch_size=batch_size)

    def extract_features(self, batch_size=32) -> pd.DataFrame:
        """Run the *online* encoder over every image in `self.input_path`
        in eval mode and collect its embedding (`h`) as that image's
        feature vector.

        Args:
            batch_size: inference batch size.

        Returns:
            A DataFrame with one 'filename' and 'features' (`h` as a
            numpy array) row per image; also written to
            `<output_path>/features.csv`. Same shape as
            `VAE.extract_features`'s/`SimCLR.extract_features`'s output,
            so it drops straight into the same downstream
            clustering/evaluation code.
        """
        device = next(self.parameters()).device
        # target_size=None: this pipeline's Normalizer step already makes
        # every image the same size, so resizing here would be redundant.
        dataset = ImageDataset(collect_image_paths(self.input_path), grayscale=True, target_size=None)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False,
            num_workers=16, pin_memory=True, persistent_workers=True,
        )

        filenames = [Path(p).name for p in dataset.paths]
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
        return result

    @staticmethod
    def _byol_loss(pred, target):
        """BYOL's per-sample loss: the squared error between L2-normalized
        vectors, `||pred/||pred|| - target/||target|||^2`, which expands
        to `2 - 2 * cosine_similarity(pred, target)` - 0 when pred and
        target point in exactly the same direction, up to 4 when they
        point in opposite directions. `target` is expected to already be
        detached from the graph (see `forward`'s `torch.no_grad()` block).

        Args:
            pred: (N, projection_dim) online network's prediction.
            target: (N, projection_dim) target network's projection.

        Returns:
            (N,) per-sample loss values (not yet reduced to a scalar).
        """
        pred = F.normalize(pred, dim=1)
        target = F.normalize(target, dim=1)
        return 2 - 2 * (pred * target).sum(dim=1)
