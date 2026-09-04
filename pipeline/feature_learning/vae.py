"""Concrete feature learning steps built around a convolutional VAE.

Two Step subclasses live here:
    VAE           - unsupervised: trains an Encoder/Decoder pair to
                    reconstruct pottery-sherd images, and can extract the
                    encoder's latent means as a feature vector per image.
    VAEClassifier - supervised: wraps a *trained* VAE's encoder with a
                    small classifier head and fine-tunes it to predict
                    pottery type labels, for a direct comparison against
                    the unsupervised features.

Encoder/Decoder are the underlying conv-net architecture; image loading
(path discovery + tensor conversion) is delegated to
`pipeline.preprocessing.dataset`, shared with the preprocessing steps.
"""

import math
from typing import Literal
import pandas as pd

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from PIL import ImageOps

from pipeline.feature_learning.featurelearning_step import FeatureLearningStep
from pipeline.preprocessing.dataset import ImageDataset, collect_image_paths

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight


class VAEClassifier(nn.Module, FeatureLearningStep):
    """Supervised classifier built on top of a (typically pretrained) VAE
    encoder: reuses its conv layers to embed an image into the latent
    space, then predicts a pottery type label from that embedding with a
    small MLP head (see `__init__`).

    Inherits from both `nn.Module` (for the actual network/training) and
    `FeatureLearningStep` (so it fits into the same Step/Pipeline
    interface as the unsupervised approaches, even though its `run()`
    trains a classifier rather than extracting features).
    """

    def __init__(
        self,
        vae,
        num_classes=28,
        labels_path=None,
        freeze_encoder=True,
        output_path=None,
        name=None,
        delete_input_data=False,
        save_results=False,
        run_id="default_run",
        input_path=None,
    ):
        """
        Args:
            vae: a `VAE` instance whose `encoder` is reused here. Its
                weights are shared directly (not copied), so training the
                classifier will also update the VAE's encoder in place,
                unless `freeze_encoder` is set.
            num_classes: placeholder output size for the classifier head;
                it gets rebuilt in `run()` to match the actual number of
                classes present in the labelled data, so this value only
                matters if you use the model before calling `run()`.
            labels_path: CSV path with 'ID' and 'TypeRev' columns mapping
                image stems to pottery type labels. Required by `run()`.
            freeze_encoder: if True, the VAE encoder's weights are frozen
                (only the classifier head is trained) - useful to check
                how good the VAE's unsupervised features are on their own,
                versus fine-tuning the whole network end to end.
            input_path: root folder to search for labelled images under;
                defaults to `vae.input_path` if not given.
        """
        nn.Module.__init__(self)

        FeatureLearningStep.__init__(
            self,
            input_path=input_path or vae.input_path,
            output_path=output_path,
            name=name,
            delete_input_data=delete_input_data,
            save_results=save_results,
            run_id=run_id
        )

        self.encoder = vae.encoder
        self.labels_path = labels_path
        self.freeze_encoder = freeze_encoder

        latent_size = self.encoder.fc_mu.out_features

        if freeze_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False

        # Two-hidden-layer MLP head, halving the width each layer
        # (latent_size -> latent_size/2 -> latent_size/4 -> num_classes),
        # clamped to a minimum of 32 units so this stays sane for small
        # latent sizes. LayerNorm on the input stabilizes training since
        # the VAE's latent scale isn't otherwise normalized; Dropout
        # combats overfitting given the relatively small labelled dataset.
        hidden_size_1 = max(latent_size // 2, 32)
        hidden_size_2 = max(hidden_size_1 // 2, 32)
        self.classifier = nn.Sequential(
            nn.LayerNorm(latent_size),
            nn.Linear(latent_size, hidden_size_1),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(hidden_size_1, hidden_size_2),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(hidden_size_2, num_classes),
        )

    def forward(self, x):
        """Embed `x` with the VAE encoder (using only its mean, not a
        sampled latent - classification should be deterministic) and run
        the classifier head on top. Returns raw logits (n, num_classes)."""
        mu, _ = self.encoder(x)
        return self.classifier(mu)

    def run(
        self,
        epochs=30,
        lr=1e-4,
        batch_size=64,
        validation_size=0.2,
        num_workers=0,
        early_stopping_patience=15,
        early_stopping_min_delta=0.0,
        early_stopping_smoothing_window=10,
        class_weighted_loss=True,
        min_samples_per_class=10,
    ) -> pd.DataFrame:
        """Train the classifier head (and, unless frozen, the wrapped VAE
        encoder) end to end on the labelled images under `input_path`.

        Overview of what happens here, in order:
            1. Load labels from `labels_path`, keep only images that have
               one, and drop classes with too few samples to be useful
               (see `_filter_rare_classes`).
            2. Rebuild the classifier's final layer to match the actual
               number of surviving classes (`num_classes` from `__init__`
               was just a placeholder).
            3. Stratified train/val split, optionally class-weighted loss.
            4. Standard train loop with early stopping on a moving average
               of validation accuracy (see the epoch loop below).
            5. Restore the best-epoch weights, save the model + a per-epoch
               training history CSV, and return that history as a DataFrame.

        Args:
            epochs: maximum number of epochs to train for (may stop
                earlier via early stopping).
            lr: Adam learning rate.
            batch_size: batch size for both train and validation loaders.
            validation_size: fraction of labelled images held out for
                validation (stratified by class).
            num_workers: DataLoader worker processes.
            early_stopping_patience: stop after this many consecutive
                epochs with no improvement in the smoothed validation
                accuracy.
            early_stopping_min_delta: minimum increase in smoothed
                validation accuracy to count as an improvement.
            early_stopping_smoothing_window: number of most recent epochs
                averaged together for the early-stopping signal, so a
                single noisy epoch can't trigger a stop on its own.
            class_weighted_loss: if True, weight the loss inversely to
                each class's frequency in the training split, so rare
                pottery types aren't drowned out by common ones.
            min_samples_per_class: classes with fewer labelled images than
                this are excluded entirely (see `_filter_rare_classes`).

        Returns:
            A DataFrame with one row per epoch actually run, columns
            'epoch', 'loss', 'train_accuracy', 'validation_accuracy', and
            'validation_accuracy_smoothed'.
        """

        # Checking label path correctness and reading labels

        if not self.labels_path:
            raise ValueError("labels_path is required")

        labels_df = pd.read_csv(self.labels_path)
        labels_df["ID"] = labels_df["ID"].astype(str)

        # Checking that the number of classes is correct and encoding labels
        encoder = LabelEncoder()
        labels_df["label"] = encoder.fit_transform(
            labels_df["TypeRev"].astype(str)
        )

        if len(encoder.classes_) != 28:
            raise ValueError(
                f"Expected 28 classes, got {len(encoder.classes_)}"
            )

        # Creating a mapping from image IDs to labels and filtering the dataset to include only labelled images

        labels_by_id = dict(
            zip(labels_df["ID"], labels_df["label"])
        )

        # Filtering the dataset to include only labelled images

        paths = sorted(
            path for path in collect_image_paths(self.input_path)
            if path.stem in labels_by_id
        )

        print(f"Found {len(paths)} labelled images in {self.input_path}")

        if not paths:
            raise ValueError("No labelled images found")

        labels = [labels_by_id[path.stem] for path in paths]

        paths, labels, label_order = self._filter_rare_classes(
            paths, labels, min_samples_per_class
        )

        classes = encoder.classes_[label_order]
        self.classes_ = classes.tolist()
        print(
            f"Using {len(classes)} classes and {len(paths)} labelled images "
            f"after excluding classes with fewer than {min_samples_per_class} samples"
        )
        classifier_device = next(self.parameters()).device
        self.classifier[-1] = nn.Linear(
            self.classifier[-1].in_features,
            len(classes),
        ).to(classifier_device)

        

        train_paths, val_paths, train_labels, val_labels = train_test_split(
            paths,
            labels,
            test_size=validation_size,
            random_state=42,
            stratify=labels,
        )
        # Kept for downstream evaluation (e.g. confusion matrix) once run() returns.
        self.val_paths_ = val_paths
        self.val_labels_ = val_labels

        loader_kwargs = {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "pin_memory": torch.cuda.is_available(),
        }

        train_loader = torch.utils.data.DataLoader(
            ImageDataset(train_paths, labels=train_labels, grayscale=True, target_size=256),
            shuffle=True,
            **loader_kwargs,
        )

        val_loader = torch.utils.data.DataLoader(
            ImageDataset(val_paths, labels=val_labels, grayscale=True, target_size=256),
            shuffle=False,
            **loader_kwargs,
        )

        device = next(self.parameters()).device

        if class_weighted_loss:
            # Inverse-frequency weighting so rare classes contribute as much
            # to the loss as common ones, computed on the train split only
            # (val stays untouched so val_acc reflects true performance).
            class_weights = compute_class_weight(
                class_weight="balanced",
                classes=np.arange(len(classes)),
                y=train_labels,
            )
            weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
            criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        else:
            criterion = nn.CrossEntropyLoss()

        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=lr,
        )

        history = []
        val_acc_history = []
        best_smoothed_val_acc = -1.0
        best_epoch = 0
        best_state_dict = None
        epochs_without_improvement = 0

        for epoch in range(epochs):
            self.train()

            if self.freeze_encoder:
                self.encoder.eval()

            train_loss = 0.0
            train_correct = 0
            train_count = 0

            for images, targets in train_loader:
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                optimizer.zero_grad()

                logits = self(images)
                loss = criterion(logits, targets)

                loss.backward()
                optimizer.step()

                train_loss += loss.item() * images.size(0)
                train_correct += (
                    logits.argmax(dim=1) == targets
                ).sum().item()
                train_count += images.size(0)

            self.eval()
            val_correct = 0
            val_count = 0

            with torch.no_grad():
                for images, targets in val_loader:
                    images = images.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)

                    logits = self(images)

                    val_correct += (
                        logits.argmax(dim=1) == targets
                    ).sum().item()
                    val_count += images.size(0)

            train_acc = train_correct / train_count
            val_acc = val_correct / val_count
            average_loss = train_loss / train_count

            val_acc_history.append(val_acc)
            smoothed_val_acc = sum(
                val_acc_history[-early_stopping_smoothing_window:]
            ) / len(val_acc_history[-early_stopping_smoothing_window:])

            history.append({
                "epoch": epoch + 1,
                "loss": average_loss,
                "train_accuracy": train_acc,
                "validation_accuracy": val_acc,
                "validation_accuracy_smoothed": smoothed_val_acc,
            })

            print(
                f"[{epoch + 1}/{epochs}] "
                f"loss={average_loss:.4f} | "
                f"train_acc={train_acc:.4f} | "
                f"val_acc={val_acc:.4f} | "
                f"val_acc_smoothed={smoothed_val_acc:.4f}"
            )

            # Early stopping runs on the moving average of val_acc rather
            # than the raw per-epoch value, so a single noisy epoch can't
            # trigger it while the underlying trend is still improving.
            if smoothed_val_acc > best_smoothed_val_acc + early_stopping_min_delta:
                best_smoothed_val_acc = smoothed_val_acc
                best_epoch = epoch + 1
                best_state_dict = {
                    key: value.detach().cpu().clone()
                    for key, value in self.state_dict().items()
                }
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= early_stopping_patience:
                    print(
                        f"Early stopping at epoch {epoch + 1} "
                        f"(no improvement in smoothed val_acc for {early_stopping_patience} epochs, "
                        f"best smoothed={best_smoothed_val_acc:.4f} @ epoch {best_epoch})"
                    )
                    break

        if best_state_dict is not None:
            self.load_state_dict(best_state_dict)

        output_dir = Path(self.output_path or "classifier_results")
        output_dir.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "classes": classes.tolist(),
            },
            output_dir / "vae_classifier.pt",
        )

        history_df = pd.DataFrame(history)
        history_df.to_csv(output_dir / "training_history.csv", index=False)

        return history_df

    @staticmethod
    def _filter_rare_classes(paths, labels, min_samples_per_class):
        """Drop samples belonging to classes with fewer than
        `min_samples_per_class` labelled images, then remap the surviving
        class ids to a dense 0..N-1 range (required since the classifier's
        output layer size and CrossEntropyLoss both assume contiguous ids).

        Args:
            paths: sequence of image paths, one per sample.
            labels: sequence of original (encoder-assigned) label ids,
                aligned with `paths`.
            min_samples_per_class: classes with fewer labelled images than
                this are dropped entirely. Must be >= 2, since the
                downstream stratified train/val split requires at least
                two samples per class.

        Returns:
            (filtered_paths, remapped_labels, label_order) where
            label_order is the sorted list of original label ids that
            survived filtering; remapped_labels[i] indexes into
            label_order (i.e. `label_order[remapped_labels[i]]` recovers
            the original label id for filtered_paths[i]).
        """
        if min_samples_per_class < 2:
            raise ValueError(
                "min_samples_per_class must be >= 2 for a stratified "
                f"train/val split, got {min_samples_per_class}"
            )

        label_counts = pd.Series(labels).value_counts()
        valid_labels = set(
            label_counts[label_counts >= min_samples_per_class].index.tolist()
        )

        filtered_data = [
            (path, label)
            for path, label in zip(paths, labels)
            if label in valid_labels
        ]
        if not filtered_data:
            raise ValueError(
                f"No classes with at least {min_samples_per_class} labelled images found"
            )

        filtered_paths, filtered_labels = zip(*filtered_data)

        label_order = sorted(valid_labels)
        label_mapping = {label: index for index, label in enumerate(label_order)}
        remapped_labels = [label_mapping[label] for label in filtered_labels]

        return list(filtered_paths), remapped_labels, label_order

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
        dataset = ImageDataset(collect_image_paths(folder), grayscale=True, target_size=256)
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
        mode and collect the latent mean (`mu`) as that image's feature
        vector - not a sampled `z`, so features are deterministic and
        reproducible across calls.

        Args:
            batch_size: inference batch size.

        Returns:
            A DataFrame with one 'filename' and 'features' (`mu` as a
            numpy array) row per image; also written to
            `<output_path>/features.csv`.
        """
        device = next(self.parameters()).device
        dataset = ImageDataset(collect_image_paths(self.input_path), grayscale=True, target_size=256)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False,
            num_workers=16, pin_memory=True, persistent_workers=True
        )

        filenames = [Path(p).name for p in dataset.paths]
        self.eval()
        all_features = []
        with torch.no_grad():
            for inputs in loader:
                inputs = inputs.to(device, non_blocking=True)
                mu, _ = self.encoder(inputs)
                all_features.append(mu.cpu())

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