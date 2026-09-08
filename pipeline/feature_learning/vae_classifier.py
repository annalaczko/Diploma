"""Supervised classifier built on top of a trained VAE's encoder.

See `pipeline.feature_learning.vae.VAE` for the unsupervised counterpart
whose encoder `VAEClassifier` wraps and (optionally) fine-tunes.
"""

import pandas as pd

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

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
