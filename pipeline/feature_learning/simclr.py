"""SimCLR: contrastive self-supervised feature learning (Chen et al., 2020).

The idea: make two random variations ("views") of every image. The encoder
is trained so that the two views of the same image get similar vectors,
and views of different images get different ones. No labels are needed.

Compared with the VAE, which learns whatever helps to redraw the pixels,
this objective directly rewards telling sherds apart. The encoder has the
same layout as the VAE's (same f/n/s/k), so the two can be compared on the
training objective alone; `SimCLREncoder` only adds BatchNorm.
"""

from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms

from pipeline.feature_learning.featurelearning_step import FeatureLearningStep
from pipeline.feature_learning.early_stopping import EarlyStopping
from pipeline.preprocessing.augmentation import build_augmentation_transform
from pipeline.preprocessing.dataset import ImageDataset, TwoViewDataset


class SimCLREncoder(nn.Module):
    """The VAE's Encoder layout (see `vae.Encoder`), with two differences:
    a `BatchNorm2d` after every convolution, and one `k`-dim output (an
    embedding, not a mean/variance pair).

    Why BatchNorm: contrastive training can "collapse", with the encoder
    giving every image nearly the same vector (the loss then sticks at
    ln(2N-1), the random-guess value). BatchNorm normalizes across the
    batch, which keeps the vectors of a batch spread out. The VAE's loss
    has no such failure mode, so its Encoder does without.
    """

    def __init__(self, f=4, n=3, s=1, k=128, p=1):
        """See `vae.Encoder.__init__` for what f/n/s/k/p control - the
        same architecture, plus a BatchNorm2d after each conv."""
        super().__init__()

        self.elu = nn.ELU()
        self.dropout = nn.Dropout(p=0.25)

        # block0: 1 -> 2**f channels, no downsampling yet
        self.conv0 = nn.Conv2d(1, 2**f, kernel_size=n, stride=s, padding=p)
        self.bn0 = nn.BatchNorm2d(2**f)

        # block1: downsample (stride s+1) then refine (stride s)
        self.conv1_0 = nn.Conv2d(2**f, 2**(f+1), kernel_size=n, stride=s+1, padding=p)
        self.bn1_0 = nn.BatchNorm2d(2**(f+1))
        self.conv1_1 = nn.Conv2d(2**(f+1), 2**(f+1), kernel_size=n, stride=s, padding=p)
        self.bn1_1 = nn.BatchNorm2d(2**(f+1))

        # block2: downsample again, double channels again
        self.conv2_0 = nn.Conv2d(2**(f+1), 2**(f+2), kernel_size=n, stride=s+1, padding=p)
        self.bn2_0 = nn.BatchNorm2d(2**(f+2))
        self.conv2_1 = nn.Conv2d(2**(f+2), 2**(f+2), kernel_size=n, stride=s, padding=p)
        self.bn2_1 = nn.BatchNorm2d(2**(f+2))

        # block3: downsample a third time
        self.conv3_0 = nn.Conv2d(2**(f+2), 2**(f+3), kernel_size=n, stride=s+1, padding=p)
        self.bn3_0 = nn.BatchNorm2d(2**(f+3))
        self.conv3_1 = nn.Conv2d(2**(f+3), 2**(f+3), kernel_size=n, stride=s, padding=p)
        self.bn3_1 = nn.BatchNorm2d(2**(f+3))

        # block4: flatten and project to the embedding.
        # Hardcoded for the default f=4/n=3/s=1 architecture (128 channels
        # x 32 x 32), same reasoning as vae.Encoder's flatten_size.
        flatten_size = 131072
        self.fc = nn.Linear(flatten_size, k)

    def forward(self, x):
        """Returns a (batch, k) embedding tensor."""
        x = self.bn0(self.conv0(x))

        x = self.elu(self.bn1_0(self.conv1_0(x)))
        x = self.elu(self.bn1_1(self.conv1_1(x)))
        x = self.dropout(x)

        x = self.elu(self.bn2_0(self.conv2_0(x)))
        x = self.elu(self.bn2_1(self.conv2_1(x)))
        x = self.dropout(x)

        x = self.elu(self.bn3_0(self.conv3_0(x)))
        x = self.elu(self.bn3_1(self.conv3_1(x)))
        x = self.dropout(x)

        x = x.view(x.size(0), -1)
        return self.fc(x)

class SimCLR(nn.Module, FeatureLearningStep):
    """Contrastive feature learner: a `SimCLREncoder` plus a small
    projection head, trained with the NT-Xent loss on pairs of views.

    The features are the encoder's output `h`, not the projection head's
    output `z`: `z` is only used by the loss (the SimCLR paper found `h`
    works better for later tasks).
    """

    def __init__(
        self,
        input_path: str | None = None,
        f=4,
        n=3,
        s=1,
        k=128,
        projection_dim=64,
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
            f, n, s, k: encoder layout, as in `vae.Encoder` (use the VAE's
                values for a fair comparison).
            projection_dim: output size of the projection head, only used
                by the loss; usually smaller than `k`.
        """
        nn.Module.__init__(self)
        # Architecture settings, logged to MLflow with each training run.
        self.hparams = {"f": f, "n": n, "s": s, "k": k, "projection_dim": projection_dim}
        FeatureLearningStep.__init__(
            self,
            input_path=input_path or "",
            output_path=output_path,
            name=name,
            delete_input_data=delete_input_data,
            save_results=save_results,
            run_id=run_id,
        )
        self.encoder = SimCLREncoder(f, n, s, k)
        self.projection_head = nn.Sequential(
            nn.Linear(k, k),
            nn.ReLU(inplace=True),
            nn.Linear(k, projection_dim),
        )

    def forward(self, x):
        """Returns (h, z): `h` is the encoder's embedding; `z` is `h`
        passed through the projection head, L2-normalized, used only for
        the contrastive loss.
        """
        h = self.encoder(x)
        z = F.normalize(self.projection_head(h), dim=1)
        return h, z

    def run(
        self,
        mode: str = "extract_features",
        *,
        epochs=100,
        lr=1e-3,
        batch_size=128,
        temperature=0.5,
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
                temperature=temperature,
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
        batch_size=128,
        temperature=0.5,
        view_transform=None,
        early_stopping_patience=15,
        early_stopping_min_delta=0.0,
        early_stopping_smoothing_window=10,
    ):
        """Train with the NT-Xent loss on pairs of views, then extract and
        return the features. Early stopping watches a moving average of the
        training loss (there is no validation set) and restores the best
        epoch's weights.

        Args:
            folder: image folder to train on (ignored in the in-memory mode).
            epochs: maximum number of epochs.
            lr: Adam learning rate.
            batch_size: training batch size. The other images of a batch are
                the "negatives" of the loss, so a very small batch gives a
                weak training signal.
            temperature: NT-Xent temperature; lower focuses on the hardest
                negatives but needs larger batches. 0.5 is safer than the
                paper's 0.1, which assumed batches of thousands.
            view_transform: PIL -> tensor transform making each view;
                default: `build_augmentation_transform()` + `ToTensor()`.
                To reuse a configured Augmentor:
                `transforms.Compose([augmentor.transform, transforms.ToTensor()])`.
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
            # The Augmentor's default recipe.
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
        optimizer = optim.Adam(self.parameters(), lr=lr)
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
                _, z1 = self(view1)
                _, z2 = self(view2)
                loss = self._nt_xent_loss(z1, z2, temperature)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            average_loss = train_loss / len(loader)
            smoothed_loss, should_stop = early_stopping.step(average_loss, epoch + 1, model=self)
            self._log_epoch(epoch + 1, loss=average_loss, smoothed_loss=smoothed_loss,
                            lr=optimizer.param_groups[0]["lr"])
            print(
                f"[epoch {epoch + 1}/{epochs}] loss: {average_loss:.3f} | "
                f"smoothed: {smoothed_loss:.3f}"
            )

            if should_stop:
                print(
                    f"Early stopping at epoch {epoch + 1} "
                    f"(no improvement in smoothed loss for {early_stopping_patience} epochs, "
                    f"best smoothed={early_stopping.best_smoothed:.3f} @ epoch {early_stopping.best_epoch})"
                )
                break

        self._log_training_end(early_stopping, epoch + 1)
        early_stopping.restore_best(self)

        return self.extract_features(batch_size=batch_size)

    def extract_features(self, batch_size=32) -> pd.DataFrame:
        """Encode every image (eval mode) and return the encoder output `h`
        as its features: a DataFrame with 'filename' and 'features', also
        written to `<output_path>/features.csv`."""
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
                h = self.encoder(inputs)
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
    def _nt_xent_loss(z1, z2, temperature):
        """NT-Xent (normalized temperature-scaled cross entropy), SimCLR's loss.

        A batch of N images gives 2N vectors (two views each). For each
        vector, the other view of the same image is the one "correct
        answer" and the remaining 2N-2 are wrong. This is written as a
        classification: cross-entropy over each row of the (2N, 2N) cosine
        similarity matrix, whose target is the partner's index. The
        diagonal (a vector with itself) is masked out.

        Args:
            z1, z2: (N, projection_dim) L2-normalized embeddings of the
                two views, aligned so z1[i] and z2[i] are the same image.
            temperature: softmax temperature; see `run`'s docstring.

        Returns:
            A scalar loss tensor.
        """
        batch_size = z1.size(0)
        device = z1.device

        z = torch.cat([z1, z2], dim=0)  # (2N, D)
        similarity = torch.matmul(z, z.T) / temperature  # (2N, 2N) cosine sim, since z is L2-normalized

        # No embedding can be its own positive/negative.
        self_mask = torch.eye(2 * batch_size, dtype=torch.bool, device=device)
        similarity.masked_fill_(self_mask, float("-inf"))

        # Row i's positive is at i+N (mod 2N): view1[i] <-> view2[i].
        targets = (torch.arange(2 * batch_size, device=device) + batch_size) % (2 * batch_size)

        return F.cross_entropy(similarity, targets)
