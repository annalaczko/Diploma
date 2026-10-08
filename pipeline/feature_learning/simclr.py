"""SimCLR: contrastive self-supervised feature learning, as an
alternative to the VAE's reconstruction-based objective.

Motivation (see the project's notes on why the VAE encoder underperforms
on classification): a VAE's encoder is only ever pushed to encode
whatever makes *pixel reconstruction* easy - which, on largely-white
line-drawing images, ends up being coarse shape rather than the fine
typological detail that actually distinguishes pottery types. SimCLR
(Chen et al., "A Simple Framework for Contrastive Learning of Visual
Representations", 2020) instead trains the encoder to recognize that two
differently-augmented views of the *same* image belong together and are
different from every other image in the batch - a directly discriminative
objective, with no labels required.

Uses the *same* f/n/s/k-controlled conv architecture as
`pipeline.feature_learning.vae.Encoder` (so the two approaches are an
apples-to-apples comparison of *objective*, not of network capacity), but
as its own `SimCLREncoder` rather than reusing that class directly -
`SimCLREncoder` adds BatchNorm after every convolution, which the VAE's
Encoder has never needed but contrastive training does (see
`SimCLREncoder`'s docstring for why).
"""

from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision import transforms

from pipeline.feature_learning.featurelearning_step import FeatureLearningStep
from pipeline.feature_learning.early_stopping import EarlyStopping
from pipeline.preprocessing.augmentation import build_augmentation_transform
from pipeline.preprocessing.dataset import ImageDataset, image_entries, open_image


class SimCLREncoder(nn.Module):
    """Downsamples a 256x256 grayscale image to a `k`-dim embedding,
    through the same 4-block, f/n/s-controlled conv architecture as
    `vae.Encoder` - but with a `BatchNorm2d` after every convolution, and
    a single embedding head instead of a mu/sigma split (SimCLR only ever
    needs one embedding per image, never a distribution to sample from).

    Why BatchNorm here but not in the VAE's Encoder: contrastive losses
    like NT-Xent are prone to *representation collapse* - the encoder
    settling into emitting a near-constant embedding for every input,
    since early in training (before any real structure has been learned)
    that can look like a stable low-gradient point to the optimizer. Once
    collapsed, every pairwise similarity is roughly equal, which is
    exactly what produces a loss stuck at ln(2N-1) (the "random guessing"
    baseline) no matter how long training runs. BatchNorm normalizes
    activations *across the batch*, which prevents this by forcing the
    batch's embeddings to stay spread out - this is why virtually every
    contrastive/self-supervised architecture uses it (see also Grill et
    al., "Bootstrap Your Own Latent", 2020, which studied this failure
    mode directly). The VAE's reconstruction loss has no equivalent
    collapse mode, so its Encoder never needed BatchNorm.
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


class SimCLRDataset(Dataset):
    """Yields two independently-augmented views of each image - the
    positive pair SimCLR's contrastive loss is trained to recognize as
    "the same underlying image". Unlike `ImageDataset`, this has no
    labels and no notion of a fixed "correct" output - both views are
    randomly generated fresh every time an item is fetched (i.e. every
    epoch sees different crops/jitter/blur of the same source image).
    """

    def __init__(self, paths, transform):
        """
        Args:
            paths: sequence of image file paths, or an `InMemoryImages`.
            transform: a torchvision transform (PIL in, PIL out) applied
                independently twice per image to produce the two views;
                typically `build_augmentation_transform(...)` composed
                with `transforms.ToTensor()`.
        """
        self.paths, _ = image_entries(paths)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = open_image(self.paths, idx).convert("L")
        view1 = self.transform(image)
        view2 = self.transform(image)
        return view1, view2


class SimCLR(nn.Module, FeatureLearningStep):
    """Contrastive feature learner: a `SimCLREncoder` backbone plus a
    small projection head, trained with the NT-Xent contrastive loss on
    pairs of augmented views. Once trained, features for downstream
    clustering are taken from the *encoder* output (`h`), not the
    projection head's output (`z`) - per the SimCLR paper, `h` transfers
    better to downstream tasks, while `z` is only useful for computing
    the contrastive loss during training and is discarded afterward.
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
            f, n, s, k: `SimCLREncoder` architecture hyperparameters - see
                its docstring for what each controls. Match the VAE's own
                `f`/`n`/`s`/`k` values here to keep a fair,
                architecture-controlled comparison between the two.
            projection_dim: output size of the projection head `g(h)`
                used only for the contrastive loss; smaller than `k` is
                standard practice (forces the loss to work with a more
                compressed space than the features actually extracted).
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
        """Either train the encoder+projection head from scratch with the
        contrastive loss, or just embed `input_path`'s images with the
        current (e.g. previously loaded) encoder weights.

        Args:
            mode: 'train_model' trains for `epochs` epochs and then
                extracts features from the trained encoder;
                'extract_features' (default) skips training and just runs
                inference - use this once you already have trained
                weights loaded.
            epochs: maximum training epochs, only used when
                mode='train_model' (may stop earlier via early stopping).
            lr: Adam learning rate, only used when mode='train_model'.
            batch_size: batch size for both training and feature
                extraction. Training batches double as the negative-pair
                pool for the contrastive loss, so very small batch sizes
                give the loss too few negatives to be a useful signal.
            temperature: NT-Xent temperature - lower values sharpen the
                loss's focus on the hardest negatives, but need a larger
                batch_size to stay stable. 0.5 is a safer default than
                the original paper's 0.1, which assumed batch sizes in
                the thousands.
            view_transform: PIL-to-tensor transform used to generate each
                of the two augmented views per image; only used when
                mode='train_model'. Defaults to
                `build_augmentation_transform()` + `ToTensor()` if not
                given. Pass this in to reuse an existing, already-tuned
                `Augmentor`'s exact augmentation settings instead of
                SimCLR defining its own separate config - see
                `train_model`'s docstring for how.
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
        """Train the encoder+projection head with the NT-Xent contrastive
        loss on pairs of randomly-augmented views of each image, then
        extract and return features from the trained encoder.

        There's no validation split here (this is unsupervised
        contrastive training, not classification), so early stopping
        instead watches a smoothed moving average of the *training* loss
        - see `EarlyStopping` - and stops once it's stopped improving for
        `early_stopping_patience` epochs, restoring the best-epoch
        weights before returning.

        Args:
            folder: image folder to train on.
            epochs: maximum number of training epochs (may stop earlier
                via early stopping).
            lr: Adam learning rate.
            batch_size: training batch size (see `run`'s note on why this
                also controls the number of negatives per loss term).
            temperature: NT-Xent temperature (see `run`).
            view_transform: PIL-to-tensor transform used to generate the
                two augmented views of each image. If not given, builds
                one from `build_augmentation_transform()`'s defaults.

                To reuse an `Augmentor` you've already configured (e.g.
                with custom `crop_scale`/`brightness`/...), pass its
                `.transform` attribute instead of letting SimCLR define
                a second, separately-tuned augmentation config:

                    from pipeline.preprocessing.augmentation import Augmentor
                    from torchvision import transforms

                    # Only used for its config/.transform - never call .run().
                    augmentor = Augmentor(input_path=folder, crop_scale=(0.7, 1.0))
                    view_transform = transforms.Compose([augmentor.transform, transforms.ToTensor()])

                    simclr.train_model(folder, view_transform=view_transform)
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
        self._log_training_start(**{k: v for k, v in locals().items() if k not in ("self", "folder")})
        device = next(self.parameters()).device

        if view_transform is None:
            # Same augmentation recipe as the Augmentor preprocessing step -
            # SimCLR needs two independently, strongly augmented views of
            # each image to have something non-trivial to contrast against.
            view_transform = transforms.Compose([
                build_augmentation_transform(),
                transforms.ToTensor(),
            ])
        if self.train_transform is not None:
            print(
                f"Warning: {self.name} makes its own augmented views (view_transform); "
                "the pipeline's train_transform (Augmentor) is ignored."
            )
        dataset = SimCLRDataset(self._collect_inputs(folder), view_transform)
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
        """Run the encoder over every image in `self.input_path` in eval
        mode and collect `h` (the encoder's embedding, pre-projection) as
        that image's feature vector.

        Args:
            batch_size: inference batch size.

        Returns:
            A DataFrame with one 'filename' and 'features' (`h` as a
            numpy array) row per image; also written to
            `<output_path>/features.csv`. Same shape as
            `VAE.extract_features`'s output, so it drops straight into
            the same downstream clustering/evaluation code.
        """
        device = next(self.parameters()).device
        # target_size=None: this pipeline's Normalizer step already makes
        # every image the same size, so resizing here would be redundant.
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
        """Normalized Temperature-scaled Cross Entropy loss (SimCLR's
        contrastive loss).

        For a batch of N images, z1/z2 are the N projected, L2-normalized
        embeddings of each image's two augmented views. Stacking them
        gives 2N embeddings; for each one, its *only* positive is the
        other view of the same source image, and every other one of the
        2N-2 remaining embeddings is a negative. The loss pushes the
        cosine similarity high for the positive pair and low for all
        negatives, scaled by `temperature` (lower = sharper distinction).

        Implemented as a single cross-entropy over the (2N, 2N) cosine
        similarity matrix - self-similarity (the diagonal) is masked out
        so nothing is ever its own negative or positive, and the target
        class for row i is simply the index of i's positive partner.

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
