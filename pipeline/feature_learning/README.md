# `pipeline.feature_learning`

Turns preprocessed pottery-sherd images into fixed-length feature vectors -
unsupervised, via `VAE` (reconstruction-based), `SimCLR` (contrastive,
needs negative pairs) or `BYOL` (no negative pairs) - or predicts a
pottery type label directly (supervised, `VAEClassifier`, built on top of
a trained `VAE`'s encoder). Having three unsupervised approaches with
different training objectives but the *same* encoder architecture lets
you isolate whether the objective or the network capacity is the
bottleneck for downstream classification.

```
pipeline/feature_learning/
├── featurelearning_step.py   # FeatureLearningStep: abstract base
├── early_stopping.py          # EarlyStopping: shared moving-average stopping logic
├── vae.py                     # VAE, Encoder, Decoder, resize_with_padding
├── reconstruction_losses.py   # interchangeable VAE reconstruction losses (BCE, Dice, soft-IoU, BCE+Dice)
├── vae_classifier.py          # VAEClassifier
├── simclr.py                  # SimCLR, SimCLREncoder, SimCLRDataset
├── byol.py                    # BYOL
├── handcrafted_features.py    # HandcraftedFeatures: profile measures from the Vectorizer's SVGs (model.type: handcrafted)
└── dinov3.py                  # DINOv3: pretrained backbone (frozen or fine-tuned), CLS token as features (model.type: dinov3)
```

Image loading (path discovery + tensor conversion) is delegated to
`pipeline.preprocessing.dataset` (`collect_image_paths`, `ImageDataset`) —
see [`pipeline/preprocessing/README.md`](../preprocessing/README.md).

## `FeatureLearningStep`

```python
from pipeline.feature_learning import FeatureLearningStep
```

Common contract for feature-learning approaches. `run()` takes preprocessed
images (from `self.input_path`) and returns a fixed-length feature vector
(or matrix of vectors) per input, so different approaches stay directly
comparable downstream. Also provides:

```python
FeatureLearningStep.l2_normalize(vectors: np.ndarray) -> np.ndarray
```

L2-normalizes feature vectors row-wise — required before distance/cosine
similarity comparisons.

```python
step.set_images(images: InMemoryImages, train_transform=None)
```

In-memory mode (see `Pipeline` in [`pipeline/README.md`](../README.md)):
the step then trains on / extracts features from these images instead of
reading `input_path`. If given, `train_transform` (e.g. `Augmentor.transform`)
is applied on the fly to training images only, never during feature
extraction or validation, so extracted features stay deterministic.
`set_images(None)` switches back to `input_path`.

MLflow logging (no-ops without an active run, see `pipeline/tracking.py`):
every training loop calls `_log_training_start(**train_args)` (the model's
`hparams`, e.g. `f/n/s/k/recon_loss`, and the training arguments), then
`_log_epoch(epoch, ...)` once per epoch (`feature/loss`, `smoothed_loss`,
`lr`, and model-specific values) and `_log_training_end(early_stopping,
last_epoch)` (`best_epoch`, `stopped_epoch`, `best_smoothed`).
`extract_features` attaches `features.csv` via `_log_features`.

## `EarlyStopping` — shared early-stopping logic

```python
from pipeline.feature_learning import EarlyStopping
```

Every training loop in this package (`VAE`, `SimCLR`, `BYOL`, `VAEClassifier`)
stops early the same way: a metric is smoothed with a trailing moving
average over `smoothing_window` epochs, then compared against the best
smoothed value seen so far. Training stops once `patience` consecutive
epochs pass with no improvement of at least `min_delta`. Smoothing first
means a single noisy epoch can't trigger a stop (or reset the patience
counter) on its own — only a real, sustained plateau does.

```python
EarlyStopping(patience=15, min_delta=0.0, smoothing_window=10, mode="min")
```

- `mode="min"`: lower is better — used by `VAE`/`SimCLR`/`BYOL`, which
  have no validation split and instead watch their own smoothed
  **training loss**.
- `mode="max"`: higher is better — used by `VAEClassifier`, which watches
  smoothed **validation accuracy**.

```python
smoothed_value, should_stop = early_stopping.step(value, epoch, model=self)
...
early_stopping.restore_best(self)
```

`step(value, epoch, model=None)` records this epoch's raw value, and — if
`model` is given and this epoch improves on the best smoothed value so
far — snapshots `model.state_dict()` (detached, on CPU) as the new best.
`restore_best(model)` loads that snapshot back once training loops end,
so every one of these models resumes/returns with its *best*-epoch
weights, not necessarily its *last*-epoch ones.

## `VAE` — unsupervised feature learning

A convolutional Variational Autoencoder: trained to reconstruct grayscale
256×256 pottery-sherd images, then its encoder's latent means (`mu`) are
used as a fixed-length feature vector per image for downstream clustering.

```python
VAE(
    input_path=None,   # folder of images (searched recursively); optional in in-memory mode (set_images)
    f=4,    # base channel-width exponent; block i uses 2**(f+i) channels
    n=3,    # convolution kernel size
    s=1,    # base stride (see Encoder/Decoder below for the exact layout)
    k=128,  # latent dimensionality — the feature vector size
    recon_loss="bce",  # 'bce', 'dice', 'iou', 'bce_dice', or any callable loss(recon_x, x) -> scalar
    beta_max=1.0,      # KL weight on the flat stage of each beta cycle (see below); 1.0 = plain VAE
    output_path=None,
    run_id="default_run",
)
```

Input images must already be 256×256 (e.g. after `Normalizer`). The VAE's
`ImageDataset` is built with `target_size=None`, so a wrongly sized image
fails when the `DataLoader` batches it instead of being silently resized.

### `run(mode="extract_features", *, epochs=10, lr=1e-6, batch_size=32, early_stopping_patience=15, early_stopping_min_delta=0.0, early_stopping_smoothing_window=10, lr_scheduler_factor=0.5, lr_scheduler_patience=5, min_lr=1e-8, beta_max=None)`

- `mode="train_model"`: trains for up to `epochs` epochs (may stop earlier
  via early stopping), then extracts features from the trained model.
- `mode="extract_features"` (default): skips training, just runs
  inference — use this once you already have trained weights loaded (e.g.
  via `vae.load_state_dict(...)`).

The remaining arguments are passed through to `train_model` (below).

### `forward(x)`

Encodes `x`, samples a latent `z` via the reparameterization trick (so
gradients can flow through the sampling step: `z = mu + eps * std`, `eps ~
N(0, I)`), and decodes it back into a reconstruction. Returns
`(x_reconstructed, mu, sigma)` — `sigma` is a **log-variance**, not a raw
variance.

### `train_model(folder, epochs=10, lr=1e-6, batch_size=32, early_stopping_patience=15, early_stopping_min_delta=0.0, early_stopping_smoothing_window=10, lr_scheduler_factor=0.5, lr_scheduler_patience=5, min_lr=1e-8, beta_max=None)`

`folder` is ignored in in-memory mode (see `set_images` above).

Loss (`_loss_function`, a standard beta-VAE loss):

```
loss = recon_loss(reconstruction, input) + beta * KLD(mu, logvar)
```

The reconstruction term is `self.recon_loss` (BCE by default, see
[Reconstruction losses](#reconstruction-losses) below). `KLD` pulls the
latent distribution toward a standard normal. A higher `beta` trades
reconstruction quality for a more disentangled/regularized latent space,
which is generally better for downstream clustering.

**Cyclical beta.** The KL weight is annealed on a cyclical schedule
(`_cyclical_beta`, after Fu et al., "Cyclical Annealing Schedule", 2019)
instead of held constant. Training is split into 4 cycles. In each cycle,
beta ramps linearly from 0 to `beta_max` over the first half, then holds at
`beta_max` for the second half. This avoids *posterior collapse* (the
decoder learning to ignore the latent code, which a constant high beta from
epoch 1 tends to cause) while still reaching full KL regularization in each
cycle. `beta_max` is set in the constructor (so a `Pipeline` can configure
it with the model); a `beta_max` passed to `run` / `train_model` overrides it
for that training run, and the default `None` keeps the constructor's
value. `beta_max=1.0` is the plain VAE. `beta_max > 1` is a beta-VAE: the
reconstruction term is summed over all 256×256 pixels but the KL only over
the `k` latent dims, so at beta=1 the model can afford to spend latent
capacity on pixel-level detail (line thickness, small breaks) rather than
the sherd's shape.

**Early stopping and LR schedule.** There's no validation split (this is
unsupervised reconstruction), so early stopping (see
[`EarlyStopping`](#earlystopping--shared-early-stopping-logic) above) watches
a smoothed moving average of the epoch's **training loss** (`mode="min"`).
The learning rate is lowered by `ReduceLROnPlateau` (`lr_scheduler_factor`,
`lr_scheduler_patience`, `min_lr`) on the same loss. Both only see epochs
from the *flat* stage of each cycle (beta == `beta_max`). During the ramp the
loss rises with beta, and when beta drops back to 0 at a cycle restart the
KL term blows up for several epochs. Without this rule, every new cycle
would use up the patience and halve the LR. Ramp epochs don't enter the
smoothing window or count toward patience, so the restored best weights
always come from a flat-stage epoch.

Each epoch prints the loss, the beta-independent ELBO (recon + KLD), the
smoothed loss, beta and the current LR. A non-finite loss raises a
`RuntimeError` (usually a sign that `lr` is too high).

### `extract_features(batch_size=32) -> pd.DataFrame`

Runs the encoder over every image (from `input_path`, or the in-memory
images) in eval mode. It collects the latent **mean** (`mu`, not a sampled
`z`, so features are deterministic and reproducible) as each image's
feature vector, plus the log-variance. Returns, and also writes to
`<output_path>/features.csv`, a DataFrame with one row per image and
columns `filename`, `features` (`mu`) and `sigma` (log-variance). `sigma`
shows how uncertain the encoder is per latent dimension. Downstream steps
only use `features`.

### Reconstruction losses

`reconstruction_losses.py` defines interchangeable losses with the same
signature, `loss(recon_x, x) -> scalar`, for `(batch, 1, H, W)` inputs in
`[0, 1]`. `VAE(recon_loss=...)` takes one by name or any callable with that
signature (e.g. `functools.partial(dice_loss, smooth=0.1)`):

| Name | Function | What it measures |
|---|---|---|
| `"bce"` | `bce_loss` | Pixel-wise binary cross-entropy, summed (the original VAE loss). |
| `"dice"` | `dice_loss(smooth=1.0, ink_is_dark=True)` | Soft Dice on the ink pixels: `1 - 2·|P∩T| / (|P| + |T|)`. |
| `"iou"` | `soft_iou_loss(smooth=1.0, ink_is_dark=True)` | Soft IoU (Jaccard) on the ink pixels; penalizes partial overlap more harshly than Dice. |
| `"bce_dice"` | `bce_dice_loss(dice_weight=1.0)` | BCE + Dice. Dice drives outline overlap, BCE keeps per-pixel gradients (Dice alone can be unstable early, while predicted ink is a uniform gray). |

The images are thin dark outlines on a mostly white background. A
pixel-averaged loss like BCE is dominated by that background, so a
reconstruction that misses most of the outline still scores a small error.
Dice and IoU measure how well the reconstruction's *ink* overlaps the
target's, so background size doesn't matter, while ink drawn where there is
none is still penalized. `ink_is_dark=True` inverts pixel values first,
because `ToTensor` makes the white background 1 and the black outline 0.

To keep the beta-weighted KL term at about the same relative strength
whichever loss is used, the overlap losses are computed per image in
`[0, 1]` and multiplied by the pixel count. That puts them on the same scale
as the summed BCE. An unknown name raises `ValueError`
(`get_reconstruction_loss`).

## `Encoder` / `Decoder`

The conv-net architecture underlying `VAE`. `Encoder` downsamples a
1×256×256 grayscale image to a `k`-dim latent distribution `(mu, sigma)`
through 4 blocks, each roughly halving spatial resolution while doubling
channel count; with the defaults (`f=4, n=3, s=1`) this ends at
128×32×32 before flattening to a 131072-dim vector that `fc_mu`/`fc_sigma`
project down to `k`. **That flatten size (`131072`) is hardcoded** to match
those exact defaults — recompute it by hand if `f`/`n`/`s`/the input
resolution change. `Decoder` mirrors the same architecture in reverse, using
transposed convolutions to upsample, ending in a sigmoid so output pixels
land in `[0, 1]` (matching the BCE reconstruction loss).

## `VAEClassifier` — supervised classification

Wraps a (typically pretrained) `VAE`'s encoder, reusing its conv layers to
embed an image into the latent space, then predicts a pottery type label
from that embedding with a small MLP head. Inherits from both `nn.Module`
(the network) and `FeatureLearningStep` (so it fits the same Step/Pipeline
interface, even though `run()` trains a classifier rather than extracting
features).

```python
VAEClassifier(
    vae,                    # a VAE instance; its encoder is reused (shared, not copied)
    num_classes=28,          # placeholder; run() rebuilds the final layer to the real class count
    labels_path=None,        # CSV with 'ID' and 'TypeRev' columns; required by run()
    freeze_encoder=True,     # if True, only the classifier head is trained
    input_path=None,         # defaults to vae.input_path
)
```

If `freeze_encoder=False`, training the classifier also updates the VAE's
encoder weights in place (they're shared, not copied) — useful for
end-to-end fine-tuning once you've checked the frozen-encoder baseline.

### Classifier head

A two-hidden-layer MLP, halving width each layer down from the latent size
(clamped to a minimum of 32 units):

```
LayerNorm(latent) → Linear → ReLU → Dropout(0.3)
                  → Linear → ReLU → Dropout(0.3)
                  → Linear(num_classes)
```

`LayerNorm` on the input stabilizes training since the VAE's latent scale
isn't otherwise normalized; `Dropout` combats overfitting given the
relatively small labelled dataset.

### `run(...)`

```python
run(
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
) -> pd.DataFrame
```

What happens, in order:

1. **Load labels** from `labels_path` (`ID`, `TypeRev` columns), keep only
   images that have one, and drop classes with fewer than
   `min_samples_per_class` labelled images (`_filter_rare_classes` — also
   remaps the surviving class ids to a dense `0..N-1` range). Raises if
   `min_samples_per_class < 2`, since the stratified split below needs at
   least two samples per class.
2. **Rebuild the classifier's final layer** to match the actual number of
   surviving classes (the `num_classes` passed to `__init__` was only a
   placeholder).
3. **Stratified train/val split** (`validation_size`, `random_state=42`).
   `self.val_paths_` / `self.val_labels_` / `self.classes_` are saved on the
   instance afterward — handy for building a confusion matrix or other
   evaluation once `run()` returns, without re-deriving the split. In
   in-memory mode `val_paths_` is an `InMemoryImages` subset rather than a
   list of paths. `ImageDataset` accepts either, so the confusion-matrix
   example below works in both modes. If `set_images` was given a
   `train_transform`, it is applied to the training split only.
4. **Optionally class-weighted loss** (`class_weighted_loss=True` by
   default): `sklearn.utils.class_weight.compute_class_weight("balanced",
   ...)` on the *train* split only (validation stays untouched, so
   `val_acc` reflects true, unweighted performance), so rare pottery types
   don't get drowned out by common ones in the loss.
5. **Standard train loop**, with early stopping via the shared
   [`EarlyStopping`](#earlystopping--shared-early-stopping-logic) helper
   (`mode="max"`): each epoch's validation accuracy is smoothed over the
   last `early_stopping_smoothing_window` epochs before being compared to
   the running best, so a single noisy epoch can't trigger a stop while
   the underlying trend is still improving. Training stops after
   `early_stopping_patience` consecutive epochs with no improvement of at
   least `early_stopping_min_delta` in the smoothed value.
6. **Restore the best-epoch weights** (by smoothed val accuracy, not the
   last epoch's), save the model (`<output_path>/vae_classifier.pt`, with
   `model_state_dict` and `classes`) and a per-epoch training history CSV
   (`<output_path>/training_history.csv`), and return that history as a
   DataFrame (columns: `epoch`, `loss`, `train_accuracy`,
   `validation_accuracy`, `validation_accuracy_smoothed`).

### Building a confusion matrix after training

```python
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from pipeline.preprocessing.dataset import ImageDataset

device = next(classifier.parameters()).device
val_loader = torch.utils.data.DataLoader(
    ImageDataset(classifier.val_paths_, labels=classifier.val_labels_),
    batch_size=64, shuffle=False,
)

y_true, y_pred = [], []
classifier.eval()
with torch.no_grad():
    for images, targets in val_loader:
        logits = classifier(images.to(device))
        y_pred.extend(logits.argmax(dim=1).cpu().tolist())
        y_true.extend(targets.tolist())

cm = confusion_matrix(y_true, y_pred, labels=range(len(classifier.classes_)))
ConfusionMatrixDisplay(cm, display_labels=classifier.classes_).plot(xticks_rotation="vertical")
```

## `SimCLR` — contrastive feature learning

An unsupervised alternative to `VAE`: instead of reconstruction, it trains
via **contrastive learning** (Chen et al., "A Simple Framework for
Contrastive Learning of Visual Representations", 2020) - the encoder
learns to recognize that two differently-augmented views of the *same*
image belong together, and are different from every other image in the
batch. This is a directly discriminative objective (no labels needed),
unlike the VAE's reconstruction loss, which only ever rewards encoding
whatever makes pixel reconstruction easy - typically coarse shape rather
than the fine typological detail that distinguishes pottery types.

Uses the same `f`/`n`/`s`/`k`-controlled 4-block conv architecture as
`vae.Encoder`, so a difference in downstream clustering/classification
accuracy between the two can be attributed to the training objective, not
to one having a bigger network - but as its own `SimCLREncoder` class,
not `vae.Encoder` directly, since it adds a `BatchNorm2d` after every
convolution. Contrastive losses like NT-Xent are prone to *representation
collapse* (the encoder settling into a near-constant output for every
input, since early in training that can look like a stable low-gradient
point) - BatchNorm prevents this by forcing each batch's embeddings to
stay spread out. In practice, training without it gets visibly stuck: the
loss sits exactly at `ln(2N-1)` (the "random guessing" baseline for batch
size N) and never moves, no matter how many epochs run - a telltale sign
of collapse, not just slow convergence. The VAE's reconstruction loss has
no equivalent collapse mode, so its `Encoder` never needed BatchNorm.

```python
SimCLR(
    input_path=None,   # folder of images (searched recursively); optional in in-memory mode (set_images)
    f=4, n=3, s=1, k=128,  # SimCLREncoder architecture - match your VAE's values
    projection_dim=64,      # output size of the projection head used for the loss
    output_path=None,
    run_id="default_run",
)
```

### `run(mode="extract_features", *, epochs=100, lr=1e-3, batch_size=128, temperature=0.5, view_transform=None, early_stopping_patience=15, early_stopping_min_delta=0.0, early_stopping_smoothing_window=10)`

Same two-mode contract as `VAE.run`: `mode="train_model"` trains then
extracts features; `mode="extract_features"` (default) just runs
inference with whatever encoder weights are currently loaded. `batch_size`
matters more here than for the VAE - each training batch also supplies
the negative examples for the contrastive loss, so very small batches
give it too few negatives to be a useful signal. `temperature` controls
how sharply the loss focuses on hard negatives; the default (0.5) is
safer than the original paper's 0.1, which assumed batch sizes in the
thousands.

### `forward(x)`

Returns `(h, z)`: `h` is the `SimCLREncoder`'s embedding; `z` is `h` run
through the projection head and L2-normalized. **Downstream feature
extraction uses `h`, never `z`** - per the SimCLR paper, the
pre-projection representation transfers better to downstream tasks, while
`z` exists only to compute the contrastive loss during training.

### `train_model(folder, epochs=100, lr=1e-3, batch_size=128, temperature=0.5, view_transform=None, early_stopping_patience=15, early_stopping_min_delta=0.0, early_stopping_smoothing_window=10)`

For each batch, builds two independently-augmented views of every image
(via `SimCLRDataset`), embeds both views, and minimizes the NT-Xent
contrastive loss (`_nt_xent_loss`) between them. As with `VAE`, there's no
validation split, so early stopping watches a smoothed moving average of
the epoch's **training loss** (`mode="min"`) and restores the best-epoch
weights once it stops improving — see
[`EarlyStopping`](#earlystopping--shared-early-stopping-logic).
`view_transform`
defaults to `build_augmentation_transform()` (the same recipe the
`Augmentor` preprocessing step uses) plus `ToTensor()`, but you can pass
in an existing, already-tuned `Augmentor`'s `.transform` instead of
letting SimCLR define a second, separately-configured augmentation setup:

```python
from pipeline.preprocessing.augmentation import Augmentor
from torchvision import transforms

# Built only for its config/.transform - never call .run() on it.
augmentor = Augmentor(input_path=folder, crop_scale=(0.7, 1.0))
view_transform = transforms.Compose([augmentor.transform, transforms.ToTensor()])

simclr.run("train_model", view_transform=view_transform)
```

`_nt_xent_loss(z1, z2, temperature)`: for a batch of N images (2N views
total), the only positive for a view is the other view of the same
source image; every other one of the remaining 2N-2 views is a negative.
Implemented as a single `cross_entropy` over the `(2N, 2N)` cosine
similarity matrix, with the diagonal (self-similarity) masked out and the
target class for each row set to the index of its positive partner.

### `extract_features(batch_size=32) -> pd.DataFrame`

Same output shape as `VAE.extract_features`: a `filename` + `features`
(the encoder's `h`) row per image, written to
`<output_path>/features.csv` - drop-in compatible with the same
downstream clustering/evaluation code, just point it at a different CSV.

### `SimCLRDataset(paths, transform)`

Yields two independently-augmented views of each image per `__getitem__`
call (freshly randomized every time, so every training epoch sees
different crops/jitter/blur) - the positive pair the contrastive loss is
trained on. Has no labels, unlike `ImageDataset`.

## `BYOL` — self-supervised feature learning without negative pairs

A third unsupervised alternative (Grill et al., "Bootstrap Your Own
Latent", 2020): like `SimCLR`, it relates two augmented views of the same
image, but it never contrasts against *other* images at all - no
negative pairs, and therefore no dependence on batch size for the loss to
have "enough negatives" (a real limitation of `SimCLR` on a dataset this
size; see `SimCLR.run`'s docstring).

Instead, BYOL keeps two copies of the network:
- the **online** network (encoder + projector + predictor) - trained by
  gradient descent, as usual.
- the **target** network (encoder + projector, no predictor) - *never*
  trained by gradient descent; its weights are an exponential moving
  average (EMA) of the online network's.

The online network's *prediction* of one view's projection is trained to
match the *target* network's actual projection of the other view
(stop-gradient on the target side), and vice versa - no labels, no
negatives.

Reuses `SimCLREncoder` and `SimCLRDataset` from `simclr.py` rather than
duplicating them. If anything, BYOL depends on `SimCLREncoder`'s
BatchNorm even more than SimCLR does: without negative pairs at all,
BatchNorm's cross-batch normalization is the main mechanism left
preventing the online/target networks from collapsing onto "predict the
same constant regardless of input" - a trivially perfect solution to a
loss with no negatives to push against. This is literally the failure
mode the BYOL paper studied when it investigated training without
BatchNorm.

```python
BYOL(
    input_path=None,   # folder of images (searched recursively); optional in in-memory mode (set_images)
    f=4, n=3, s=1, k=128,  # SimCLREncoder architecture - match your VAE's/SimCLR's values
    hidden_dim=256,          # projector/predictor MLP hidden size
    projection_dim=64,       # projector/predictor output size
    tau=0.99,                 # EMA momentum for the target network
    output_path=None,
    run_id="default_run",
)
```

### `run(mode="extract_features", *, epochs=100, lr=1e-3, batch_size=64, view_transform=None, early_stopping_patience=15, early_stopping_min_delta=0.0, early_stopping_smoothing_window=10)`

Same two-mode contract as `VAE.run`/`SimCLR.run`. Unlike `SimCLR`,
`batch_size` here doesn't need to be large - there's no negative-pair pool
to fill, so pick it based on memory/throughput rather than "enough
negatives". `view_transform` works exactly like `SimCLR.run`'s (defaults
to `build_augmentation_transform()`; pass an existing `Augmentor`'s
`.transform` to reuse its exact settings - see `SimCLR.train_model`'s
docstring for the pattern).

### `forward(view1, view2)`

Unlike `VAE`/`SimCLR`'s `forward` (which take one input and return an
embedding), `BYOL.forward` takes the *pair* of views and returns the
scalar training loss directly - its online-predicts-target computation
only makes sense given two views at once, so this avoids duplicating that
logic between `forward` and the training loop.

### `train_model(folder, epochs=100, lr=1e-3, batch_size=64, view_transform=None, early_stopping_patience=15, early_stopping_min_delta=0.0, early_stopping_smoothing_window=10)`

Optimizes only the **online** network's parameters (encoder + projector +
predictor) - the target network is never touched by the optimizer.
After every step, `_update_target_network()` pushes the target network's
weights toward the online network's current weights by `tau`:
`target = tau * target + (1 - tau) * online`. The target starts training
as an exact copy of the online network (see `__init__`).

No validation split here either, so early stopping watches a smoothed
moving average of the epoch's **training loss** (`mode="min"`) exactly
like `VAE`/`SimCLR`, restoring the best-epoch weights — online *and*
target networks both, since the snapshot is taken over the whole `BYOL`
module's `state_dict()` — once training stops. See
[`EarlyStopping`](#earlystopping--shared-early-stopping-logic).

### `extract_features(batch_size=32) -> pd.DataFrame`

Same output shape as `VAE.extract_features`/`SimCLR.extract_features`:
a `filename` + `features` row per image, taken from the **online
encoder's** embedding (never the target encoder's, and never the
projector/predictor outputs) - drop-in compatible with the same
downstream clustering/evaluation code.

### `_byol_loss(pred, target)`

`2 - 2 * cosine_similarity(pred, target)` between L2-normalized vectors -
0 when they point in exactly the same direction, up to 4 when opposite.
Computed in both directions (`forward` sums both) and averaged over the
batch to get the final scalar loss.

## `resize_with_padding(img, target_size=256, fill=255)`

Standalone helper (used directly in notebooks): resizes a PIL image to fit
within a `target_size` × `target_size` box (`thumbnail`, preserving aspect
ratio), then pads it to exactly that size, centered. Same goal as
`Normalizer.pad_to_square` (an aspect-ratio-preserving square canvas), but a
different implementation (resize-then-pad here, vs. pad-then-resize in
`Normalizer`) for use outside a full pipeline `Step`.

## Hand-crafted features (`handcrafted_features.py`)

Measures the vectorized profile drawings directly instead of learning
features. Input: a Vectorizer output folder (`_svg/*.svg` and
`extracted_metadata.csv`). Each drawing is moved into its vessel's
coordinate system in cm (x = distance from the rotation axis, y = distance
from the rim/base line, positive downwards), using the recorded axes and
`px_per_cm`; vertical-axis remnants are cut out of the lines first (the
horizontal line is kept: rim tops and bases lie on it).

Each `add_*(df, profiles)` function adds one kind of columns to the feature
DataFrame (one row per image); `extract_features(folder, groups, names,
efd_harmonics)` runs the chosen groups of `FEATURE_GROUPS` (position, size,
axis_distance, thickness, fourier):

| Function | Columns |
|---|---|
| `add_position_features` | `position` (below / above / both the horizontal line; ink within 1 mm of the line not counted), `below_frac`, `above_frac` |
| `add_size_features` | `rim_height_cm`, `rim_width_cm` |
| `add_axis_distance_features` | `rim_axis_dist_mean_cm`, `_min_cm`, `_max_cm` (from the rotation axis) |
| `add_thickness_features` | `rim_thickness_mean_cm`: normal rays from every line point to the nearest other line, so the lines need not be joined |
| `add_fourier_features` | `efd_<n>_<a,b,c,d>`, n = 1..10: elliptic Fourier descriptors of the cross-section's outline, size- and start-point-normalized, not rotation-normalized. The drawings leave the break edges out, so free line ends are joined pairwise (closest first, up to 3 cm) and the enclosed area is filled; if nothing is enclosed, the ink is closed morphologically wider than the wall instead |

Images without a coordinate system get NaN.

```bash
python -m pipeline.feature_learning.handcrafted_features <Vectorizer folder> [--out features.csv]
```

### `HandcraftedFeatures` (the pipeline step)

```python
HandcraftedFeatures(
    input_path=None,            # Vectorizer output folder, or a train/test subfolder of it
    features=("position", "size", "axis_distance", "thickness", "fourier"),
    efd_harmonics=10,
    output_path=None, name=None, run_id="default_run",
)
```

In configs: `model.type: handcrafted`, `model.params: {features: [...],
efd_harmonics: 10}`. Nothing is trained and there is no checkpoint;
`training` is ignored. `run()` accepts and ignores the training arguments,
so it stands in for a learned model, and writes to its folder:
`handcrafted_features.csv` (the readable table), `features.csv` (filename +
vector, read by clustering) and `feature_columns.txt`. The vector is every
numeric column except `above_frac` (= 1 - `below_frac`); missing values are
filled with the column median. Different units (cm, fractions, Fourier
coefficients) are why clustering should keep `standardize: true`.

It reads files, not images: the SVGs (`_svg/`) and `extracted_metadata.csv`
of a Vectorizer with `save_svg: true`, in `input_path` or its parent
(`split_by_labels.py` leaves them there). With `data.source: raw` in memory
mode, `Pipeline` points it at the saved last preprocessing folder
(`reads_saved_images`). Only the images in `input_path` are used, so a
train/ subfolder gives the train split.

## DINOv3 (`dinov3.py`)

A pretrained DINOv3 backbone (Hugging Face `transformers`) as a
feature extractor, as in the "DINOv3" section of `notebooks/sandbox.ipynb`:
each image is converted to RGB, prepared by the model's own
`AutoImageProcessor`, and its feature vector is the CLS token of the last
hidden state (768 numbers for ViT-B/16). By default (`finetune=False`)
the backbone is frozen and nothing is trained; it is a baseline for whether generic pretrained features already separate
the pottery types better than the purpose-built encoders.

```python
DINOv3(
    input_path=None,            # folder of preprocessed images (or set_images)
    model_name="facebook/dinov3-vitb16-pretrain-lvd1689m",   # or -vits16- / -vitl16-
    finetune=False,             # True: run(mode="train_model") fine-tunes first (skeleton)
    device=None,                # None = cuda if available
    output_path=None, name=None, run_id="default_run",
)
```

`run(mode="extract_features", batch_size=128)` accepts and ignores the
other training arguments, so it stands in for a learned model, and writes
`features.csv` (filename + vector) to its folder. It works in both disk
and memory mode (same features). `transformers` is imported, and the
weights loaded, only on the first run.

With `finetune=True`, `run(mode="train_model", epochs=20, lr=1e-5, ...)`
first calls `finetune_model` (AdamW on the unfrozen parameters,
`EarlyStopping` on the training loss), then extracts features with the
fine-tuned weights; with `finetune=False` that mode raises `ValueError`.
Fine-tuning is still a skeleton: `_setup_finetuning` (what to unfreeze,
head), `_train_loader` and `_finetune_loss` raise `NotImplementedError`,
saving `model.pt` is a TODO, and `run_experiment` does not train it yet
(`dinov3` is still in `MEASURED`).

In configs: `model.type: dinov3`, `model.params: {model_name: ...}`;
`training` is ignored except `batch_size`, and there is no checkpoint. The
run name shows the backbone (`dinov3-vitl16__...`) when `model_name` is
set. The checkpoints are gated: accept the license on the model page and
run `huggingface-cli login` (or set `HF_TOKEN`) once; afterwards the
weights come from the local cache.
