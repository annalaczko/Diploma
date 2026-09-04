# `pipeline.feature_learning`

Turns preprocessed pottery-sherd images into fixed-length feature vectors
(unsupervised, `VAE`) or predicts a pottery type label directly
(supervised, `VAEClassifier`, built on top of a trained `VAE`'s encoder) —
letting the two approaches be compared directly.

```
pipeline/feature_learning/
├── featurelearning_step.py   # FeatureLearningStep: abstract base
└── vae.py                     # VAE, VAEClassifier, Encoder, Decoder, resize_with_padding
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

## `VAE` — unsupervised feature learning

A convolutional Variational Autoencoder: trained to reconstruct grayscale
256×256 pottery-sherd images, then its encoder's latent means (`mu`) are
used as a fixed-length feature vector per image for downstream clustering.

```python
VAE(
    input_path: str,   # folder of images (searched recursively)
    f=4,    # base channel-width exponent; block i uses 2**(f+i) channels
    n=3,    # convolution kernel size
    s=1,    # base stride (see Encoder/Decoder below for the exact layout)
    k=128,  # latent dimensionality — the feature vector size
    output_path=None,
    run_id="default_run",
)
```

### `run(mode="extract_features", *, epochs=10, lr=1e-6, batch_size=32)`

- `mode="train_model"`: trains for `epochs` epochs, then extracts features
  from the trained model.
- `mode="extract_features"` (default): skips training, just runs
  inference — use this once you already have trained weights loaded (e.g.
  via `vae.load_state_dict(...)`).

### `forward(x)`

Encodes `x`, samples a latent `z` via the reparameterization trick (so
gradients can flow through the sampling step: `z = mu + eps * std`, `eps ~
N(0, I)`), and decodes it back into a reconstruction. Returns
`(x_reconstructed, mu, sigma)` — `sigma` is a **log-variance**, not a raw
variance.

### `train_model(folder, epochs=10, lr=1e-6, batch_size=32)`

Standard VAE training loop, with one twist: the KL-divergence weight
(`beta`) is annealed on a **cyclical schedule** (`_cyclical_beta`, after Fu
et al., "Cyclical Annealing Schedule", 2019) instead of held constant. Beta
ramps linearly from 0 to 1 over the first half of each cycle, then holds at
1 for the rest of the cycle, repeating for 4 cycles across training. This
avoids *posterior collapse* (the decoder learning to ignore the latent code
entirely, which a constant high beta from epoch 1 tends to cause) while
still reaching full KL regularization within each cycle.

Loss (`_loss_function`, a standard beta-VAE loss):

```
loss = BCE(reconstruction, input) + beta * KLD(mu, logvar)
```

`BCE` is a pixel-wise reconstruction error (pixels are in `[0, 1]`); `KLD`
pulls the latent distribution toward a standard normal. A higher `beta`
trades reconstruction quality for a more disentangled/regularized latent
space (generally better for downstream clustering).

### `extract_features(batch_size=32) -> pd.DataFrame`

Runs the encoder over every image in `self.input_path` in eval mode and
collects the latent **mean** (`mu`, not a sampled `z`, so features are
deterministic and reproducible) as each image's feature vector. Returns —
and also writes to `<output_path>/features.csv` — a DataFrame with one
`filename` and `features` row per image.

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
   evaluation once `run()` returns, without re-deriving the split.
4. **Optionally class-weighted loss** (`class_weighted_loss=True` by
   default): `sklearn.utils.class_weight.compute_class_weight("balanced",
   ...)` on the *train* split only (validation stays untouched, so
   `val_acc` reflects true, unweighted performance), so rare pottery types
   don't get drowned out by common ones in the loss.
5. **Standard train loop**, with **early stopping on a moving average**:
   each epoch's validation accuracy is smoothed over the last
   `early_stopping_smoothing_window` epochs before being compared to the
   running best, so a single noisy epoch can't trigger a stop while the
   underlying trend is still improving. Training stops after
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

## `resize_with_padding(img, target_size=256, fill=255)`

Standalone helper (used directly in notebooks): resizes a PIL image to fit
within a `target_size` × `target_size` box (`thumbnail`, preserving aspect
ratio), then pads it to exactly that size, centered. Same goal as
`Normalizer.pad_to_square` (an aspect-ratio-preserving square canvas), but a
different implementation (resize-then-pad here, vs. pad-then-resize in
`Normalizer`) for use outside a full pipeline `Step`.
