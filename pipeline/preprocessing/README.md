# `pipeline.preprocessing`

Concrete `PreprocessingStep` implementations that clean up raw pottery-sherd
photographs before feature learning, plus the shared image-loading helpers
they (and `pipeline.feature_learning`) all use.

```
pipeline/preprocessing/
├── preprocessing_step.py   # PreprocessingStep: abstract base
├── background_eraser.py    # BackgroundEraser
├── repair.py                # Repair
├── normalizer.py            # Normalizer
├── dataset.py                # collect_image_paths, ImageDataset (shared, also used by feature_learning)
└── utils.py                   # convert_json_to_yolo: standalone LabelMe -> YOLO conversion script
```

A typical run chains these three steps in this order (see the example in
[`pipeline/README.md`](../README.md)):

```
BackgroundEraser  ->  Normalizer  ->  Repair
```

## `PreprocessingStep`

```python
from pipeline.preprocessing import PreprocessingStep
```

Common contract for all preprocessing steps. Preprocessing steps are usually
chained one after another, so `run()` should take an image (or batch) and
return the same type, ready for the next step in the chain. See
[`pipeline/steps/README.md`](../steps/README.md) for the constructor
arguments shared with every other step.

## `BackgroundEraser`

Runs a YOLO segmentation model on each image, whites out everything outside
the detected shard mask, and (optionally) crops to a square canvas centered
on the detection box.

```python
BackgroundEraser(
    input_path=None,
    mode='segmentation',        # only 'segmentation' is implemented
    model='.../yolov8n_seg_finetuned.pt',
    run_id="default_run",
    image_extensions=(".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"),
    **kwargs,                   # e.g. device='cuda'/'cpu'
)
```

`run(keep_results=True) -> str | None`

1. Collects input images (`collect_image_paths`, see below).
2. Symlinks them into a single flat temp directory (`_make_flat_symlink_dir`)
   — YOLO's `model.predict` walks a flat folder more predictably than an
   arbitrary directory tree.
3. Runs YOLO segmentation over that directory (`_detect_shards`), streaming
   results so large batches don't all sit in memory at once. Images with no
   detected mask, or that error out, are skipped and logged rather than
   failing the whole run.
4. For each detected image (`_clear_images`): whites out every pixel outside
   the mask; if `crop=True` (the default when called from `run`), also crops
   to a square canvas centered on the detection box, white-padding wherever
   that square runs off the edge of the original image.

Returns `self.output_path` (or `None` if `keep_results=False`).

## `Repair`

Cleans up and repairs broken hand-drawn/scanned pottery line drawings.
Reproduces a manual notebook workflow as a single pipeline step:

1. **Contrast enhancement** (`enhance_contrast`) — pushes the image toward
   pure black/white.
2. **Threshold + dilate** (`threshold_and_dilate`) — Otsu-binarizes (dark
   lines on light background → white lines on black background) and dilates
   to pre-close small gaps before skeletonization.
3. **Skeletonize** (`to_skeleton`) — thins the binary image to a 1px
   skeleton (`skimage.morphology.skeletonize`).
4. **Repair gaps** (`repair_skeleton`) — optional small-component noise
   removal, then finds skeleton endpoints and joins nearby pairs with a
   straight line, then (optionally) re-skeletonizes so the output stays a
   clean 1px skeleton instead of having slightly thicker joins.
5. **Thicken** (`thicken`, optional) — uniformly dilates the final skeleton
   to `output_thickness` px.

```python
Repair(
    input_path,
    output_path=None,
    contrast_factor=10.0,
    dilate_kernel_size=3,
    dilate_iterations=3,
    max_distance=15,       # max px between two skeleton endpoints to join them
    line_thickness=1,
    min_component_size=0,  # connected components smaller than this are dropped as noise; 0 disables
    re_skeletonize=True,
    output_thickness=1,    # >1 dilates the final 1px skeleton to this width
    image_extensions=(".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"),
)
```

`run() -> Path` processes every image found under `input_path` and returns
the output directory. Each stage is also exposed as its own method
(`enhance_contrast`, `threshold_and_dilate`, `to_skeleton`,
`repair_skeleton`, `thicken`), so it can be reused or tested independently
via `process_image(img)`, which chains all of them on a single PIL image.

## `Normalizer`

Pads images to a square canvas (preserving aspect ratio) and resizes them to
a fixed size, so every downstream step (feature learning, YOLO, ...) sees
consistently-shaped input.

```python
Normalizer(
    input_path=None,        # file, directory, or list/tuple of either
    output_path=None,
    target_size=256,
    background_color=(0, 0, 0),
    image_extensions=(".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"),
    rename=False,            # currently unused/reserved
)
```

`run() -> Path`: for each collected image, `pad_to_square` centers it on a
new `max(width, height)`-sized canvas filled with `background_color`
(preserving aspect ratio — unlike a plain resize to a square, which would
stretch the image), then resizes the padded square to `target_size` and
saves it under `output_path` with its original filename.

## `dataset.py` — shared image-loading helpers

These live in `preprocessing` because it's the one place in the codebase
that already owns all image-format/extension handling; they're imported
from `pipeline.feature_learning.vae` as well, so both preprocessing steps
and the feature-learning models go through the same code path.

### `collect_image_paths(input_value, image_extensions=DEFAULT_IMAGE_EXTENSIONS) -> list[Path]`

Collects every existing image path under `input_value`, which may be a
single file, a directory (searched recursively via `Path.rglob("*")`), or a
list/tuple mixing either. `None` entries and paths that don't exist are
silently skipped. Returns a de-duplicated, deterministically-ordered list.

`DEFAULT_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")`

This one function replaces what used to be three near-identical, drifting
copies of the same path-collection logic in `BackgroundEraser`, `Repair` and
`Normalizer`.

### `ImageDataset(paths, labels=None, grayscale=True, target_size=256)`

A general-purpose `torch.utils.data.Dataset` for loading images from disk as
tensors:

- `labels=None` → unsupervised use (`pipeline.feature_learning.VAE`):
  iterating yields plain image tensors.
- `labels=<sequence of ints>` → supervised use
  (`pipeline.feature_learning.VAEClassifier`): iterating yields
  `(image, label)` tuples.
- `grayscale=True` (default) converts every image to single-channel
  grayscale — what the VAE encoder expects.
- `target_size=256` (default) resizes every image to a
  `(target_size, target_size)` square; pass `None` to skip resizing, which
  is only safe if every image already shares one size (e.g. after a
  `Normalizer` step), since a `DataLoader` can't batch mismatched shapes.

This single class replaces what used to be two separate, near-duplicate
`Dataset` classes in `pipeline.feature_learning.vae` (one for the VAE, one
for the classifier) that had drifted out of sync with each other.

## `utils.py` — `convert_json_to_yolo`

A standalone one-off script (not a `Step`), unrelated to the classes above.
Converts LabelMe-style polygon annotation JSON files into YOLO segmentation
label `.txt` files. Run directly (`python -m pipeline.preprocessing.utils`)
against a folder of LabelMe exports. Two things worth knowing if you reuse
it: it hardcodes class id `1` for every polygon (this dataset only ever
labels one class), and it only converts the *first* shape in each JSON file
— multi-polygon annotations would lose everything past the first one.
