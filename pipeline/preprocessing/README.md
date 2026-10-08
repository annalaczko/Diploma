# `pipeline.preprocessing`

Concrete `PreprocessingStep` implementations that clean up raw pottery-sherd
photographs before feature learning, plus the shared image-loading helpers
they (and `pipeline.feature_learning`) all use.

```
pipeline/preprocessing/
├── preprocessing_step.py   # PreprocessingStep: abstract base
├── background_eraser.py    # BackgroundEraser
├── background_eraser2.py   # BackgroundEraser2: BackgroundEraser + keeps the coordinate system
├── axis_detection.py       # detect_axes: finds the drawing's coordinate system on a raw scan
├── repair.py                # Repair
├── normalizer.py            # Normalizer
├── augmentation.py           # Augmentor
├── dataset.py                # collect_image_paths, ImageDataset (shared, also used by feature_learning)
└── utils.py                   # convert_json_to_yolo: standalone LabelMe -> YOLO conversion script
```

A typical run chains the first three steps in this order (see the example
in [`pipeline/README.md`](../README.md)):

```
BackgroundEraser  ->  Normalizer  ->  Repair
```

`Augmentor` is optional and sits after that chain, right before a training
step (`VAE` / `VAEClassifier`) that would otherwise overfit on a small
labelled dataset - see its own section below.

## `PreprocessingStep`

```python
from pipeline.preprocessing import PreprocessingStep
```

Common contract for all preprocessing steps. Each step supports two modes
(see "Two ways to run the pipeline" in [`pipeline/README.md`](../README.md)):

- `run()` (disk): reads every image under `input_path`, writes results
  to `output_path`.
- `process_batch(records: list[ImageRecord]) -> list[ImageRecord]`
  (in-memory): processes one batch and returns it, keeping each record's
  `name`. Nothing is written except debug images. `BackgroundEraser` drops
  images with no detection. `Augmentor` has no `process_batch`: in memory
  mode `Pipeline` applies its `transform` on the fly during training instead.

`begin_run()` / `log_summary()`: hooks around one pass over the dataset.
`begin_run` resets per-run bookkeeping, and `log_summary` reports it to the
active MLflow run (see `pipeline/tracking.py`). Both do nothing by default.
In memory mode `Pipeline` calls them before and after preprocessing;
`BackgroundEraser.run()` calls them itself.

Both call the same per-image code (`Normalizer.normalize`,
`Repair.process_image`, `BackgroundEraser._clear_one`), so the two modes
produce the same images. See
[`pipeline/steps/README.md`](../steps/README.md) for the constructor
arguments shared with every other step.

## `BackgroundEraser`

Runs a YOLO model on each image, whites out everything outside the detected
shard, and (optionally) crops to a square canvas around it. Supports two
YOLO output types via `mode`:
- `'segmentation'`: the per-pixel mask, grown by `mask_margin` (see
  `_dilate_mask` below), is used for whiting-out, and the crop is centered
  on that grown mask.
- `'detection'` (the default): only a bounding box is available, so that
  box (padded by `box_margin`) stands in for the mask and centers the crop.
  See `_clear_images` below.

```python
BackgroundEraser(
    input_path=None,
    output_path=None,           # auto-derived from run_id when omitted (see Step)
    mode='detection',           # 'detection' or 'segmentation'
    model='/caa/Homes01/alaczko/Diploma/Diploma/models/yolov8n_seg_finetuned.pt',  # absolute path
    run_id="default_run",
    batch_size=256,              # model.predict's inference batch size
    box_margin=0.15,             # 'detection' only: see _expand_box below
    mask_margin=0.10,            # 'segmentation' only: see _dilate_mask below
    debug_images=10,             # debug overlays for the first N images; 0 disables
    image_extensions=(".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"),
    **kwargs,                   # e.g. device='cuda'/'cpu'
)
```

`batch_size` controls GPU memory use during detection - lower it if
`model.predict` raises CUDA out-of-memory errors; it doesn't affect how many
images are processed overall, just how many go through YOLO at once.

`debug_images`: the first `debug_images` processed images also get a debug
overlay (`_save_debug_image`), written to a sibling `<output folder>_debug`
directory. It is a sibling folder, not a subfolder, because later steps
collect images recursively and would otherwise pick the overlays up as
training data. The overlay draws on the original image:
- red: the raw YOLO box,
- yellow: the raw YOLO mask polygon (just the padded box in detection mode),
- green: the outline of the final (grown) mask; everything outside it was
  whited out,
- blue: the square crop window (it may run off the image, where the output
  is white-padded).

In-memory mode (`process_batch`) also writes debug overlays, counting across
batches, and loads the YOLO model only once.

`run(keep_results=True) -> str | None`

1. Collects input images (`collect_image_paths`, see below).
2. Symlinks them into a single flat temp directory (`_make_flat_symlink_dir`)
   — YOLO's `model.predict` walks a flat folder more predictably than an
   arbitrary directory tree.
3. Runs YOLO over that directory (`_detect_shards`, `batch=self.batch_size`),
   streaming results so large batches don't all sit in memory at once.
   Images with no detected mask (`mode='segmentation'`) or no detected box
   at all (`mode='detection'`), or that error out, are skipped and logged
   rather than failing the whole run.
4. For each detected image (`_clear_images`, which calls `_clear_one` per
   image): whites out every pixel outside the grown mask (or, in
   `mode='detection'`, outside the padded bounding box, see `_expand_box`
   below). If `crop=True` (the default when called from `run`), it also
   crops to a square canvas centered on that same region (the grown mask's
   bounding rectangle, or the padded box), white-padding wherever the
   square runs off the edge of the original image. The result is saved
   under a flattened, collision-free filename (`_flatten_output_name`, see
   below).

`_dilate_mask(mask, margin) -> ndarray` (`mode='segmentation'`): grows the
binary mask by a uniform border so it ends up about `margin` larger along
its longer side (`margin / 2` of that side added all around). The default
`mask_margin=0.10` adds a 5% border. It uses morphological dilation with a
disk-shaped kernel rather than scaling the polygon about its center, so a
concave or elongated sherd's thin edges get the same safety margin as its
bulk. `margin <= 0` leaves the mask unchanged.

`_expand_box(box, image_width, image_height, margin) -> ndarray`: pads a
`(x1, y1, x2, y2)` box outward by `margin` (a fraction of its own
width/height) on every side, then clamps to the image bounds. A raw YOLO
box is often drawn tight against the shard's visible edge, so the default
`margin=0.15` (15% of the box's own size per side, 30% wider/taller
overall) leaves a safety buffer so a thin protruding detail - a handle, a
spout - doesn't get clipped by whiting-out or cropping. Only used in
`mode='detection'`, where the padded box becomes both the whiting-out
polygon and the region the crop is centered on. In `mode='segmentation'`,
`mask_margin` plays this role instead.

`_flatten_output_name(path) -> str`: `output_dir` is a single flat
directory, but `input_path` is typically a tree of subfolders (e.g. one per
survey/album) that independently reuse the same simple, sequential
filenames (`001.jpg`, `002.jpg`, ...). Saving under the bare basename would
make images from different subfolders silently overwrite one another
whenever their names collide, so the actual saved filename instead joins
every ancestor directory name (relative to `input_path`) with the basename,
e.g. `2012_Survey Gladiatorenschule__006.jpg` for
`<input_path>/2012_Survey Gladiatorenschule/006.jpg`. Falls back to just the
basename if `path` isn't under `input_path` at all.

Returns `self.output_path` (or `None` if `keep_results=False`).

Skipped images: the flat names (`_flatten_output_name`) of images on which
YOLO found no shard are collected in `eraser.no_shard`, in both modes.
`log_summary()` logs `n_images` and `no_shard_count` as metrics and the names
as `preprocessing/<step name>/no_shard.txt` (one per line), together with
the step's settings as parameters (see `failure_lists()`).

`_extend_mask(img, item, mask, region) -> (mask, region)`: hook called by
`_clear_one` after the shard mask and the crop region are built, before
whiting-out and cropping. The default returns both unchanged; subclasses
(`BackgroundEraser2`) use it to keep more of the image.

## `BackgroundEraser2`

`BackgroundEraser` that also keeps the drawing's coordinate system: the long
vertical rotation axis and the horizontal rim/base line it meets. For each
image it runs `axis_detection.detect_axes` on the original scan, draws the
two detected segments into the shard mask, dilates them in every direction
by `axis_margin`, and crops around the shard and the axes together. The
output therefore shows the sherd with its coordinate system. Works in both
disk and in-memory mode (it only overrides `_extend_mask`).

```python
BackgroundEraser2(
    axis_margin=0.005,   # axis dilation, fraction of the image's shorter side (~1 mm on A4)
    **kwargs,            # everything BackgroundEraser takes (mode, model, mask_margin, ...)
)
```

When no coordinate system is found, the image is processed exactly like
`BackgroundEraser` would (shard only), and its flat name is added to
`eraser.no_axes`, which `log_summary()` logs as `no_axes_count` and
`preprocessing/BackgroundEraser2/no_axes.txt`. `eraser.axes`
maps each processed image name to its `AxisDetection` (origin, the two
segments in pixels, confidence), or `None` when nothing was found.

On sheets where a line runs through the origin (T or cross shaped axes, e.g.
the 1976 and 2012 series) the whole detected line is kept, so the crop also
covers the part of the axis on the far side of the profile.

### `axis_detection.detect_axes(gray) -> AxisDetection | None`

Finds the coordinate system on a grayscale A4 scan (ported from
`giappetto/pottery_axes_batch.py`). It works on a copy downscaled to a
1600 px long side:
1. Blackhat filtering enhances thin dark lines, then a deliberately low fixed
   threshold (`LINE_THRESHOLD`) keeps faint pencil lines (Otsu would be
   pulled up by the dark ink handwriting). The sheet border (punch holes,
   scanner edges) is blanked out.
2. Morphological openings with long horizontal / vertical kernels keep only
   straight ruler lines; each remaining component becomes a fitted segment.
3. Every vertical/horizontal pair is scored: long lines that meet win, gaps
   at the corner are penalized, and so are verticals with a lot of ink right
   beside them (`CLUTTER_WEIGHT`), since the rotation axis runs through
   empty paper while lines drawn along the profile do not.

Returns `None` when there is no pair of lines that meet.

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
6. **Invert** — stages 2-5 all work with the white-lines-on-black
   convention `threshold_and_dilate` establishes; `process_image` inverts
   the final result back to black lines on a white background before
   returning it, matching what the rest of the pipeline (and a human
   looking at the file) expects.

```python
Repair(
    input_path=None,     # not needed when Pipeline wires the steps up
    output_path=None,
    run_id="default_run",
    contrast_factor=10.0,
    dilate_kernel_size=3,
    dilate_iterations=3,
    max_distance=15,       # max px between two skeleton endpoints to join them
    line_thickness=1,
    min_component_size=0,  # connected components smaller than this are dropped as noise; 0 disables
    skeletonize=True,      # False skips steps 3-4: keep the dilated lines, only drop small components
    re_skeletonize=True,
    output_thickness=1,    # >1 dilates the final 1px skeleton to this width
    debug_images=10,       # debug strips for the first N images; 0 disables
    image_extensions=(".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"),
)
```

`run() -> Path` processes every image found under `input_path` and returns
the output directory. Each stage is also exposed as its own method
(`enhance_contrast`, `threshold_and_dilate`, `to_skeleton`,
`repair_skeleton`, `thicken`), so it can be reused or tested independently
via `process_image(img)`, which chains all of them on a single PIL image.
`process_image(img, return_stages=True)` returns `(image, stages)`, where
`stages` is an ordered `{label: image}` dict of every intermediate result.

`debug_images`: the first `debug_images` images also get a debug strip
(`_save_debug_image`) showing every stage side by side, each labelled. For
the steps that add or remove pixels, it shows an overlay
(`_diff_overlay`): black = in both, red = added, blue = removed. Strips go
to a sibling `<output folder>_debug` directory, for the same reason as
`BackgroundEraser`'s overlays.

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

## `Augmentor`

Writes randomly perturbed copies of each input image - random crop+resize,
brightness/contrast jitter, occasional Gaussian blur, and a random
horizontal flip - to grow a small labelled dataset without collecting more
real images.

```python
Augmentor(
    input_path,
    output_path=None,
    num_augmentations=1,        # augmented copies to generate per input image
    keep_original=True,         # also copy each source image unchanged
    output_size=256,            # augmented copies are cropped+resized to this size
    crop_scale=(0.8, 1.0),      # fraction of the image area the random crop covers
    crop_ratio=(0.9, 1.1),      # aspect ratio range for the random crop
    brightness=0.2,             # ColorJitter brightness strength
    contrast=0.2,                # ColorJitter contrast strength
    blur_kernel_size=5,
    blur_sigma=(0.1, 2.0),       # randomized blur strength when blur is applied
    blur_probability=0.5,       # chance the Gaussian blur is applied at all
    horizontal_flip=True,       # random (p=0.5) left-right mirror
    seed=None,                  # set for reproducible augmentations
    image_extensions=(".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"),
)
```

`run() -> Path`: for every collected input image, optionally copies it
unchanged (`keep_original`), then writes `num_augmentations` randomly
transformed copies named `<stem>_aug<i><suffix>`. Built on
`torchvision.transforms`: `RandomResizedCrop` (crop+resize),
`ColorJitter` (brightness/contrast), `GaussianBlur` wrapped in
`RandomApply` (blur, applied only `blur_probability` of the time), and
`RandomHorizontalFlip`. Since `RandomResizedCrop` always produces a square
`output_size x output_size` image, every augmented copy ends up that size
regardless of the source image's own dimensions - only the kept-original
copy (if any) is left at its original size.

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

### In-memory helpers

- `ImageRecord(name, image)`: one PIL image plus the filename it would have
  on disk. This is what `process_batch` takes and returns.
- `iter_image_batches(input_path, batch_size)`: yields lists of
  `ImageRecord`s, loaded from disk `batch_size` at a time. Names are
  flattened relative to `input_path` (`flatten_name`, the same naming
  `BackgroundEraser` uses on disk).
- `InMemoryImages(names, images)`: the final images as one `(N, H, W)`
  uint8 grayscale array. `from_records(records)` builds it (all images must
  share one size), `subset(indices)` selects rows, and `save(folder)` writes
  them out. A single array, rather than a list of PIL images, lets
  DataLoader workers share the memory after fork instead of copying it.

### `ImageDataset(paths, labels=None, grayscale=True, target_size=256, transform=None)`

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
- `paths` may also be an `InMemoryImages` (in-memory pipeline mode).
- `transform` (optional, PIL → PIL) runs on every fetch before the tensor
  conversion. This is how `Augmentor.transform` is applied on the fly to
  training images.

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
