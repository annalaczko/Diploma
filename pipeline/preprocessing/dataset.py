"""Shared image-loading utilities used across the pipeline.

Two pieces, both centralizing logic that used to be copy-pasted (and
drifting inconsistently) across preprocessing steps and feature-learning
models:

    collect_image_paths - find image files under a path/list of paths.
    ImageDataset         - a torch Dataset that loads those files as
                            tensors, for both unsupervised (VAE) and
                            supervised (VAEClassifier) training.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms

# Used as the default set of file suffixes treated as images everywhere
# in the pipeline that doesn't have a reason to restrict this further.
DEFAULT_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def collect_image_paths(input_value, image_extensions=DEFAULT_IMAGE_EXTENSIONS) -> list[Path]:
    """Collect every existing image path under `input_value`.

    Args:
        input_value: a single path (str or Path), or a list/tuple of
            paths. Each entry may point at a file or a directory;
            directories are searched recursively (`Path.rglob("*")`).
            `None` entries and paths that don't exist on disk are
            silently skipped, so callers can pass optional or
            partially-missing input without pre-filtering it themselves.
        image_extensions: iterable of file suffixes (e.g. ".png") that
            count as images. Matching is case-insensitive.

    Returns:
        A de-duplicated list of `Path` objects pointing at existing image
        files, in the order they were discovered (directory contents are
        sorted before being added, so results are deterministic).
    """
    candidates = input_value if isinstance(input_value, (list, tuple)) else [input_value]
    collected: list[Path] = []
    seen: set[str] = set()

    for raw_path in candidates:
        if raw_path is None:
            continue

        path = Path(raw_path)
        if not path.exists():
            continue

        if path.is_file():
            if path.suffix.lower() in image_extensions:
                resolved = path.resolve()
                if str(resolved) not in seen:
                    seen.add(str(resolved))
                    collected.append(path)
            continue

        # Directory: recurse and keep only files with a matching extension.
        for nested in sorted(path.rglob("*")):
            if not nested.is_file():
                continue
            if nested.suffix.lower() not in image_extensions:
                continue
            resolved = nested.resolve()
            if str(resolved) not in seen:
                seen.add(str(resolved))
                collected.append(nested)

    return collected


def flatten_name(path, root) -> str:
    """Build a flat, collision-free filename for `path` that encodes its
    ancestor directory names relative to `root`, e.g.
    `"2012_Survey Gladiatorenschule__006.jpg"` for
    `<root>/2012_Survey Gladiatorenschule/006.jpg`.

    Input trees often reuse the same simple, sequential filenames
    (`001.jpg`, ...) in every subfolder, so saving (or naming) images by
    their bare basename would make them collide. Falls back to just the
    basename if `path` isn't under `root` (e.g. `root` is a list of paths
    or None).
    """
    path = Path(path)
    try:
        relative = path.resolve().relative_to(Path(root).resolve())
        ancestor_dirs = relative.parts[:-1]
    except (TypeError, ValueError):
        ancestor_dirs = ()
    return "__".join([*ancestor_dirs, path.name])


@dataclass
class ImageRecord:
    """One image travelling through the in-memory pipeline: the filename
    it would have had on disk (so downstream filename-based joins, e.g.
    labels by stem, keep working) and the image itself."""
    name: str
    image: Image.Image


def iter_image_batches(input_path, batch_size=64, image_extensions=DEFAULT_IMAGE_EXTENSIONS):
    """Yield the images under `input_path` as lists of `ImageRecord`s,
    `batch_size` at a time, so only one batch of full-resolution images
    is ever held in memory. Names are flattened relative to `input_path`
    (see `flatten_name`), matching what `BackgroundEraser` writes in disk
    mode, and EXIF orientation is applied, as `cv2.imread` does there.
    """
    paths = collect_image_paths(input_path, image_extensions)
    if not paths:
        raise FileNotFoundError(
            f"No existing images with extensions {image_extensions} found under {input_path}"
        )
    for start in range(0, len(paths), batch_size):
        batch = []
        for path in paths[start:start + batch_size]:
            with Image.open(path) as image:
                # Apply EXIF orientation, like cv2.imread does in
                # BackgroundEraser's disk mode - many scans are stored
                # rotated with an orientation tag.
                image = ImageOps.exif_transpose(image)
                image.load()
                batch.append(ImageRecord(flatten_name(path, input_path), image.copy()))
        yield batch


class InMemoryImages:
    """The final output of an in-memory preprocessing run: every image as
    one row of a single `(N, H, W)` uint8 grayscale array, plus its name.

    One big numpy array (instead of a list of PIL images) is what makes
    this cheap to share with DataLoader worker processes: after fork they
    read the same memory pages, whereas touching thousands of separate
    Python objects would copy them into every worker.
    """

    def __init__(self, names, images: np.ndarray):
        if len(names) != len(images):
            raise ValueError(f"{len(names)} names for {len(images)} images")
        self.names = list(names)
        self.images = images

    @classmethod
    def from_records(cls, records) -> "InMemoryImages":
        """Stack `ImageRecord`s into one array (converted to grayscale).
        All images must share one size - which they do after a `Normalizer`."""
        records = list(records)
        if not records:
            raise ValueError("No images to collect - did every image fail preprocessing?")
        arrays = [np.asarray(r.image.convert("L")) for r in records]
        shapes = {a.shape for a in arrays}
        if len(shapes) > 1:
            raise ValueError(
                f"Images have different sizes {sorted(shapes)}; add a Normalizer step "
                "so every image ends up the same size."
            )
        return cls([r.name for r in records], np.stack(arrays))

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx) -> Image.Image:
        return Image.fromarray(self.images[idx])

    def subset(self, indices) -> "InMemoryImages":
        """A new `InMemoryImages` holding only the given rows."""
        indices = list(indices)
        return InMemoryImages([self.names[i] for i in indices], self.images[indices])

    def save(self, output_dir) -> Path:
        """Write every image to `output_dir` under its name (for inspection)."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for name, array in zip(self.names, self.images):
            Image.fromarray(array).save(output_dir / name)
        return output_dir


def image_entries(images):
    """`(entries, names)` for either an `InMemoryImages` or a list of paths,
    where `entries[i]` can be passed to `open_image`."""
    if isinstance(images, InMemoryImages):
        return images, images.names
    paths = list(images)
    return paths, [Path(p).name for p in paths]


def open_image(entries, idx) -> Image.Image:
    """Load `entries[idx]` as a fully loaded PIL image, whether `entries`
    is a list of paths or an `InMemoryImages`."""
    entry = entries[idx]
    if isinstance(entry, Image.Image):
        return entry
    with Image.open(entry) as image:
        image.load()
        return image.copy()


class ImageDataset(Dataset):
    """General-purpose torch Dataset for the images used across this
    pipeline: loads each path with PIL and, unless disabled, converts it
    to single-channel grayscale and resizes it to a fixed square size.

    Optionally pairs each image with an integer label, which is what lets
    a single class serve both feature-learning models in this pipeline:
      - VAE (unsupervised): `ImageDataset(paths)`, no labels - iterating
        yields plain image tensors.
      - VAEClassifier (supervised): `ImageDataset(paths, labels=...)` -
        iterating yields (image, label) tuples.
    Previously these were two separate, near-duplicate Dataset classes
    that had drifted out of sync (one resized/grayscaled, the other
    didn't, and silently relied on its input already being the right
    shape) - this replaces both.
    """

    def __init__(self, paths, labels=None, grayscale=True, target_size=256, transform=None):
        """
        Args:
            paths: sequence of image file paths, or an `InMemoryImages`
                (in-memory pipeline mode).
            labels: optional sequence of integer class ids, aligned with
                `paths`. If given, `__getitem__` returns (image, label)
                tuples; if omitted, it returns just the image tensor.
            grayscale: if True (default), convert every image to
                single-channel grayscale before returning it - matches
                what the VAE encoder expects.
            target_size: if given (default 256), resize every image that
                isn't already exactly (target_size, target_size) - most
                images here already are, having gone through a Normalizer
                step, so this is usually a cheap no-op rather than a real
                resize. Pass None to skip resizing entirely and keep each
                image's original size - only safe if every image already
                shares one size, since a DataLoader can't batch mismatched
                shapes.
            transform: optional PIL -> PIL transform applied on every
                fetch, after grayscale/resize and before conversion to a
                tensor - e.g. `Augmentor.transform` for on-the-fly
                augmentation (a different random view every epoch).
        """
        self.paths, self.names = image_entries(paths)
        self.transform = transform
        self.labels = torch.tensor(labels, dtype=torch.long) if labels is not None else None
        self.grayscale = grayscale
        self.target_size = target_size

        pre_resize_steps = []
        if grayscale:
            pre_resize_steps.append(transforms.Grayscale(num_output_channels=1))
        self.pre_resize = transforms.Compose(pre_resize_steps) if pre_resize_steps else None
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = open_image(self.paths, idx)
        if self.pre_resize is not None:
            image = self.pre_resize(image)

        # Only actually resize if needed: most images here already went
        # through a Normalizer step and are already target_size x
        # target_size, so unconditionally resizing every image on every
        # __getitem__ call (i.e. every epoch) would repeat an expensive,
        # no-op interpolation millions of times over a long training run.
        if self.target_size is not None and image.size != (self.target_size, self.target_size):
            image = image.resize((self.target_size, self.target_size), Image.Resampling.LANCZOS)

        if self.transform is not None:
            image = self.transform(image)

        image = self.to_tensor(image)

        if self.labels is not None:
            return image, self.labels[idx]
        return image
