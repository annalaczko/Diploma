"""Shared image-loading utilities used across the pipeline.

Two pieces, both centralizing logic that used to be copy-pasted (and
drifting inconsistently) across preprocessing steps and feature-learning
models:

    collect_image_paths - find image files under a path/list of paths.
    ImageDataset         - a torch Dataset that loads those files as
                            tensors, for both unsupervised (VAE) and
                            supervised (VAEClassifier) training.
"""

from pathlib import Path

import torch
from PIL import Image
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

    def __init__(self, paths, labels=None, grayscale=True, target_size=256):
        """
        Args:
            paths: sequence of image file paths.
            labels: optional sequence of integer class ids, aligned with
                `paths`. If given, `__getitem__` returns (image, label)
                tuples; if omitted, it returns just the image tensor.
            grayscale: if True (default), convert every image to
                single-channel grayscale before returning it - matches
                what the VAE encoder expects.
            target_size: if given (default 256), resize every image to
                (target_size, target_size). Pass None to skip resizing
                and keep each image's original size - only safe if every
                image already shares one size (e.g. after a Normalizer
                step), since a DataLoader can't batch mismatched shapes.
        """
        self.paths = list(paths)
        self.labels = torch.tensor(labels, dtype=torch.long) if labels is not None else None

        transform_steps = []
        if grayscale:
            transform_steps.append(transforms.Grayscale(num_output_channels=1))
        if target_size is not None:
            transform_steps.append(transforms.Resize((target_size, target_size)))
        transform_steps.append(transforms.ToTensor())
        self.transform = transforms.Compose(transform_steps)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        with Image.open(self.paths[idx]) as image:
            image = self.transform(image)

        if self.labels is not None:
            return image, self.labels[idx]
        return image
