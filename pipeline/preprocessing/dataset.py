"""Finding and loading images: the helpers every step uses.

    collect_image_paths  find the image files under a folder (or list of paths)
    flatten_name         a unique flat file name for a file in a folder tree
    ImageRecord          one image plus its name, as passed between steps in memory
    iter_image_batches   load images batch by batch (in-memory mode)
    prefetch             load the next batch in the background meanwhile
    InMemoryImages       the final preprocessed images, as one numpy array
    ImageDataset         a torch Dataset that turns images into tensors for training
    TwoViewDataset       a torch Dataset that yields two random transformed views of each image
"""

import queue
import threading
from concurrent.futures import ThreadPoolExecutor
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
    """Every existing image file under `input_value`.

    Args:
        input_value: a path (str or Path) or a list/tuple of paths, each a
            file or a folder. Folders are searched recursively, except
            subfolders starting with "_" (a step's side outputs such as
            `_debug` or `_svg`, see `PreprocessingStep.side_output_dir`).
            `None` entries and missing paths are skipped.
        image_extensions: file suffixes (e.g. ".png") that count as
            images, case-insensitive.

    Returns:
        The image paths without duplicates, in a fixed order (folder
        contents sorted by name).
    """
    candidates = input_value if isinstance(input_value, (list, tuple)) else [input_value]
    collected: list[Path] = []
    seen: set[str] = set()

    def add(path: Path) -> None:
        # The same file can be reached twice (e.g. via two input paths).
        if path.suffix.lower() in image_extensions and str(path.resolve()) not in seen:
            seen.add(str(path.resolve()))
            collected.append(path)

    for raw_path in candidates:
        if raw_path is None or not Path(raw_path).exists():
            continue
        path = Path(raw_path)
        if path.is_file():
            add(path)
            continue
        for nested in sorted(path.rglob("*")):
            in_side_folder = any(part.startswith("_") for part in nested.relative_to(path).parts[:-1])
            if nested.is_file() and not in_side_folder:
                add(nested)

    return collected


def flatten_name(path, root) -> str:
    """A flat file name for `path` that includes its folders below `root`,
    joined with "__": `<root>/2012_Survey/006.jpg` -> `2012_Survey__006.jpg`.

    The raw scans reuse names like `001.jpg` in every subfolder, so the bare
    file name would not be unique. If `path` is not under `root` (or `root`
    is a list or None), the bare file name is used.
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
    """One image travelling through the in-memory pipeline, with the file
    name it would have on disk (labels are later looked up by that name)."""
    name: str
    image: Image.Image


def iter_image_batches(input_path, batch_size=64, image_extensions=DEFAULT_IMAGE_EXTENSIONS,
                       num_workers=8):
    """Yield the images under `input_path` as lists of `ImageRecord`s,
    `batch_size` at a time, so only one batch of full-size scans is in
    memory at once. The names are made with `flatten_name`, the same names
    `BackgroundEraser` writes in disk mode. The images of a batch are
    decoded on `num_workers` threads.
    """
    paths = collect_image_paths(input_path, image_extensions)
    if not paths:
        raise FileNotFoundError(
            f"No existing images with extensions {image_extensions} found under {input_path}"
        )

    def load(path):
        with Image.open(path) as image:
            # Many scans are stored rotated, with an EXIF orientation tag.
            # Apply it, as cv2.imread does in BackgroundEraser's disk mode.
            image = ImageOps.exif_transpose(image)
            image.load()
            return ImageRecord(flatten_name(path, input_path), image.copy())

    with ThreadPoolExecutor(max_workers=max(1, num_workers)) as pool:
        for start in range(0, len(paths), batch_size):
            yield list(pool.map(load, paths[start:start + batch_size]))


def prefetch(iterable, depth=1):
    """Iterate over `iterable` while a background thread already produces
    the next `depth` items, e.g. loads the next batch of images while the
    current one is processed. An error in the thread is raised here."""
    items = queue.Queue(maxsize=depth)
    stop = threading.Event()
    done = object()

    def produce():
        try:
            for item in iterable:
                while not stop.is_set():
                    try:
                        items.put((item, None), timeout=0.1)
                        break
                    except queue.Full:
                        pass
                if stop.is_set():
                    return
            items.put((done, None))
        except BaseException as error:  # handed over to the consumer
            items.put((done, error))

    thread = threading.Thread(target=produce, daemon=True)
    thread.start()
    try:
        while True:
            item, error = items.get()
            if error is not None:
                raise error
            if item is done:
                return
            yield item
    finally:
        stop.set()


class InMemoryImages:
    """The result of in-memory preprocessing: all images in one
    `(N, H, W)` uint8 grayscale array (`images`), plus their `names`.

    One big array instead of a list of PIL images is cheap to share with
    DataLoader worker processes (they read the same memory after fork).
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
    """Torch Dataset of images: each item is a (1, H, W) float tensor in
    [0, 1], grayscale and resized to `target_size` x `target_size`.

    Without labels it yields image tensors (unsupervised training, e.g.
    VAE); with `labels` it yields (image, label) pairs (VAEClassifier).
    """

    def __init__(self, paths, labels=None, grayscale=True, target_size=256, transform=None):
        """
        Args:
            paths: sequence of image file paths, or an `InMemoryImages`
                (in-memory pipeline mode).
            labels: optional sequence of integer class ids, aligned with
                `paths`. If given, `__getitem__` returns (image, label)
                tuples; if omitted, it returns just the image tensor.
            grayscale: convert to one-channel grayscale (what the encoders
                expect).
            target_size: resize images of another size to this square
                size. After a Normalizer they already have it, so usually
                nothing happens. None keeps every image's own size (they
                must then all be equal, or batching fails).
            transform: optional PIL -> PIL transform applied on every
                fetch, after grayscale/resize, e.g. `Augmentor.transform`
                for a different random view every epoch.
        """
        self.paths, self.names = image_entries(paths)
        self.transform = transform
        self.labels = torch.tensor(labels, dtype=torch.long) if labels is not None else None
        self.grayscale = grayscale
        self.target_size = target_size

        self.pre_resize = transforms.Grayscale(num_output_channels=1) if grayscale else None
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = open_image(self.paths, idx)
        if self.pre_resize is not None:
            image = self.pre_resize(image)

        # Resize only if needed: after a Normalizer most images already have
        # the right size, and this runs for every image in every epoch.
        if self.target_size is not None and image.size != (self.target_size, self.target_size):
            image = image.resize((self.target_size, self.target_size), Image.Resampling.LANCZOS)

        if self.transform is not None:
            image = self.transform(image)

        image = self.to_tensor(image)

        if self.labels is not None:
            return image, self.labels[idx]
        return image


class TwoViewDataset(Dataset):
    """Yields two random views of each image (a "positive pair"). They are
    made anew on every fetch, so each epoch sees different variations."""

    def __init__(self, paths, transform):
        """
        Args:
            paths: sequence of image file paths, or an `InMemoryImages`.
            transform: a torchvision transform applied
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