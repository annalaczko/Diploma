"""Data-augmentation preprocessing step: writes randomly perturbed copies
of each input image (rotation, flips, mild scale/translation jitter) to
grow a small labelled dataset without collecting more real images.
"""

from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from pipeline.preprocessing.preprocessing_step import PreprocessingStep
from pipeline.preprocessing.dataset import collect_image_paths, DEFAULT_IMAGE_EXTENSIONS


class Augmentor(PreprocessingStep):
    """Generates randomly augmented copies of each input image: rotation,
    horizontal/vertical flips, and mild scale/translation jitter.

    Pottery sherds have no canonical orientation, so rotation and flips
    are safe (and useful) augmentations here, unlike for e.g. natural
    photographs where "up" usually matters. This is meant to sit right
    before a training step (VAE / VAEClassifier) that would otherwise
    overfit on a small labelled dataset.

    Config (all optional, sensible defaults provided):
        num_augmentations: number of randomly augmented copies to
            generate per input image.
        keep_original: if True, also copies each source image into
            output_path unchanged, alongside its augmented copies.
        rotation_degrees: images are rotated by a random angle in
            [-rotation_degrees, rotation_degrees].
        horizontal_flip / vertical_flip: whether to randomly (p=0.5)
            mirror images along that axis.
        scale_range: (min, max) random zoom factor applied per copy.
        translate_fraction: max random shift, as a fraction of image
            width/height, applied independently on each axis.
        fill_value: background fill (0-255) used for any pixels that
            rotation/translation/scaling expose outside the original
            image content; automatically expanded to match the image's
            mode (grayscale vs. RGB/RGBA).
        seed: if given, seeds torch's RNG once at the start of `run()`
            for reproducible augmentations.
        image_extensions: file suffixes treated as input images.
    """

    def __init__(
        self,
        input_path,
        output_path=None,
        name=None,
        delete_input_data=False,
        run_id="default_run",
        num_augmentations=1,
        keep_original=True,
        rotation_degrees=180,
        horizontal_flip=True,
        vertical_flip=True,
        scale_range=(0.9, 1.1),
        translate_fraction=0.05,
        fill_value=255,
        seed=None,
        image_extensions: tuple[str, ...] = DEFAULT_IMAGE_EXTENSIONS,
    ):
        super().__init__(
            input_path=input_path,
            output_path=output_path,
            name=name,
            delete_input_data=delete_input_data,
            run_id=run_id,
        )
        if num_augmentations < 0:
            raise ValueError(
                f"num_augmentations must be >= 0, got {num_augmentations}"
            )

        self.num_augmentations = num_augmentations
        self.keep_original = keep_original
        self.rotation_degrees = rotation_degrees
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.scale_range = scale_range
        self.translate_fraction = translate_fraction
        self.fill_value = fill_value
        self.seed = seed
        self.image_extensions = image_extensions

    def run(self) -> Path:
        """Generate `num_augmentations` randomly augmented copies of every
        image found under `input_path` (plus the untouched original if
        `keep_original`), saved under `output_path`.

        Output filenames are `<stem>_aug<i><suffix>` for augmented copies
        (i from 0 to num_augmentations - 1) and the original filename for
        the untouched copy, so multiple augmented copies of one source
        image never collide with each other or with other source images.
        """
        if self.seed is not None:
            torch.manual_seed(self.seed)

        output_dir = Path(self.output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        image_paths = collect_image_paths(self.input_path, self.image_extensions)
        if not image_paths:
            raise FileNotFoundError(
                f"No existing images with extensions {self.image_extensions} found under {self.input_path}"
            )

        for image_path in tqdm(image_paths, desc="Augmenting images", unit="image"):
            image = Image.open(image_path)

            if self.keep_original:
                image.save(output_dir / image_path.name)

            transform = self._build_transform(image.mode)
            for i in range(self.num_augmentations):
                augmented = transform(image)
                out_path = output_dir / f"{image_path.stem}_aug{i}{image_path.suffix}"
                augmented.save(out_path)

        if self.delete_input_data_flag:
            self.delete_input_data()

        return output_dir

    def _build_transform(self, mode: str):
        """Build a fresh random-augmentation pipeline, with its fill color
        matched to `mode` (a plain int for grayscale, an RGB(A) tuple
        otherwise) since torchvision requires the fill to match the
        image's channel count.
        """
        if mode == "L":
            fill = self.fill_value
        elif mode == "RGBA":
            fill = (self.fill_value, self.fill_value, self.fill_value, 255)
        else:
            fill = (self.fill_value, self.fill_value, self.fill_value)

        return transforms.Compose([
            transforms.RandomAffine(
                degrees=self.rotation_degrees,
                translate=(self.translate_fraction, self.translate_fraction),
                scale=self.scale_range,
                fill=fill,
            ),
            transforms.RandomHorizontalFlip(p=0.5 if self.horizontal_flip else 0.0),
            transforms.RandomVerticalFlip(p=0.5 if self.vertical_flip else 0.0),
        ])
