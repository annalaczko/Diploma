"""Data-augmentation preprocessing step: writes randomly perturbed copies
of each input image (crop+resize, color jitter, blur, flip) to grow a
small labelled dataset without collecting more real images.
"""

from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from pipeline.preprocessing.preprocessing_step import PreprocessingStep
from pipeline.preprocessing.dataset import collect_image_paths, DEFAULT_IMAGE_EXTENSIONS


def build_augmentation_transform(
    output_size=256,
    crop_scale=(0.6, 1.0),
    crop_ratio=(0.9, 1.1),
    brightness=0.2,
    contrast=0.2,
    blur_kernel_size=5,
    blur_sigma=(0.1, 2.0),
    blur_probability=0.5,
    horizontal_flip=True,
):
    """Build a random crop+resize / color-jitter / blur / flip pipeline.

    Factored out of `Augmentor` so the same augmentation recipe can be
    reused wherever on-the-fly random views of an image are needed - e.g.
    `pipeline.feature_learning.simclr.SimCLRDataset` generates its two
    positive-pair views this way, instead of duplicating the transform
    definition. See `Augmentor`'s docstring for what each argument means.
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(output_size, scale=crop_scale, ratio=crop_ratio),
        transforms.ColorJitter(brightness=brightness, contrast=contrast),
        transforms.RandomApply(
            [transforms.GaussianBlur(blur_kernel_size, sigma=blur_sigma)],
            p=blur_probability,
        ),
        transforms.RandomHorizontalFlip(p=0.5 if horizontal_flip else 0.0),
    ])


class Augmentor(PreprocessingStep):
    """Generates randomly augmented copies of each input image: random
    cropping+resizing, brightness/contrast jitter, occasional Gaussian
    blur, and a random horizontal flip.

    This is meant to sit right before a training step (VAE / VAEClassifier)
    that would otherwise overfit on a small labelled dataset.

    Config (all optional, sensible defaults provided):
        num_augmentations: number of randomly augmented copies to
            generate per input image.
        keep_original: if True, also copies each source image into
            output_path unchanged, alongside its augmented copies.
        output_size: side length (px) each augmented copy is resized to
            after cropping. The kept-original copy (if any) is left at
            its own size.
        crop_scale: (min, max) fraction of the original image area that
            the random crop covers before being resized to output_size -
            e.g. (0.8, 1.0) crops anywhere from 80% to 100% of the image.
        crop_ratio: (min, max) aspect ratio range for the random crop;
            kept close to 1.0 so sherds aren't visibly stretched.
        brightness / contrast: jitter strength passed to
            `torchvision.transforms.ColorJitter` (e.g. 0.2 means the
            factor is drawn from [0.8, 1.2]).
        blur_kernel_size: kernel size (must be odd) for the occasional
            Gaussian blur.
        blur_sigma: (min, max) blur standard deviation, randomized each
            time the blur is applied.
        blur_probability: chance that Gaussian blur is applied at all to
            a given augmented copy, in [0, 1].
        horizontal_flip: whether to randomly (p=0.5) mirror images
            left-right.
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
        output_size=256,
        crop_scale=(0.8, 1.0),
        crop_ratio=(0.9, 1.1),
        brightness=0.2,
        contrast=0.2,
        blur_kernel_size=5,
        blur_sigma=(0.1, 2.0),
        blur_probability=0.5,
        horizontal_flip=True,
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
        self.output_size = output_size
        self.crop_scale = crop_scale
        self.crop_ratio = crop_ratio
        self.brightness = brightness
        self.contrast = contrast
        self.blur_kernel_size = blur_kernel_size
        self.blur_sigma = blur_sigma
        self.blur_probability = blur_probability
        self.horizontal_flip = horizontal_flip
        self.seed = seed
        self.image_extensions = image_extensions

        self.transform = build_augmentation_transform(
            output_size=self.output_size,
            crop_scale=self.crop_scale,
            crop_ratio=self.crop_ratio,
            brightness=self.brightness,
            contrast=self.contrast,
            blur_kernel_size=self.blur_kernel_size,
            blur_sigma=self.blur_sigma,
            blur_probability=self.blur_probability,
            horizontal_flip=self.horizontal_flip,
        )

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

            for i in range(self.num_augmentations):
                augmented = self.transform(image)
                out_path = output_dir / f"{image_path.stem}_aug{i}{image_path.suffix}"
                augmented.save(out_path)

        if self.delete_input_data_flag:
            self.delete_input_data()

        return output_dir
