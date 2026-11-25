"""Augmentor: random variations of the images (crop, brightness/contrast,
blur, mirror), so a model trained on few images overfits less.

In disk mode it writes the variations as extra files; in the in-memory mode
the Pipeline applies its `transform` on the fly during training instead.
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter, ImageOps
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
    """The random transform PIL image -> PIL image: random crop resized to
    `output_size`, brightness/contrast jitter, sometimes a blur, and
    sometimes a left-right mirror. Also used by SimCLR/BYOL to make their
    two random views of an image. The arguments are explained in
    `Augmentor`.
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


class ThickenLines:
    """Spreads the dark lines of a white-background drawing with a minimum
    filter of `size` pixels.

    A class rather than a lambda: DataLoader workers get the transform
    pickled (Python 3.14 starts them with 'forkserver'), and a lambda
    defined inside a function cannot be pickled.
    """

    def __init__(self, size=3):
        self.size = size

    def __call__(self, image):
        return image.filter(ImageFilter.MinFilter(self.size))


class EraseLineSegments:
    """Blanks out `min_patches`..`max_patches` small squares centred on line
    pixels, so pieces of the drawing go missing, as in a broken sherd or an
    interrupted line. Works on the INVERTED image (lines bright, background
    0), as used inside `build_line_drawing_augmentation`.

    The centres are picked among the line pixels (brighter than 127), so
    every patch removes some of the drawing, not just empty background. A
    patch's side is `min_size`..`max_size` times the drawing's extent (the
    longer side of its bounding box), so small drawings lose small pieces.
    Randomness comes from torch, which DataLoader workers seed separately
    (numpy's global generator would repeat the same patches in every worker).
    """

    def __init__(self, min_patches=1, max_patches=3, min_size=0.05, max_size=0.15):
        self.min_patches = min_patches
        self.max_patches = max_patches
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, image):
        pixels = torch.from_numpy(np.array(image))
        line = (pixels > 127).nonzero()  # (row, col) of the line pixels
        if len(line) == 0:
            return image
        side = int((line.max(dim=0).values - line.min(dim=0).values).max()) + 1  # the drawing's extent
        num_patches = int(torch.randint(self.min_patches, self.max_patches + 1, ()))
        for _ in range(num_patches):
            row, col = line[int(torch.randint(len(line), ()))].tolist()
            half = int(side * (self.min_size + (self.max_size - self.min_size) * float(torch.rand(()))) / 2)
            pixels[max(row - half, 0):row + half + 1, max(col - half, 0):col + half + 1] = 0
        return Image.fromarray(pixels.numpy())


class InkedRandomResizedCrop:
    """A random crop of a white-background drawing, resized to `size`,
    that shows some of the drawing: the "local crops" of DINO's multi-crop.

    Like `transforms.RandomResizedCrop` (the crop covers `scale` of the
    image's area, with an aspect ratio within `ratio`), but a crop is
    redrawn, up to `max_tries` times, while less than `min_ink` of its
    pixels are dark (below 128). A line drawing is mostly white paper, so
    plain random crops would often be blank, and a student matching the
    teacher from a blank crop learns nothing about the sherd. When no try
    has enough ink, the inkiest one is used. `min_ink=0` accepts the first
    crop, as `RandomResizedCrop` does.

    Randomness comes from torch (`RandomResizedCrop.get_params`), which
    DataLoader workers seed separately.
    """

    def __init__(self, size, scale=(0.05, 0.32), ratio=(3 / 4, 4 / 3), min_ink=0.005, max_tries=10):
        self.size = size
        self.scale = tuple(scale)
        self.ratio = tuple(ratio)
        self.min_ink = min_ink
        self.max_tries = max_tries

    def __call__(self, image):
        dark = np.asarray(image.convert("L")) < 128
        best, best_ink = None, -1.0
        for _ in range(self.max_tries):
            top, left, height, width = transforms.RandomResizedCrop.get_params(image, self.scale, self.ratio)
            ink = float(dark[top:top + height, left:left + width].mean())
            if ink > best_ink:
                best, best_ink = (top, left, height, width), ink
            if ink >= self.min_ink:
                break
        top, left, height, width = best
        return transforms.functional.resized_crop(
            image, top, left, height, width, [self.size, self.size],
            interpolation=transforms.InterpolationMode.BILINEAR)


def build_line_drawing_augmentation(
    output_size=256,
    rotation_degrees=5.0,
    translate=0.08,
    scale=(0.9, 1.1),
    elastic_alpha=50.0,
    elastic_sigma=6.0,
    elastic_probability=0.5,
    thicken_probability=0.5,
    thicken_size=3,
    erase_probability=0.5,
):
    """A random transform PIL image -> PIL image for dark line drawings on a
    white background, made for self-supervised fine-tuning (DINOv3).

    Unlike `build_augmentation_transform` it doesn't crop (the size of a
    sherd can tell its type, so the scale changes only a little) and doesn't
    mirror (the drawings follow a fixed left/right convention). It varies
    what differs between drawings of the same type, and moves the drawing
    enough that its exact position and size can't be used to match the two
    views:
      - an affine transform: a rotation of at most `rotation_degrees`
        either way, a shift of at most `translate` times the image size in
        each direction, and a resize by a factor within `scale`;
      - sometimes (`elastic_probability`) an elastic distortion, as if
        drawn by another hand: pixels moved by a smoothed random field,
        `elastic_alpha` its strength, `elastic_sigma` its smoothness
        (both in pixels);
      - sometimes (`thicken_probability`) thicker lines: a minimum filter
        of `thicken_size` pixels spreads the dark lines. There is no
        thinning, it would erase the 1-2 pixel lines;
      - sometimes (`erase_probability`) missing pieces of the lines
        (`EraseLineSegments`), as in a broken sherd.
    The emptied corners stay white, the background colour: the geometric
    steps run on the inverted image (black background, fill 0), because
    torchvision blends a non-zero fill in with a mask, which leaves grey
    lines along the transformed image's border. The result is resized to
    `output_size`.
    """
    return transforms.Compose([
        transforms.RandomApply([ThickenLines(thicken_size)], p=thicken_probability),
        transforms.Lambda(ImageOps.invert),
        transforms.RandomAffine(rotation_degrees, translate=(translate, translate), scale=scale,
                                interpolation=transforms.InterpolationMode.BILINEAR, fill=0),
        transforms.RandomApply([transforms.ElasticTransform(alpha=elastic_alpha, sigma=elastic_sigma, fill=0)],
                               p=elastic_probability),
        transforms.RandomApply([EraseLineSegments()], p=erase_probability),
        transforms.Lambda(ImageOps.invert),
        transforms.Resize((output_size, output_size)),
    ])


class Augmentor(PreprocessingStep):
    """Generates randomly augmented copies of each input image: random
    cropping+resizing, brightness/contrast jitter, occasional Gaussian
    blur, and a random horizontal flip.

    This is meant to sit right before a training step (VAE / VAEClassifier)
    that would otherwise overfit on a small labelled dataset.

    In a `Pipeline` run with mode="memory", this step doesn't write
    copies at all: its `transform` is applied on the fly to the training
    images instead, so every epoch sees a fresh random view of each image
    (`num_augmentations` and `keep_original` then have no effect).

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
        input_path=None,
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
