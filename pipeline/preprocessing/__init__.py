"""Concrete image preprocessing steps (background removal, skeleton
repair, size normalization) plus the shared image-discovery and
torch-Dataset helpers they - and the feature-learning models - all use."""

from pipeline.preprocessing.preprocessing_step import PreprocessingStep
from pipeline.preprocessing.background_eraser import BackgroundEraser
from pipeline.preprocessing.repair import Repair
from pipeline.preprocessing.normalizer import Normalizer
from pipeline.preprocessing.augmentation import Augmentor
from pipeline.preprocessing.vectorizer import Vectorizer
from pipeline.preprocessing.axis_renderer import AxisRenderer
from pipeline.preprocessing.dataset import (
    collect_image_paths,
    ImageDataset,
    DEFAULT_IMAGE_EXTENSIONS,
    ImageRecord,
    InMemoryImages,
    iter_image_batches,
    prefetch,
)

__all__ = [
    "PreprocessingStep",
    "BackgroundEraser",
    "Repair",
    "Normalizer",
    "Augmentor",
    "Vectorizer",
    "AxisRenderer",
    "collect_image_paths",
    "ImageDataset",
    "DEFAULT_IMAGE_EXTENSIONS",
    "ImageRecord",
    "InMemoryImages",
    "iter_image_batches",
    "prefetch",
]
