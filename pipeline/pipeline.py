"""Top-level pipeline: chains preprocessing steps, then runs feature
learning and (optionally) clustering and evaluation - either through
intermediate folders on disk, or batch by batch in memory."""

import inspect
from pathlib import Path
from typing import Callable, Iterator, Literal

import numpy as np
import torch
from tqdm import tqdm

from pipeline.clustering.clustering_step import ClusteringStep
from pipeline.evaluation.evaluation_step import EvaluationStep
from pipeline.feature_learning.featurelearning_step import FeatureLearningStep
from pipeline.preprocessing.augmentation import Augmentor
from pipeline.preprocessing.dataset import (
    ImageRecord,
    InMemoryImages,
    collect_image_paths,
    iter_image_batches,
)
from pipeline.preprocessing.preprocessing_step import PreprocessingStep


class Pipeline:
    """Wires together one full experiment run: a list of preprocessing
    steps applied in order, a feature-learning step, and optional
    clustering/evaluation.

    Two ways to run the same configured steps (see `run`):

      - mode="disk": exactly the step-by-step workflow, just chained -
        every step's `run()` writes its output folder, which becomes the
        next step's `input_path`.
      - mode="memory": images are loaded `preprocess_batch_size` at a time and passed
        through every step's `process_batch()`; no intermediate folders are
        written. Only the final (small, normalized) images are kept, in one
        `InMemoryImages` array, which the feature-learning step then trains
        on / extracts features from. An `Augmentor` in the list is not run
        as a step here: its transform is applied on the fly to training
        images instead (a fresh random view each epoch).

    This class owns no image-processing logic of its own - all behavior
    lives in the step objects passed in, which keeps every stage swappable.
    """

    def __init__(
        self,
        preprocessing_steps: list[PreprocessingStep],
        feature_learning_step: FeatureLearningStep | None = None,
        clustering_factory: Callable[[str], ClusteringStep] | None = None,
        evaluation_factory: Callable[[np.ndarray], EvaluationStep] | None = None,
    ):
        """
        Args:
            preprocessing_steps: steps run in order. Their `input_path`s
                don't need to be set: `run` wires them together.
            feature_learning_step: e.g. a `VAE` (already moved to the
                right device / with weights loaded). If None, `run` stops
                after preprocessing.
            clustering_factory: builds the clustering step from the path of
                the `features.csv` the feature-learning step wrote, e.g.
                `lambda csv: HierarchicalClustering(input_path=csv, cluster_num=55)`.
                A factory rather than a step, since clustering steps read
                their CSV in `__init__`, i.e. it must exist first.
            evaluation_factory: builds the evaluation step from the
                predicted cluster labels, e.g.
                `lambda y_pred: EvaluationWithGroundTruth(y_true, y_pred)`.
        """
        self.preprocessing_steps = preprocessing_steps
        self.feature_learning_step = feature_learning_step
        self.clustering_factory = clustering_factory
        self.evaluation_factory = evaluation_factory

    def run(
        self,
        input_path,
        mode: Literal["memory", "disk"] = "memory",
        preprocess_batch_size: int = 64,
        feature_mode: str | None = "extract_features",
        **feature_kwargs,
    ) -> dict:
        """Run the pipeline end to end on the raw images under `input_path`.

        Args:
            input_path: raw image file/folder (or list of them) for the
                first preprocessing step.
            mode: 'memory' (batches in memory, no intermediate folders) or
                'disk' (each step writes its output folder).
            preprocess_batch_size: images loaded and preprocessed at once
                in memory mode (YOLO's own inference batch is the
                BackgroundEraser's `batch_size`). Unused in disk mode. Named
                apart from `batch_size`, which goes to the feature step.
            feature_mode: passed as the feature-learning step's `mode`
                ('train_model' or 'extract_features'); ignored for steps
                whose `run()` has no `mode` (e.g. `VAEClassifier`).
            **feature_kwargs: forwarded to the feature-learning step's
                `run()` (epochs, lr, batch_size, ...).

        Returns:
            A dict with 'images' (memory mode: the preprocessed
            `InMemoryImages`) or 'output_path' (disk mode: the last
            preprocessing output folder), plus 'features' (whatever the
            feature step's `run()` returned), 'labels' and 'metrics' when
            clustering/evaluation factories were given.
        """
        if mode == "memory":
            images = self.preprocess_in_memory(input_path, preprocess_batch_size)
            result = {"images": images}
            if self.feature_learning_step is not None:
                self.feature_learning_step.set_images(images, train_transform=self._train_transform())
        elif mode == "disk":
            output_path = self.preprocess_on_disk(input_path)
            result = {"output_path": output_path}
            if self.feature_learning_step is not None:
                self.feature_learning_step.set_images(None)
                self.feature_learning_step.input_path = str(output_path)
        else:
            raise ValueError("mode must be either 'memory' or 'disk'")

        if self.feature_learning_step is None:
            return result

        result["features"] = self._run_feature_learning(feature_mode, feature_kwargs)

        if self.clustering_factory is not None:
            features_csv = Path(self.feature_learning_step.output_path) / "features.csv"
            result["labels"] = self.clustering_factory(str(features_csv)).run()
            if self.evaluation_factory is not None:
                result["metrics"] = self.evaluation_factory(result["labels"]).run()
        return result

    # ------------------------------------------------------------------ #
    # Preprocessing
    # ------------------------------------------------------------------ #

    def iter_batches(self, input_path, batch_size: int = 64) -> Iterator[list[ImageRecord]]:
        """Yield preprocessed batches: each batch of raw images from
        `input_path` passed through every step's `process_batch` in order
        (`Augmentor`s skipped, see the class docstring). Handy on its own,
        e.g. to look at a few preprocessed images in a notebook."""
        steps = self._batch_steps()
        n_images = len(collect_image_paths(input_path))
        with tqdm(total=n_images, desc="Preprocessing", unit="image") as progress:
            for batch in iter_image_batches(input_path, batch_size):
                n_loaded = len(batch)
                for step in steps:
                    batch = step.process_batch(batch)
                    if not batch:
                        break
                progress.update(n_loaded)
                if batch:
                    yield batch

    def preprocess_in_memory(self, input_path, batch_size: int = 64) -> InMemoryImages:
        """Run every batch through the preprocessing steps and keep only
        the final images, stacked into one `InMemoryImages`."""
        records = [record for batch in self.iter_batches(input_path, batch_size) for record in batch]
        images = InMemoryImages.from_records(records)
        print(f"Preprocessed {len(images)} images in memory, shape {images.images.shape}")
        return images

    def preprocess_on_disk(self, input_path) -> Path:
        """Run each step's `run()` in order, pointing each step at the
        previous one's output folder. Returns the last output folder."""
        current = input_path
        for step in self.preprocessing_steps:
            step.input_path = current
            output = step.run()
            current = output if output is not None else step.output_path
        return Path(current)

    def _batch_steps(self) -> list[PreprocessingStep]:
        return [step for step in self.preprocessing_steps if not isinstance(step, Augmentor)]

    def _train_transform(self):
        """The (last) Augmentor's transform, to apply on the fly in memory
        mode, or None if there's no Augmentor in the pipeline."""
        augmentors = [step for step in self.preprocessing_steps if isinstance(step, Augmentor)]
        if not augmentors:
            return None
        augmentor = augmentors[-1]
        if augmentor.seed is not None:
            torch.manual_seed(augmentor.seed)
        return augmentor.transform

    # ------------------------------------------------------------------ #
    # Feature learning
    # ------------------------------------------------------------------ #

    def _run_feature_learning(self, feature_mode, feature_kwargs):
        step = self.feature_learning_step
        if feature_mode is not None and "mode" in inspect.signature(step.run).parameters:
            return step.run(feature_mode, **feature_kwargs)
        return step.run(**feature_kwargs)
