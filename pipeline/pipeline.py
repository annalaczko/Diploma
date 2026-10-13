"""Top-level pipeline: chains preprocessing steps, then runs feature
learning and (optionally) clustering and evaluation - either through
intermediate folders on disk, or batch by batch in memory."""

import inspect
from contextlib import nullcontext
from pathlib import Path
from typing import Callable, Iterator, Literal

import numpy as np
import torch
from tqdm import tqdm

from pipeline import tracking
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
        save_preprocessed: Literal["last", "all"] | None = None,
        experiment_name: str | None = None,
        run_name: str | None = None,
        **feature_kwargs,
    ) -> dict:
        """Run the pipeline end to end on the raw images under `input_path`.

        With `experiment_name`, everything is tracked in one MLflow run (see
        `pipeline.tracking`): every step's configuration, the images each
        BackgroundEraser skipped, the feature-learning curves and
        features.csv, the clusters and the evaluation metrics. Without it,
        the same is logged into an MLflow run that is already active (e.g.
        `with tracking.start_run(...)`), or nothing at all.

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
            save_preprocessed: memory mode only - also write preprocessed
                images to disk, each into its step's `output_path` (the
                same folder disk mode would use): None (default) writes
                nothing, 'last' only the final images (after the last
                image-transforming step, i.e. what feature learning gets),
                'all' the output of every step. Disk mode always writes
                every step's output, so it ignores this.
            experiment_name: MLflow experiment to open a run in; None
                logs into the active run, if any.
            run_name: name of that MLflow run (MLflow picks one if None).
            **feature_kwargs: forwarded to the feature-learning step's
                `run()` (epochs, lr, batch_size, ...).

        Returns:
            A dict with 'images' (memory mode: the preprocessed
            `InMemoryImages`; with `save_preprocessed`, also
            'output_path', the folder of the final images) or
            'output_path' (disk mode: the last preprocessing output folder), plus 'features' (whatever the
            feature step's `run()` returned), 'labels' and 'metrics' when
            clustering/evaluation factories were given.
        """
        if mode not in ("memory", "disk"):
            raise ValueError("mode must be either 'memory' or 'disk'")
        if save_preprocessed not in (None, "last", "all"):
            raise ValueError("save_preprocessed must be None, 'last' or 'all'")
        run_context = (
            tracking.start_run(experiment_name, run_name=run_name)
            if experiment_name is not None
            else nullcontext()
        )
        with run_context:
            self._log_config(input_path, mode, preprocess_batch_size, feature_mode, feature_kwargs)
            tracking.log_params({"save_preprocessed": save_preprocessed})
            return self._run(input_path, mode, preprocess_batch_size, feature_mode, feature_kwargs,
                             save_preprocessed)

    def _run(self, input_path, mode, preprocess_batch_size, feature_mode, feature_kwargs,
             save_preprocessed=None) -> dict:
        if mode == "memory":
            images = self.preprocess_in_memory(input_path, preprocess_batch_size, save=save_preprocessed)
            result = {"images": images}
            steps = self._batch_steps()
            if save_preprocessed is not None and steps:  # nothing to save without steps
                result["output_path"] = Path(steps[-1].output_path)
                print(f"Preprocessed images saved under {result['output_path']}")
            if self.feature_learning_step is not None:
                self.feature_learning_step.set_images(images, train_transform=self._train_transform())
        elif mode == "disk":
            output_path = self.preprocess_on_disk(input_path)
            result = {"output_path": output_path}
            if self.feature_learning_step is not None:
                self.feature_learning_step.set_images(None)
                self.feature_learning_step.input_path = str(output_path)

        if self.feature_learning_step is None:
            return result

        result["features"] = self._run_feature_learning(feature_mode, feature_kwargs)

        if self.clustering_factory is not None:
            features_csv = Path(self.feature_learning_step.output_path) / "features.csv"
            result["labels"] = self.clustering_factory(str(features_csv)).run()
            if self.evaluation_factory is not None:
                result["metrics"] = self.evaluation_factory(result["labels"]).run()
        return result

    def _log_config(self, input_path, mode, preprocess_batch_size, feature_mode, feature_kwargs):
        """Log the run setup and every step's configuration (no-op without
        an active MLflow run). Steps log their own results as they run."""
        tracking.log_params({
            "input_path": str(input_path),
            "mode": mode,
            "preprocess_batch_size": preprocess_batch_size,
            "feature_mode": feature_mode,
            "preprocessing_steps": " -> ".join(step.name for step in self.preprocessing_steps),
        })
        # Same prefix the steps use themselves (e.g. BackgroundEraser.log_summary).
        for step in self.preprocessing_steps:
            tracking.log_params(tracking.params_of(step), prefix=f"preprocessing/{step.name}/")
        tracking.log_params(feature_kwargs, prefix="feature/train/")

    # ------------------------------------------------------------------ #
    # Preprocessing
    # ------------------------------------------------------------------ #

    def iter_batches(self, input_path, batch_size: int = 64, save=None) -> Iterator[list[ImageRecord]]:
        """Yield preprocessed batches: each batch of raw images from
        `input_path` passed through every step's `process_batch` in order
        (`Augmentor`s skipped, see the class docstring). Handy on its own,
        e.g. to look at a few preprocessed images in a notebook.

        Args:
            save: None, 'last' or 'all' - write the last step's / every
                step's output batch to that step's `output_path` (see
                `run`'s `save_preprocessed`).
        """
        steps = self._batch_steps()
        saving = {"last": steps[-1:], "all": steps}.get(save, [])
        n_images = len(collect_image_paths(input_path))
        with tqdm(total=n_images, desc="Preprocessing", unit="image") as progress:
            for batch in iter_image_batches(input_path, batch_size):
                n_loaded = len(batch)
                for step in steps:
                    batch = step.process_batch(batch)
                    if not batch:
                        break
                    if step in saving:
                        self._save_batch(batch, step.output_path)
                progress.update(n_loaded)
                if batch:
                    yield batch

    @staticmethod
    def _save_batch(records: list[ImageRecord], output_dir) -> None:
        """Write a batch under its names, like a step's disk-mode `run()`."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for record in records:
            record.image.save(output_dir / record.name)

    def preprocess_in_memory(self, input_path, batch_size: int = 64, save=None) -> InMemoryImages:
        """Run every batch through the preprocessing steps and keep only
        the final images, stacked into one `InMemoryImages` (optionally
        also written to disk, see `iter_batches`'s `save`)."""
        steps = self._batch_steps()
        for step in steps:
            step.begin_run()
        records = [record for batch in self.iter_batches(input_path, batch_size, save=save)
                   for record in batch]
        for step in steps:
            step.log_summary()
        images = InMemoryImages.from_records(records)
        print(f"Preprocessed {len(images)} images in memory, shape {images.images.shape}")
        tracking.log_metrics({"n_images": len(images)}, prefix="preprocessing/")
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
