"""Abstract base for feature learning steps."""

from abc import abstractmethod
from typing import Any
import pandas as pd

import numpy as np

from pipeline import tracking
from pipeline.preprocessing.dataset import InMemoryImages, collect_image_paths
from pipeline.steps.step import Step


class FeatureLearningStep(Step):
    """Common contract for feature learning approaches
    (pretrained models, self-supervised retraining, VAE baseline).

    `run` takes preprocessed images and returns a fixed-length
    feature vector (or matrix of vectors) per input, so that all
    approaches are directly comparable downstream.
    """

    def __init__(
        self,
        input_path: str,
        output_path: str | None = None,
        name: str | None = None,
        delete_input_data: bool = False,
        save_results: bool = False,
        run_id: str | None = None,
    ):
        """See `Step.__init__` for what each argument controls."""
        super().__init__(
            input_path=input_path,
            output_path=output_path,
            name=name,
            delete_input_data=delete_input_data,
            save_results=save_results,
            run_id=run_id,
        )
        # In-memory pipeline mode (see `set_images`); None = read from input_path.
        self.images: InMemoryImages | None = None
        self.train_transform = None

    def set_images(self, images: InMemoryImages, train_transform=None) -> None:
        """Use already-preprocessed in-memory images instead of reading
        `input_path` from disk (see `pipeline.pipeline.Pipeline`).

        Args:
            images: the preprocessed images.
            train_transform: optional PIL -> PIL transform applied on the
                fly to training images only (e.g. `Augmentor.transform`),
                never to feature extraction or validation.
        """
        self.images = images
        self.train_transform = train_transform

    def _collect_inputs(self, folder=None):
        """The images this step works on: the in-memory ones if
        `set_images` was called, otherwise every image under `folder`
        (defaults to `input_path`)."""
        if self.images is not None:
            return self.images
        folder = folder or self.input_path
        if not folder:
            raise ValueError(
                f"{self.name} needs either input_path or in-memory images (set_images)"
            )
        return collect_image_paths(folder)

    @abstractmethod
    def run(self) -> pd.DataFrame:
        raise NotImplementedError

    # MLflow logging shared by the training loops (no-ops without an active
    # run, see pipeline.tracking). Metric names are prefixed 'feature/'.

    def _log_training_start(self, **train_args) -> None:
        """Log the model's configuration (its scalar attributes, e.g. f, n,
        s, k, recon_loss) and the training arguments (epochs, lr, ...)."""
        tracking.log_params({"feature_step": self.name})
        tracking.log_params(tracking.params_of(self), prefix=f"feature/{self.name}/")
        tracking.log_params(getattr(self, "hparams", {}), prefix=f"feature/{self.name}/")
        tracking.log_params(train_args, prefix="feature/train/")

    def _log_epoch(self, epoch: int, **metrics) -> None:
        """Log one epoch's metrics (loss, smoothed loss, lr, ...)."""
        tracking.log_metrics(metrics, step=epoch, prefix="feature/")

    def _log_training_end(self, early_stopping, last_epoch: int) -> None:
        """Log where training stopped and which epoch's weights were kept."""
        tracking.log_metrics({
            "stopped_epoch": last_epoch,
            "best_epoch": early_stopping.best_epoch,
            "best_smoothed": early_stopping.best_smoothed,
        }, prefix="feature/")

    def _log_features(self, features_csv) -> None:
        """Attach the extracted features.csv to the run."""
        tracking.log_artifact(features_csv, artifact_path="feature")

    @staticmethod
    def l2_normalize(vectors: np.ndarray) -> np.ndarray:
        """L2-normalize feature vectors row-wise (required before
        distance/cosine similarity comparisons)."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return vectors / norms
