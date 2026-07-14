"""Abstract base for feature learning steps."""

from abc import abstractmethod
from typing import Any

import numpy as np

from pipeline.steps.base import Step


class FeatureLearningStep(Step):
    """Common contract for feature learning approaches
    (pretrained models, self-supervised retraining, VAE baseline).

    `run` takes preprocessed images and returns a fixed-length
    feature vector (or matrix of vectors) per input, so that all
    approaches are directly comparable downstream.
    """

    @abstractmethod
    def run(self, data: Any) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    def l2_normalize(vectors: np.ndarray) -> np.ndarray:
        """L2-normalize feature vectors row-wise (required before
        distance/cosine similarity comparisons)."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return vectors / norms
