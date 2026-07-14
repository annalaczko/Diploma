"""Abstract base for clustering steps."""

from abc import abstractmethod
from typing import Any

import numpy as np

from pipeline.steps.base import Step


class ClusteringStep(Step):
    """Common contract for clustering methods applied to fixed
    feature representations.

    `run` takes a feature matrix (n_samples, n_features) and
    returns cluster labels (n_samples,).
    """

    @abstractmethod
    def run(self, data: np.ndarray) -> np.ndarray:
        raise NotImplementedError
