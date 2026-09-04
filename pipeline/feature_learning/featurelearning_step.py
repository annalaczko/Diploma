"""Abstract base for feature learning steps."""

from abc import abstractmethod
from typing import Any
import pandas as pd

import numpy as np

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

    @abstractmethod
    def run(self) -> pd.DataFrame:
        raise NotImplementedError

    @staticmethod
    def l2_normalize(vectors: np.ndarray) -> np.ndarray:
        """L2-normalize feature vectors row-wise (required before
        distance/cosine similarity comparisons)."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return vectors / norms
