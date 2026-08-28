"""Abstract base for clustering steps."""

from abc import abstractmethod
from typing import Any

import numpy as np

from pipeline.steps.step import Step


class ClusteringStep(Step):
    """Common contract for clustering methods applied to fixed
    feature representations.

    `run` takes a feature matrix (n_samples, n_features) and
    returns cluster labels (n_samples,).
    """

    def __init__(
        self,
        input_path: str,
        output_path: str | None = None,
        name: str | None = None,
        delete_input_data: bool = False,
        save_results: bool = False,
    ):
        super().__init__(
            input_path=input_path,
            output_path=output_path,
            name=name,
            delete_input_data=delete_input_data,
            save_results=save_results,
        )

    @abstractmethod
    def run(self) -> dict[str, Any]:
        raise NotImplementedError
