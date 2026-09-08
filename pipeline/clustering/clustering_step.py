"""Abstract base for clustering steps."""

from abc import abstractmethod
from typing import Any

import numpy as np

from pipeline.steps.step import Step


class ClusteringStep(Step):
    """Common contract for clustering methods applied to fixed
    feature representations.

    Unlike the base `Step.run(data)`, concrete clustering steps take no
    `run()` argument - they read their input (a feature matrix, or the
    CSV path it's stored at) from config passed to `__init__` instead,
    and `run()` returns cluster labels (n_samples,).
    """

    def __init__(
        self,
        input_path: str,
        output_path: str | None = None,
        name: str | None = None,
        delete_input_data: bool = False,
        save_results: bool = False,
        run_id: str = "default_run",
    ):
        """See `Step.__init__` for what each argument controls; `input_path`
        here is expected to be a CSV of precomputed feature vectors."""
        super().__init__(
            input_path=input_path,
            output_path=output_path,
            name=name,
            delete_input_data=delete_input_data,
            save_results=save_results,
            run_id=run_id,
        )

    @abstractmethod
    def run(self) -> dict[str, Any]:
        raise NotImplementedError
