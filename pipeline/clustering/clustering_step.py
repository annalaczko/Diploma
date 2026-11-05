"""Abstract base class of the clustering steps."""

from abc import abstractmethod
from typing import Any

from pipeline.steps.step import Step


class ClusteringStep(Step):
    """Base class of clustering methods that group fixed feature vectors.

    A clustering step reads its features (a features.csv written by a
    feature-learning step) in `__init__`, and `run()`, which takes no
    argument, returns one cluster label per image.
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
        """`input_path` is the features CSV; the other arguments are
        explained in `Step.__init__`."""
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
