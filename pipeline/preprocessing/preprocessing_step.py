"""Abstract base for preprocessing steps."""

from abc import abstractmethod
from pathlib import Path
from typing import Any

from pipeline.steps.step import Step


class PreprocessingStep(Step):
    """Common contract for all preprocessing steps
    (e.g. image enhancement, object detection).

    Preprocessing steps are usually chained one after another,
    so `run` should take an image (or batch) and return the
    same type, ready for the next step in the chain.
    """

    def __init__(
        self,
        input_path: str,
        output_path: str | None = None,
        name: str | None = None,
        delete_input_data: bool = False,
        save_results: bool = True,
        run_id: str | None = None,
    ):
        # See Step.__init__ for what each argument controls (output path
        # resolution, whether to delete input data after the step runs, etc).
        super().__init__(
            input_path=input_path,
            output_path=output_path,
            name=name,
            delete_input_data=delete_input_data,
            save_results=save_results,
            run_id=run_id,
        )

    @abstractmethod
    def run(self) -> Path:
        raise NotImplementedError
