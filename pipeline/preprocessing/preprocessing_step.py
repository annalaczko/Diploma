"""Abstract base for preprocessing steps."""

from abc import abstractmethod
from pathlib import Path
from typing import Any

from pipeline.preprocessing.dataset import ImageRecord
from pipeline.steps.step import Step


class PreprocessingStep(Step):
    """Common contract for all preprocessing steps
    (e.g. image enhancement, object detection).

    Preprocessing steps are usually chained one after another, in one of
    two modes:
      - disk mode: `run()` reads every image under `self.input_path` and
        writes its results to `self.output_path`, which the next step
        then uses as its `input_path`.
      - in-memory mode: `process_batch(records)` takes a list of
        `ImageRecord`s and returns the processed list, nothing is written
        (see `pipeline.pipeline.Pipeline`).
    Both modes should call the same per-image code, so they can't drift apart.
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

    def process_batch(self, records: list[ImageRecord]) -> list[ImageRecord]:
        """In-memory counterpart of `run()`: process one batch of images
        and return the results (images that fail may be dropped)."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support in-memory (batch) mode"
        )

    def begin_run(self) -> None:
        """Reset per-run bookkeeping (e.g. lists of skipped images) before
        a pass over the dataset. `Pipeline` calls it before in-memory
        preprocessing; disk-mode `run()`s call it themselves."""

    def log_summary(self) -> None:
        """Report per-run bookkeeping to the active MLflow run (see
        `pipeline.tracking`) after a pass over the dataset. Counterpart of
        `begin_run`; does nothing by default."""
