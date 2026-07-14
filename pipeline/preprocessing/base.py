"""Abstract base for preprocessing steps."""

from abc import abstractmethod
from typing import Any

from pipeline.steps.base import Step


class PreprocessingStep(Step):
    """Common contract for all preprocessing steps
    (e.g. image enhancement, object detection).

    Preprocessing steps are usually chained one after another,
    so `run` should take an image (or batch) and return the
    same type, ready for the next step in the chain.
    """

    @abstractmethod
    def run(self, data: Any) -> Any:
        raise NotImplementedError
