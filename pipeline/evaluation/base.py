"""Abstract base for evaluation steps."""

from abc import abstractmethod
from typing import Any, Dict

from pipeline.steps.base import Step


class EvaluationStep(Step):
    """Common contract for evaluation methods
    (ARI, NMI, F-measure, precision@k, nearest-neighbour retrieval, silhouette).

    `run` takes predicted labels (and ground truth / features as needed)
    and returns a dict of metric name -> value.
    """

    @abstractmethod
    def run(self, data: Any) -> Dict[str, float]:
        raise NotImplementedError
