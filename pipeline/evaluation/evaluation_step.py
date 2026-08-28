"""Abstract base for evaluation steps."""

from abc import abstractmethod
from typing import Any, Dict
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, accuracy_score

from pipeline.steps.step import Step
import pandas as pd


class EvaluationStep(Step):
    """Common contract for evaluation methods
    (ARI, NMI, F-measure, precision@k, nearest-neighbour retrieval, silhouette).

    `run` takes predicted labels (and ground truth / features as needed)
    and returns a dict of metric name -> value.
    """

    @abstractmethod
    def run(self) -> None:
        raise NotImplementedError


class EvaluationWithGroundTruth(EvaluationStep):

    def __init__(self, y_true, y_pred):
        self.ground_truth = y_true
        self.predicted = y_pred
        self.acc=0
        self.ari=0
        self.nmi=0

    def run(self) -> None:
        self.evaluate(self.ground_truth, self.predicted)

    def evaluate(self, y_true, y_pred) -> None:
        self.ari = adjusted_rand_score(y_true, y_pred)
        self.nmi = normalized_mutual_info_score(y_true, y_pred)
        self.acc = accuracy_score(y_true, y_pred)
        print(self)

    def __str__(self):
        return f"-----\nEvaluation results: \nACC = {self.acc}, \nARI = {self.ari}, \nNMI = {self.nmi}\n-----\n\n"
    