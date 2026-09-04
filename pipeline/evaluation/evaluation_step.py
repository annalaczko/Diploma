"""Abstract base for evaluation steps."""

from abc import abstractmethod
from typing import Any, Dict
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

from pipeline.steps.step import Step
import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment


class EvaluationStep(Step):
    """Common contract for evaluation methods
    (ARI, NMI, F-measure, precision@k, nearest-neighbour retrieval, silhouette).

    `run` takes predicted labels (and ground truth / features as needed)
    and returns a dict of metric name -> value.
    """

    @abstractmethod
    def run(self) -> None:
        raise NotImplementedError

class EvaluationWithNoGroundTruth(EvaluationStep):
    """Unsupervised evaluation: scores cluster labels using only the
    features themselves (no true labels needed), via the silhouette
    score - useful when the dataset isn't fully labelled.
    """

    def __init__(self, features, y_pred):
        """
        Args:
            features: (n_samples, n_features) array the clusters were
                computed from.
            y_pred: (n_samples,) predicted cluster labels.
        """
        self.features = features
        self.predicted = y_pred
        print ("Predicted labels:", self.predicted)

    def run(self) -> None:
        self.evaluate(self.features, self.predicted)

    def evaluate(self, features, y_pred) -> None:
        """Compute and store the silhouette score in `self.silhouette`."""
        print("Calculating silhouette score...")
        self.silhouette = silhouette_score(features, y_pred)
        print(self)

    def __str__(self):
        return f"-----\nEvaluation results: \nSilhouette Score = {self.silhouette}\n-----\n\n"

class EvaluationWithGroundTruth(EvaluationStep):
    """Supervised evaluation: compares predicted cluster/class labels
    against known true labels using ARI, NMI, and a best-match accuracy
    (see `clustering_accuracy`).
    """

    def __init__(self, y_true, y_pred):
        """
        Args:
            y_true: (n_samples,) ground-truth labels.
            y_pred: (n_samples,) predicted cluster/class labels. Cluster
                ids don't need to match label ids - `clustering_accuracy`
                finds the best possible matching between them.
        """
        self.ground_truth = y_true
        self.predicted = y_pred
        print ("Ground truth labels:", self.ground_truth)
        print ("Predicted labels:", self.predicted)

        self.acc=0
        self.ari=0
        self.nmi=0

    def run(self) -> None:
        self.evaluate(self.ground_truth, self.predicted)

    def evaluate(self, y_true, y_pred) -> None:
        """Compute ARI, NMI and clustering accuracy, storing each on `self`."""
        self.ari = adjusted_rand_score(y_true, y_pred)
        self.nmi = normalized_mutual_info_score(y_true, y_pred)
        self.acc = self.clustering_accuracy(y_true, y_pred)
        print(self)

    def __str__(self):
        return f"-----\nEvaluation results: \nACC = {self.acc}, \nARI = {self.ari}, \nNMI = {self.nmi}\n-----\n\n"


    def clustering_accuracy(self, y_true, y_pred):
        """Best-case accuracy under the optimal cluster-id <-> true-label
        matching (since cluster labels are arbitrary integers with no
        inherent correspondence to the true class ids).

        Builds a (true classes x predicted clusters) contingency matrix,
        then uses the Hungarian algorithm (`linear_sum_assignment`) to
        find the one-to-one matching between clusters and true classes
        that maximizes the number of correctly "matched" samples -
        equivalent to the standard unsupervised clustering accuracy
        metric used in the clustering literature.

        Args:
            y_true: (n_samples,) ground-truth labels (any hashable type).
            y_pred: (n_samples,) predicted cluster labels (any hashable
                type; does not need to share y_true's label space).

        Returns:
            Fraction of samples whose predicted cluster matches its true
            class under the best possible matching, in [0, 1].
        """
        y_true = np.asarray(y_true, dtype=object)
        y_pred = np.asarray(y_pred, dtype=object)

        if y_true.shape[0] != y_pred.shape[0]:
            raise ValueError("y_true and y_pred must have the same length")
        if y_true.shape[0] == 0:
            raise ValueError("y_true and y_pred must not be empty")

        # Map arbitrary label values to dense integer ids so they can index
        # a contingency matrix.
        true_codes, _ = pd.factorize(y_true, sort=False)
        pred_codes, _ = pd.factorize(y_pred, sort=False)
        matrix = np.zeros(
            (true_codes.max() + 1, pred_codes.max() + 1),
            dtype=np.int64,
        )
        np.add.at(matrix, (true_codes, pred_codes), 1)

        # linear_sum_assignment minimizes cost, so negate the counts to
        # turn "maximize matches" into "minimize -matches".
        row_ind, col_ind = linear_sum_assignment(-matrix)

        matched = matrix[row_ind, col_ind].sum()
        return matched / len(y_true)