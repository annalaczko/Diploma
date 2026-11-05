"""Evaluation steps: how good is a clustering?

Two kinds of evaluation:
  - EvaluationWithNoGroundTruth: needs only the features and the cluster
    labels (silhouette score: are clusters compact and well separated?).
  - EvaluationWithGroundTruth: compares the clusters with the known pottery
    types (ARI, NMI and accuracy under the best cluster <-> type matching).
"""

from abc import abstractmethod
from typing import Dict

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

from pipeline import tracking
from pipeline.steps.step import Step


class EvaluationStep(Step):
    """Base class of the evaluations. They get their inputs in `__init__`
    (they don't call `Step.__init__`: there is no folder to read or write),
    and `run()` returns {metric name: value}, also logged to MLflow as
    'eval/<name>'.
    """

    @abstractmethod
    def run(self) -> Dict[str, float]:
        raise NotImplementedError

class EvaluationWithNoGroundTruth(EvaluationStep):
    """Silhouette score of the clusters, computed from the features alone
    (no true labels needed). Ranges from -1 to 1; higher means each image
    is closer to its own cluster than to the nearest other one.
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

    def run(self) -> Dict[str, float]:
        self.evaluate(self.features, self.predicted)
        metrics = {"silhouette": self.silhouette}
        tracking.log_metrics(metrics, prefix="eval/")
        return metrics

    def evaluate(self, features, y_pred) -> None:
        """Compute and store the silhouette score in `self.silhouette`."""
        print("Calculating silhouette score...")
        self.silhouette = silhouette_score(features, y_pred)
        print(self)

    def __str__(self):
        return f"-----\nEvaluation results: \nSilhouette Score = {self.silhouette}\n-----\n\n"

class EvaluationWithGroundTruth(EvaluationStep):
    """Compares the clusters with the true labels:
      - ARI (adjusted Rand index): agreement of the two groupings, 0 = random.
      - NMI (normalized mutual information): shared information, 0..1.
      - ACC: accuracy under the best one-to-one cluster <-> class matching
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
        self.per_class = None

    def run(self) -> Dict[str, float]:
        self.evaluate(self.ground_truth, self.predicted)
        metrics = {"acc": self.acc, "ari": self.ari, "nmi": self.nmi}
        tracking.log_metrics(metrics, prefix="eval/")
        tracking.log_table(self.per_class, "eval/per_class_accuracy.csv")
        return metrics

    def evaluate(self, y_true, y_pred) -> None:
        """Compute ARI, NMI and clustering accuracy, storing each on `self`."""
        self.ari = adjusted_rand_score(y_true, y_pred)
        self.nmi = normalized_mutual_info_score(y_true, y_pred)
        self.acc = self.clustering_accuracy(y_true, y_pred)
        self.per_class = self.per_class_accuracy(y_true, y_pred)
        print(self)

    def __str__(self):
        return f"-----\nEvaluation results: \nACC = {self.acc}, \nARI = {self.ari}, \nNMI = {self.nmi}\n-----\n\n"


    def clustering_accuracy(self, y_true, y_pred):
        """Accuracy under the best one-to-one matching of clusters to
        classes. Cluster ids are arbitrary (cluster 3 has nothing to do with
        class 3), so first each cluster is paired with the class it fits
        best: the Hungarian algorithm (`linear_sum_assignment`) picks the
        pairing that maximizes the number of correctly placed samples. This
        is the standard clustering accuracy of the literature.

        Args:
            y_true: (n_samples,) ground-truth labels (any hashable type).
            y_pred: (n_samples,) predicted cluster labels (any hashable
                type; does not need to share y_true's label space).

        Returns:
            Fraction of samples whose predicted cluster matches its true
            class under the best possible matching, in [0, 1].
        """
        matrix, _, _, row_ind, col_ind = self._best_matching(y_true, y_pred)
        matched = matrix[row_ind, col_ind].sum()
        return matched / matrix.sum()

    def per_class_accuracy(self, y_true, y_pred):
        """Accuracy broken down by true class, under the same optimal
        one-to-one cluster <-> class matching that `clustering_accuracy`
        uses - so the overall ACC is exactly these per-class correct
        counts summed and divided by the total.

        A class's accuracy is the fraction of its samples that landed in
        the cluster matched to it. With fewer clusters than classes some
        classes get no cluster at all (accuracy 0, `matched_cluster`
        None) - `top_cluster`/`top_cluster_share` still show where most
        of such a class's samples went, independent of the matching.

        Returns:
            A DataFrame with one row per true class, sorted by class size:
            'class', 'n', 'matched_cluster', 'correct', 'accuracy',
            'top_cluster', 'top_cluster_share'.
        """
        matrix, true_labels, pred_labels, row_ind, col_ind = self._best_matching(y_true, y_pred)
        matched = dict(zip(row_ind, col_ind))

        rows = []
        for i, label in enumerate(true_labels):
            n = int(matrix[i].sum())
            j = matched.get(i)
            correct = int(matrix[i, j]) if j is not None else 0
            top = int(matrix[i].argmax())
            rows.append({
                "class": label,
                "n": n,
                "matched_cluster": pred_labels[j] if j is not None else None,
                "correct": correct,
                "accuracy": correct / n,
                "top_cluster": pred_labels[top],
                "top_cluster_share": matrix[i, top] / n,
            })
        df = pd.DataFrame(rows)
        # Keep cluster ids as-is: a None (unmatched class) would otherwise
        # make pandas turn the whole column into floats (0 -> 0.0, None -> NaN).
        df["matched_cluster"] = pd.Series([r["matched_cluster"] for r in rows], dtype=object)
        return df.sort_values("n", ascending=False).reset_index(drop=True)

    @staticmethod
    def _best_matching(y_true, y_pred):
        """Build the (true classes x predicted clusters) contingency matrix
        and find the one-to-one class <-> cluster matching maximizing the
        number of matched samples (Hungarian algorithm).

        Returns:
            (matrix, true_labels, pred_labels, row_ind, col_ind): the
            contingency matrix, the label values its rows/columns stand
            for, and the matched (row, column) index pairs.
        """
        y_true = np.asarray(y_true, dtype=object)
        y_pred = np.asarray(y_pred, dtype=object)

        if y_true.shape[0] != y_pred.shape[0]:
            raise ValueError("y_true and y_pred must have the same length")
        if y_true.shape[0] == 0:
            raise ValueError("y_true and y_pred must not be empty")

        # Map arbitrary label values to dense integer ids so they can index
        # a contingency matrix.
        true_codes, true_labels = pd.factorize(y_true, sort=False)
        pred_codes, pred_labels = pd.factorize(y_pred, sort=False)
        matrix = np.zeros(
            (true_codes.max() + 1, pred_codes.max() + 1),
            dtype=np.int64,
        )
        np.add.at(matrix, (true_codes, pred_codes), 1)

        # linear_sum_assignment minimizes cost, so negate the counts to
        # turn "maximize matches" into "minimize -matches".
        row_ind, col_ind = linear_sum_assignment(-matrix)
        return matrix, true_labels, pred_labels, row_ind, col_ind