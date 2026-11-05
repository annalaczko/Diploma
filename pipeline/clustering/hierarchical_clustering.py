"""HierarchicalClustering: groups the feature vectors into a fixed number
of clusters with agglomerative (bottom-up) clustering.

Agglomerative clustering starts with every image as its own cluster and
repeatedly merges the two closest clusters. The `metric` says how the
distance of two vectors is measured (e.g. euclidean, cosine); the `method`
(linkage) says how the distance of two clusters follows from it (e.g.
'ward' merges the pair that increases the within-cluster variance least).
The merge tree is finally cut so that `cluster_num` clusters remain.
"""

import re

import numpy as np
import pandas as pd
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist

from pipeline import tracking
from pipeline.clustering.clustering_step import ClusteringStep


class HierarchicalClustering(ClusteringStep):
    """Hierarchical clustering of the vectors in a features.csv (scipy).
    `compare_all` can pick the method/metric pair with the best cophenetic
    correlation (the same methods and metrics as Parisotto's work).
    """

    METHODS = ('single', 'complete', 'average', 'weighted', 'centroid', 'ward')
    METRICS = ('euclidean', 'cityblock', 'chebyshev', 'cosine')
    # Linkages defined on Euclidean distances only: scipy runs them on any
    # distance matrix without complaint, but the result is meaningless.
    EUCLIDEAN_ONLY = ('centroid', 'ward')

    def __init__(self, input_path, cluster_num=30, method=None, metric=None, output_path=None, run_id="default_run"):
        """
        Args:
            input_path: CSV path with a 'features' column, as produced by
                a feature-learning step (each cell holding a bracketed,
                comma/space-separated vector string).
            cluster_num: number of flat clusters to extract in `run`.
            method: scipy linkage method to use; falls back to 'average'
                if not one of `self.methods`.
            metric: scipy distance metric to use; falls back to 'euclidean'
                if not one of `self.metrics`.
            output_path: where results are saved; if omitted, auto-derived
                and namespaced under `run_id` (see `Step.__init__`).
            run_id: identifies this pipeline run; namespaces the
                auto-derived `output_path` when one isn't given explicitly.
        """

        super().__init__(
            input_path=input_path,
            output_path=output_path,
            name=None,
            delete_input_data=False,
            save_results=True,
            run_id=run_id,
        )
        features = pd.read_csv(input_path)
        self.vectors = self.parse_feature_string(features['features'].values)
        self.filenames = features['filename'].tolist() if 'filename' in features else None
        self.methods = list(self.METHODS)
        self.metrics = list(self.METRICS)

        self.best_method = method if method in self.methods else 'average'
        self.best_metric = metric if metric in self.metrics else 'euclidean'
        self.cluster_num = cluster_num


    def run(self):
        """Cluster `self.vectors` into `self.cluster_num` clusters with
        `best_method`/`best_metric` (as given, or chosen by `compare_all`).

        Returns:
            An array with the cluster number (1, 2, ...) of every image.
        """
        result = hierarchy.fclusterdata(self.vectors, t=self.cluster_num, criterion="maxclust", metric=self.best_metric, method=self.best_method)
        self._log_run(result)
        return result

    def _log_run(self, labels):
        """Log the clustering setup, cluster size statistics and a
        clusters.csv (filename -> cluster) to the active MLflow run."""
        tracking.log_params({
            "cluster_num": self.cluster_num,
            "method": self.best_method,
            "metric": self.best_metric,
            "n_samples": len(labels),
            "n_features": self.vectors.shape[1],
        }, prefix="clustering/")
        sizes = np.bincount(labels)[1:]  # fcluster ids start at 1
        sizes = sizes[sizes > 0]
        tracking.log_metrics({
            "n_clusters": len(sizes),
            "cluster_size_min": sizes.min(),
            "cluster_size_max": sizes.max(),
            "cluster_size_mean": sizes.mean(),
            "singleton_clusters": int((sizes == 1).sum()),
        }, prefix="clustering/")
        names = self.filenames if self.filenames is not None else range(len(labels))
        tracking.log_table(pd.DataFrame({"filename": names, "cluster": labels}), "clustering/clusters.csv")

    def compare_all(self):
        """Try every (method, metric) pair and keep the one with the highest
        cophenetic correlation: how well the distances in the merge tree
        match the original distances between the vectors (higher = more
        faithful). Sets `best_method`/`best_metric`, which `run()` uses.

        Returns:
            One dict per pair: 'method', 'metric' and 'cophenetic
            coefficient score' (scipy's `cophenet` result, a tuple whose
            first element is the coefficient).
        """
        results = []
        for metric in self.metrics:
            distance_matrix = pdist(self.vectors, metric=metric)
            for method in self.methods:
                linkage_matrix = hierarchy.linkage(distance_matrix, method = method, metric = metric)
                cophenetic_score = hierarchy.cophenet(linkage_matrix, distance_matrix)

                results.append({'method' : method,
                                 'metric' : metric,
                                 'cophenetic coefficient score': cophenetic_score})

        best_result = max(
            results,
            key=lambda result: result['cophenetic coefficient score']
        )
        self.best_method = best_result['method']
        self.best_metric = best_result['metric']

        # cophenet(Z, Y) returns (coefficient, cophenetic distances); log
        # only the coefficient.
        def _coefficient(score):
            return score[0] if isinstance(score, tuple) else score
        table = pd.DataFrame([
            {**r, 'cophenetic coefficient score': _coefficient(r['cophenetic coefficient score'])}
            for r in results
        ])
        tracking.log_table(table, "clustering/compare_all.csv")
        tracking.log_metrics({"best_cophenetic": _coefficient(best_result['cophenetic coefficient score'])},
                             prefix="clustering/")
        return results

    def parse_feature_string(self, values):
        """Turn the 'features' column of a features.csv back into an
        (n_samples, n_features) float matrix.

        A CSV cell cannot hold an array, so the vectors are stored as text,
        e.g. "[0.1 0.2 ...]" (numpy's print form), "0.1, 0.2" or "0.1 0.2".
        Brackets and commas are dropped and the numbers parsed.

        Args:
            values: the column (pandas Series or array of strings); a single
                string or an already numeric row also works.

        Returns:
            A 1D array for a single string, else the stacked matrix. Raises
            ValueError if the rows have different lengths.
        """
        def _parse_one(raw):
            """Parse a single cell into a 1D float array, or None if it's missing (NaN)."""
            if raw is None or pd.isna(raw):
                return None

            if not isinstance(raw, str):
                return np.asarray(raw, dtype=float)

            cleaned = raw.strip()
            if cleaned.startswith('[') and cleaned.endswith(']'):
                cleaned = cleaned[1:-1]

            cleaned = cleaned.replace(',', ' ')
            cleaned = re.sub(r'\s+', ' ', cleaned).strip()

            if not cleaned:
                return np.empty(0, dtype=float)

            return np.fromstring(cleaned, sep=' ', dtype=float)

        if isinstance(values, pd.Series):
            values = values.to_numpy()

        if isinstance(values, np.ndarray) and values.ndim == 0:
            values = values.reshape(1)

        if isinstance(values, str):
            parsed = _parse_one(values)
            return parsed

        parsed = []
        for item in np.asarray(values, dtype=object).ravel():
            row = _parse_one(item)
            if row is not None:
                parsed.append(np.asarray(row, dtype=float))

        if len(parsed) == 0:
            return np.empty((0, 0), dtype=float)

        lengths = {len(v) for v in parsed}
        if len(lengths) != 1:
            raise ValueError(
                "All feature rows must have the same dimensionality before clustering. "
                f"Found lengths: {sorted(lengths)}"
            )

        return np.vstack(parsed).astype(float)

