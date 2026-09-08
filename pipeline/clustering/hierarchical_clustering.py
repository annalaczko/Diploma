"""Concrete clustering steps."""

import numpy as np
import re as regex_module

from pipeline.clustering.clustering_step import ClusteringStep
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist
import pandas as pd


class HierarchicalClustering(ClusteringStep):
    """Hierarchical clustering over precomputed feature vectors, built on
    scipy. Supports the same linkage methods and distance metrics used by
    Parisotto, and can sweep all of their combinations to pick the best one
    by cophenetic correlation (see `compare_all`).
    """

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
        self.vectors = self.parse_feature_string(pd.read_csv(input_path)['features'].values)
        self.methods=['single', 'complete', 'average', 'weighted', 'centroid', 'ward']
        self.metrics=['euclidean', 'cityblock', 'chebyshev', 'cosine'] 

        self.best_method = method if method in self.methods else 'average'
        self.best_metric = metric if metric in self.metrics else 'euclidean'
        self.cluster_num = cluster_num


    def run(self):
        """Cluster `self.vectors` into `self.cluster_num` flat clusters
        using `self.best_method`/`self.best_metric` (either passed in at
        construction, or the winners from a prior `compare_all()` call).

        Returns:
            fclusterdata : ndarray
                    A vector of length n. T[i] is the flat cluster number to
                    which original observation i belongs.
        """

        result = hierarchy.fclusterdata(self.vectors, t=self.cluster_num, criterion="maxclust", metric=self.best_metric, method=self.best_method)
        return result

    def compare_all(self):
        """Try every (method, metric) combination and rank them by
        cophenetic correlation, a measure of how well a linkage's
        dendrogram distances preserve the original pairwise distances
        (higher = the clustering better reflects the true data geometry).

        As a side effect, updates `self.best_method`/`self.best_metric`
        to the winning combination, so a subsequent `run()` call uses it.

        Returns:
            list of dict: one entry per (method, metric) combination, with
            keys 'method', 'metric' and 'cophenetic coefficient score'.
        """

        results=[]
        
        for metric in self.metrics:

            # Compute the distance matrix for the current metric
            distance_matrix=pdist(self.vectors, metric=metric)

            for method in self.methods:    

                # Creating the linkage matrix and computing the cophenetic score for the current method and metric
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

        return results

    def parse_feature_string(self, values):
        """Turn the 'features' column of a feature-learning CSV into a
        dense (n_samples, n_features) float matrix.

        Feature vectors get written to CSV as their string repr (e.g.
        numpy's `"[0.1, 0.2, ...]"` or space-separated form), since CSV
        cells can't hold arrays directly. This undoes that: for each row,
        strips brackets, normalizes comma/whitespace separators, and
        parses the numbers back out.

        Args:
            values: the CSV column's values - typically a pandas Series
                or numpy array of strings, but a single string or an
                already-numeric array/row is also accepted.

        Returns:
            A single 1D array if `values` was a single row/string, else a
            stacked (n_samples, n_features) array. Raises ValueError if
            the parsed rows don't all share the same length, since
            clustering requires a consistent feature dimensionality.
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
            cleaned = regex_module.sub(r'\s+', ' ', cleaned).strip()

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

