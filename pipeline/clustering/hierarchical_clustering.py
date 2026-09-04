"""Concrete clustering steps."""

import numpy as np
import re as regex_module

from numpy import strings

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

    def __init__(self, input_path, cluster_num=30, method=None, metric=None):
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
        """

        super().__init__(input_path=input_path, output_path=None, name=None, delete_input_data=False, save_results=True)
        self.vectors = self.parse_feature_string(pd.read_csv(input_path)['features'].values)
        self.methods=['single', 'complete', 'average', 'weighted', 'centroid', 'ward']
        self.metrics=['euclidean', 'cityblock', 'chebyshev', 'cosine'] 

        self.best_method = method if method in self.methods else 'average'
        self.best_metric = metric if metric in self.metrics else 'euclidean'
        self.cluster_num = cluster_num


    def run(self):

        """
        Runs the full clustering as a black box

        Returns:
            fclusterdata : ndarray
                    A vector of length n. T[i] is the flat cluster number to
                    which original observation i belongs.
        """

        result = hierarchy.fclusterdata(self.vectors, t=self.cluster_num, criterion="maxclust", metric=self.best_metric, method=self.best_method)
        return result
    
    def compare_all(self):

        """
        Runs all the variation for clustering between methods and metrics.

        We here run the same clustering as in "run" but we can control all the pipeline pieces separately.


        Returns:
            list of dict: A list of dictionaries containing the clustering results for each combination of method and metric.
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
        def _parse_one(raw):
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

