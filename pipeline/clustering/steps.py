"""Concrete clustering steps."""

import numpy as np

from pipeline.clustering.base import ClusteringStep


class HierarchicalClustering(ClusteringStep):
    def __init__(self, n_clusters: int = 10, linkage: str = "ward"):
        super().__init__()
        self.n_clusters = n_clusters
        self.linkage = linkage

    def run(self, data: np.ndarray) -> np.ndarray:
        # TODO: use sklearn.cluster.AgglomerativeClustering
        return np.zeros(len(data), dtype=int)


class KMeansClustering(ClusteringStep):
    def __init__(self, n_clusters: int = 10, random_state: int = 42):
        super().__init__()
        self.n_clusters = n_clusters
        self.random_state = random_state

    def run(self, data: np.ndarray) -> np.ndarray:
        # TODO: use sklearn.cluster.KMeans
        return np.zeros(len(data), dtype=int)
