"""Concrete clustering steps."""

import numpy as np

from pipeline.clustering.base import ClusteringStep
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist


class HierarchicalClustering(ClusteringStep):

    """_summary_
    """

    # TODO: tests

    def __init__(self, vectors):
        super().__init__()
        self.vectors = vectors
        self.methods=['single', 'complete', 'average', 'weighted', 'median centroid', 'ward']
        self.metrics=['euclidean', 'cityblock', 'chebyshev', 'cosine'] 

        self.best_method = 'average'
        self.best_metric = 'euclidean'

    def run(self):
        result = hierarchy.fclusterdata(self.vectors, metric=self.best_metric, method=self.best_method)
        return result
    
    def compare_all(self):
        results=[]
        
        for metric in self.metrics:
            distance_matrix=pdist(self.vectors, metric=metric)
            for method in self.methods:    
                linkage_matrix = hierarchy.linkage(distance_matrix, method = method, metric = metric)
                cophenetic_score = hierarchy.cophenet(linkage_matrix, distance_matrix)

                results.append({'method' : method,
                                 'metric' : metric,
                                 'cophenetic coefficient score': cophenetic_score})

        return results
    


# class KMeansClustering(ClusteringStep):
#     def __init__(self, n_clusters: int = 10, random_state: int = 42):
#         super().__init__()
#         self.n_clusters = n_clusters
#         self.random_state = random_state

#     def run(self, data: np.ndarray) -> np.ndarray:
#         # TODO: use sklearn.cluster.KMeans
#         return np.zeros(len(data), dtype=int)
