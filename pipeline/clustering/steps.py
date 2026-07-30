"""Concrete clustering steps."""

import numpy as np

from pipeline.clustering.base import ClusteringStep
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist


class HierarchicalClustering(ClusteringStep):
    """ Does the hierarchical clustering, and evaluates the results via the metrics used by Parisotto

    Args:
        ClusteringStep (_type_): _description_

    Returns:
        _type_: _description_
    """

    # TODO: tests

    def __init__(self, vectors):

        """
        Initialization. We build this on scipy and have only the methods imported that Parisotto used right now. 
        """

        super().__init__()
        self.vectors = vectors
        self.methods=['single', 'complete', 'average', 'weighted', 'median centroid', 'ward']
        self.metrics=['euclidean', 'cityblock', 'chebyshev', 'cosine'] 

        self.best_method = 'average'
        self.best_metric = 'euclidean'

    def run(self):

        """
        Runs the full clustering as a black box

        Returns:
            fclusterdata : ndarray
                    A vector of length n. T[i] is the flat cluster number to
                    which original observation i belongs.
        """

        result = hierarchy.fclusterdata(self.vectors, metric=self.best_metric, method=self.best_method)
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

        return results
