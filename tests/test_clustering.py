"""Tests for clustering steps."""

import numpy as np
import pytest

from pipeline.clustering.steps import HierarchicalClustering, KMeansClustering


# --- HierarchicalClustering ---

class TestHierarchicalClustering:
    def test_default_params(self):
        step = HierarchicalClustering()
        assert step.n_clusters == 10
        assert step.linkage == "ward"

    def test_custom_params_stored(self):
        step = HierarchicalClustering(n_clusters=5, linkage="complete")
        assert step.n_clusters == 5
        assert step.linkage == "complete"

    def test_run_returns_ndarray(self, hierarchical_step, sample_features):
        labels = hierarchical_step.run(sample_features)
        assert isinstance(labels, np.ndarray)

    def test_run_output_length_matches_input(self, hierarchical_step, sample_features):
        labels = hierarchical_step.run(sample_features)
        assert len(labels) == len(sample_features)

    def test_run_dtype_is_int(self, hierarchical_step, sample_features):
        labels = hierarchical_step.run(sample_features)
        assert np.issubdtype(labels.dtype, np.integer)

    def test_callable_alias(self, hierarchical_step, sample_features):
        assert np.array_equal(
            hierarchical_step(sample_features),
            hierarchical_step.run(sample_features),
        )

    def test_name_defaults_to_class_name(self):
        step = HierarchicalClustering()
        assert step.name == "HierarchicalClustering"


# --- KMeansClustering ---

class TestKMeansClustering:
    def test_default_params(self):
        step = KMeansClustering()
        assert step.n_clusters == 10
        assert step.random_state == 42

    def test_custom_params_stored(self):
        step = KMeansClustering(n_clusters=3, random_state=7)
        assert step.n_clusters == 3
        assert step.random_state == 7

    def test_run_returns_ndarray(self, kmeans_step, sample_features):
        labels = kmeans_step.run(sample_features)
        assert isinstance(labels, np.ndarray)

    def test_run_output_length_matches_input(self, kmeans_step, sample_features):
        labels = kmeans_step.run(sample_features)
        assert len(labels) == len(sample_features)

    def test_run_dtype_is_int(self, kmeans_step, sample_features):
        labels = kmeans_step.run(sample_features)
        assert np.issubdtype(labels.dtype, np.integer)

    def test_callable_alias(self, kmeans_step, sample_features):
        assert np.array_equal(
            kmeans_step(sample_features),
            kmeans_step.run(sample_features),
        )

    def test_name_defaults_to_class_name(self):
        step = KMeansClustering()
        assert step.name == "KMeansClustering"
