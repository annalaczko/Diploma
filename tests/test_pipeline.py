"""Tests for the top-level Pipeline class."""

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from pipeline.clustering.base import ClusteringStep
from pipeline.evaluation.base import EvaluationStep
from pipeline.feature_learning.base import FeatureLearningStep
from pipeline.pipeline import Pipeline
from pipeline.preprocessing.base import PreprocessingStep


# --- helper stubs ---

class PassthroughPreprocessing(PreprocessingStep):
    def run(self, data: Any) -> Any:
        return data


class TrackingPreprocessing(PreprocessingStep):
    """Records every call so we can assert ordering."""

    def __init__(self, tag: str):
        super().__init__(name=tag)
        self.calls = []

    def run(self, data: Any) -> Any:
        self.calls.append(data)
        return data + [self.name]


class ConstantFeatureLearning(FeatureLearningStep):
    def __init__(self, features: np.ndarray):
        super().__init__()
        self._features = features

    def run(self, data: Any) -> np.ndarray:
        return self._features


class ConstantClustering(ClusteringStep):
    def __init__(self, labels: np.ndarray):
        super().__init__()
        self._labels = labels

    def run(self, data: np.ndarray) -> np.ndarray:
        return self._labels


class ConstantEvaluation(EvaluationStep):
    def run(self, data: Any):
        return {"ari": 0.9, "nmi": 0.8}


# --- fixtures ---

@pytest.fixture
def features():
    return np.ones((5, 8))


@pytest.fixture
def labels():
    return np.zeros(5, dtype=int)


@pytest.fixture
def simple_pipeline(features, labels):
    return Pipeline(
        preprocessing_steps=[PassthroughPreprocessing()],
        feature_learning_step=ConstantFeatureLearning(features),
        clustering_step=ConstantClustering(labels),
    )


# --- tests ---

class TestPipelineRun:
    def test_result_contains_features_key(self, simple_pipeline, features):
        result = simple_pipeline.run("data")
        assert "features" in result

    def test_result_contains_labels_key(self, simple_pipeline):
        result = simple_pipeline.run("data")
        assert "labels" in result

    def test_features_value_matches_fl_output(self, simple_pipeline, features):
        result = simple_pipeline.run("data")
        np.testing.assert_array_equal(result["features"], features)

    def test_labels_value_matches_clustering_output(self, simple_pipeline, labels):
        result = simple_pipeline.run("data")
        np.testing.assert_array_equal(result["labels"], labels)

    def test_no_metrics_key_without_evaluation_step(self, simple_pipeline):
        result = simple_pipeline.run("data")
        assert "metrics" not in result

    def test_metrics_key_present_with_evaluation_step(self, features, labels):
        pipeline = Pipeline(
            preprocessing_steps=[PassthroughPreprocessing()],
            feature_learning_step=ConstantFeatureLearning(features),
            clustering_step=ConstantClustering(labels),
            evaluation_step=ConstantEvaluation(),
        )
        result = pipeline.run("data")
        assert "metrics" in result

    def test_evaluation_metrics_correct(self, features, labels):
        pipeline = Pipeline(
            preprocessing_steps=[PassthroughPreprocessing()],
            feature_learning_step=ConstantFeatureLearning(features),
            clustering_step=ConstantClustering(labels),
            evaluation_step=ConstantEvaluation(),
        )
        result = pipeline.run("data")
        assert result["metrics"] == {"ari": 0.9, "nmi": 0.8}

    def test_preprocessing_steps_are_chained_in_order(self, features, labels):
        step_a = TrackingPreprocessing("A")
        step_b = TrackingPreprocessing("B")
        pipeline = Pipeline(
            preprocessing_steps=[step_a, step_b],
            feature_learning_step=ConstantFeatureLearning(features),
            clustering_step=ConstantClustering(labels),
        )
        pipeline.run([])
        # A sees the original input, B sees A's output
        assert step_a.calls == [[]]
        assert step_b.calls == [["A"]]

    def test_empty_preprocessing_list_is_valid(self, features, labels):
        pipeline = Pipeline(
            preprocessing_steps=[],
            feature_learning_step=ConstantFeatureLearning(features),
            clustering_step=ConstantClustering(labels),
        )
        result = pipeline.run("raw")
        assert "features" in result and "labels" in result

    def test_multiple_preprocessing_steps(self, features, labels):
        steps = [PassthroughPreprocessing() for _ in range(5)]
        pipeline = Pipeline(
            preprocessing_steps=steps,
            feature_learning_step=ConstantFeatureLearning(features),
            clustering_step=ConstantClustering(labels),
        )
        result = pipeline.run("data")
        assert "features" in result

    def test_feature_learning_receives_preprocessed_data(self, labels):
        """FL step should receive the output of the last preprocessing step."""
        sentinel = object()

        class SentinelPreprocessing(PreprocessingStep):
            def run(self, data):
                return sentinel

        class AssertingFL(FeatureLearningStep):
            def run(self, data):
                assert data is sentinel, f"Expected sentinel, got {data!r}"
                return np.zeros((1, 4))

        pipeline = Pipeline(
            preprocessing_steps=[SentinelPreprocessing()],
            feature_learning_step=AssertingFL(),
            clustering_step=ConstantClustering(np.zeros(1, dtype=int)),
        )
        pipeline.run("ignored")
