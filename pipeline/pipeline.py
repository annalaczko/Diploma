"""Top-level pipeline: chains preprocessing steps, then runs
feature learning, clustering, and evaluation."""

from typing import Any, List

from pipeline.clustering.base import ClusteringStep
from pipeline.evaluation.base import EvaluationStep
from pipeline.feature_learning.base import FeatureLearningStep
from pipeline.preprocessing.base import PreprocessingStep


class Pipeline:
    def __init__(
        self,
        preprocessing_steps: List[PreprocessingStep],
        feature_learning_step: FeatureLearningStep,
        clustering_step: ClusteringStep,
        evaluation_step: EvaluationStep | None = None,
    ):
        self.preprocessing_steps = preprocessing_steps
        self.feature_learning_step = feature_learning_step
        self.clustering_step = clustering_step
        self.evaluation_step = evaluation_step

    def run(self, data: Any) -> dict:
        for step in self.preprocessing_steps:
            data = step.run(data)

        features = self.feature_learning_step.run(data)
        labels = self.clustering_step.run(features)

        result = {"features": features, "labels": labels}
        if self.evaluation_step is not None:
            result["metrics"] = self.evaluation_step.run(labels)
        return result
