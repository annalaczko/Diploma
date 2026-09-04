"""Top-level pipeline: chains preprocessing steps, then runs
feature learning, clustering, and evaluation."""

from typing import Any, List

from pipeline.clustering.clustering_step import ClusteringStep
from pipeline.evaluation.evaluation_step import EvaluationStep
from pipeline.feature_learning.featurelearning_step import FeatureLearningStep
from pipeline.preprocessing.preprocessing_step import PreprocessingStep


class Pipeline:
    """Wires together one full experiment run: a list of preprocessing
    steps applied in order, one feature-learning step, one clustering
    step, and an optional evaluation step.

    This is a thin orchestrator - it owns no logic of its own beyond
    calling each step's `run()` in sequence and threading its output into
    the next step's input. All actual behavior (what "preprocessing"
    means, which feature-learning approach, etc.) lives in the step
    objects passed in, which is what makes each stage swappable (e.g.
    the note in main.py) without changing this class.
    """

    def __init__(
        self,
        preprocessing_steps: List[PreprocessingStep],
        feature_learning_step: FeatureLearningStep,
        clustering_step: ClusteringStep,
        evaluation_step: EvaluationStep | None = None,
    ):
        """
        Args:
            preprocessing_steps: steps run in order, each receiving the
                previous step's output as its input (the first receives
                the raw `data` passed to `run`).
            feature_learning_step: turns preprocessed data into a fixed
                feature representation.
            clustering_step: clusters the learned features.
            evaluation_step: optional; if given, scores the resulting
                cluster labels and its metrics are included in the result.
        """
        self.preprocessing_steps = preprocessing_steps
        self.feature_learning_step = feature_learning_step
        self.clustering_step = clustering_step
        self.evaluation_step = evaluation_step

    def run(self, data: Any) -> dict:
        """Run the full pipeline end to end on `data`.

        Returns:
            A dict with 'features' (from the feature-learning step),
            'labels' (from clustering), and 'metrics' (from evaluation,
            only present if an evaluation_step was configured).
        """
        for step in self.preprocessing_steps:
            data = step.run(data)

        features = self.feature_learning_step.run(data)
        labels = self.clustering_step.run(features)

        result = {"features": features, "labels": labels}
        if self.evaluation_step is not None:
            result["metrics"] = self.evaluation_step.run(labels)
        return result
