"""Example usage - instantiate concrete steps and run the pipeline."""

from pipeline.clustering import HierarchicalClustering
from pipeline.feature_learning import PretrainedFeatureLearning
from pipeline.pipeline import Pipeline
from pipeline.preprocessing import ImageEnhancement, ObjectDetection

pipeline = Pipeline(
    preprocessing_steps=[
        ImageEnhancement(method="denoise"),
        ObjectDetection(model_name="yolo", conf_threshold=0.5),
    ],
    feature_learning_step=PretrainedFeatureLearning(backbone="resnet50"),
    clustering_step=HierarchicalClustering(n_clusters=10),
)

if __name__ == "__main__":
    data = []  # TODO: load raw images
    result = pipeline.run(data)
    print(result)
