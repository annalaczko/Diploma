"""Example usage - configure the steps once, then run the pipeline either
in memory (batches, no intermediate folders) or on disk (every step
writes its output folder, like running the steps one by one)."""

import torch

from pipeline.clustering import HierarchicalClustering
from pipeline.feature_learning import VAE
from pipeline.pipeline import Pipeline
from pipeline.preprocessing import Augmentor, BackgroundEraser, Normalizer, Repair

run_id = "example_run"
source_folder = "data/pipeline/example_run/PreprocessingStep/Repair"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vae = VAE(f=4, n=3, s=1, k=16, beta_max=32, run_id=run_id).to(device)

pipeline = Pipeline(
    preprocessing_steps=[
        # BackgroundEraser(mode="segmentation", model="models/yolov8n_seg_finetuned.pt",
        #                  mask_margin=0.01, record_axes=True, run_id=run_id),
        # Normalizer(background_color=(255, 255, 255), run_id=run_id),
        # Repair(contrast_factor=5, dilate_kernel_size=1, dilate_iterations=1,
        #        skeletonize=False, output_thickness=2, run_id=run_id),
        # Augmentor(num_augmentations=2, run_id=run_id),
    ],
    feature_learning_step=vae,
    clustering_factory=lambda features_csv: HierarchicalClustering(input_path=features_csv, cluster_num=55),
)

if __name__ == "__main__":
    # mode="disk" runs the very same steps through intermediate folders instead.
    # experiment_name logs the whole run to MLflow (see pipeline/README.md);
    # browse it with `mlflow ui --backend-store-uri sqlite:///mlflow.db`.
    result = pipeline.run(source_folder, mode="memory", preprocess_batch_size=128,
                          feature_mode="train_model", epochs=1000, lr=1e-4, batch_size=128,
                          # also write the final preprocessed images (Repair's output folder);
                          # "all" writes every preprocessing step's output, None nothing
                          save_preprocessed=None,
                          experiment_name="pottery_pipeline", run_name=run_id)
    print(result["features"].head())
    print(result["labels"])
