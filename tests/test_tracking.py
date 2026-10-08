"""Tests for MLflow tracking (pipeline.tracking): steps log only inside an
active run, and a tracked Pipeline run records every stage's results,
including the names of images without a shard or a coordinate system.
conftest.py points MLFLOW_TRACKING_URI at a per-test SQLite file."""

from pathlib import Path
from types import SimpleNamespace

import cv2
import mlflow
import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from pipeline import tracking
from pipeline.clustering import HierarchicalClustering
from pipeline.evaluation import EvaluationWithGroundTruth
from pipeline.feature_learning.vae import VAE
from pipeline.pipeline import Pipeline
from pipeline.preprocessing import BackgroundEraser2, Normalizer

H, W = 1400, 1000
SHARD_BOX = [400.0, 520.0, 520.0, 640.0]


def _sheet(with_axes=True, with_shard=True):
    """A4-like page: pencil-gray ⅂ axes and a dark shard outline, either optional."""
    img = np.full((H, W, 3), 255, np.uint8)
    if with_axes:
        cv2.line(img, (350, 500), (700, 500), (150, 150, 150), 2)
        cv2.line(img, (700, 500), (700, 900), (150, 150, 150), 2)
    if with_shard:
        cv2.ellipse(img, (460, 580), (40, 50), 0, 0, 360, (0, 0, 0), 3)
    return img


def _mock_yolo(eraser):
    """YOLO stand-in: a fixed box on every page with ink, nothing on a blank page."""
    hit = SimpleNamespace(xyxy=torch.tensor([SHARD_BOX]))
    miss = SimpleNamespace(xyxy=torch.zeros((0, 4)))

    def result(image, path=None):
        return SimpleNamespace(path=path, boxes=hit if image.min() < 100 else miss)

    def predict(source, **_):
        if isinstance(source, str):  # disk mode: flat symlink folder
            return [result(cv2.imread(str(p)), str(p)) for p in sorted(Path(source).iterdir())]
        return [result(img) for img in source]

    eraser._get_model = lambda: SimpleNamespace(predict=predict)
    return eraser


def _only_run(experiment_name):
    runs = mlflow.search_runs(experiment_names=[experiment_name], output_format="list")
    assert len(runs) == 1
    return runs[0]


def _artifact_text(run, path):
    return Path(mlflow.artifacts.download_artifacts(run_id=run.info.run_id, artifact_path=path)).read_text()


def test_helpers_are_noops_without_active_run(tmp_path):
    tracking.log_metrics({"x": 1.0})
    tracking.log_params({"y": 2})
    tracking.log_lines(["a"], "a.txt")

    assert not tracking.is_active()
    assert not (tmp_path / "mlflow.db").exists()


def test_disk_run_logs_flat_names_of_failed_images(tmp_path):
    source = tmp_path / "raw"
    (source / "survey").mkdir(parents=True)
    Image.fromarray(_sheet()).save(source / "survey" / "ok.png")
    Image.fromarray(_sheet(with_axes=False)).save(source / "survey" / "no_axes.png")
    Image.fromarray(_sheet(with_axes=False, with_shard=False)).save(source / "survey" / "blank.png")

    eraser = _mock_yolo(BackgroundEraser2(input_path=str(source), output_path=str(tmp_path / "out"),
                                          mode="detection", box_margin=0.0, debug_images=0))
    with tracking.start_run("eraser"):
        eraser.run()

    run = _only_run("eraser")
    prefix = "preprocessing/BackgroundEraser2"
    assert run.data.metrics[f"{prefix}/n_images"] == 3
    assert run.data.metrics[f"{prefix}/no_shard_count"] == 1
    assert run.data.metrics[f"{prefix}/no_axes_count"] == 1
    assert _artifact_text(run, f"{prefix}/no_shard.txt") == "survey__blank.png\n"
    assert _artifact_text(run, f"{prefix}/no_axes.txt") == "survey__no_axes.png\n"
    assert run.data.params[f"{prefix}/axis_margin"] == "0.005"


@pytest.mark.parametrize("mode", ["memory", "disk"])
def test_pipeline_run_tracks_every_stage(tmp_path, recwarn, mode):
    source = tmp_path / "raw"
    source.mkdir()
    Image.fromarray(_sheet()).save(source / "a.png")
    Image.fromarray(_sheet()).save(source / "b.png")
    Image.fromarray(_sheet(with_axes=False)).save(source / "c.png")
    Image.fromarray(_sheet(with_axes=False, with_shard=False)).save(source / "d.png")

    eraser = _mock_yolo(BackgroundEraser2(mode="detection", box_margin=0.0, debug_images=0,
                                          output_path=str(tmp_path / "erased")))
    torch.manual_seed(0)
    vae = VAE(output_path=str(tmp_path / "features"))
    pipe = Pipeline(
        [eraser, Normalizer(target_size=256, background_color=(255, 255, 255),
                            output_path=str(tmp_path / "normalized"))],
        feature_learning_step=vae,
        clustering_factory=lambda csv: HierarchicalClustering(input_path=csv, cluster_num=2),
        evaluation_factory=lambda y_pred: EvaluationWithGroundTruth([0, 0, 1], y_pred),
    )

    result = pipe.run(str(source), mode=mode, feature_mode="train_model",
                      epochs=2, batch_size=2, lr=1e-4,
                      experiment_name="e2e", run_name="smoke")

    run = _only_run("e2e")
    metrics, params = run.data.metrics, run.data.params
    assert run.info.run_name == "smoke"
    # Preprocessing: counts and names of the skipped images.
    assert metrics["preprocessing/BackgroundEraser2/no_shard_count"] == 1
    assert metrics["preprocessing/BackgroundEraser2/no_axes_count"] == 1
    if mode == "memory":
        assert metrics["preprocessing/n_images"] == 3
    assert _artifact_text(run, "preprocessing/BackgroundEraser2/no_shard.txt") == "d.png\n"
    assert _artifact_text(run, "preprocessing/BackgroundEraser2/no_axes.txt") == "c.png\n"
    assert params["preprocessing/BackgroundEraser2/axis_margin"] == "0.005"
    # Feature learning: one value per epoch, config and features.csv.
    history = mlflow.tracking.MlflowClient().get_metric_history(run.info.run_id, "feature/loss")
    assert [m.step for m in history] == [1, 2]
    assert params["feature/train/epochs"] == "2"
    assert params["feature/VAE/k"] == str(vae.hparams["k"])
    assert "feature/stopped_epoch" in metrics
    assert len(pd.read_csv(mlflow.artifacts.download_artifacts(
        run_id=run.info.run_id, artifact_path="feature/features.csv"))) == 3
    # Clustering and evaluation.
    assert params["clustering/cluster_num"] == "2"
    assert metrics["clustering/n_clusters"] >= 1
    clusters = pd.read_csv(mlflow.artifacts.download_artifacts(
        run_id=run.info.run_id, artifact_path="clustering/clusters.csv"))
    assert clusters["filename"].tolist() == ["a.png", "b.png", "c.png"]
    assert metrics["eval/acc"] == result["metrics"]["acc"]
    assert "eval/per_class_accuracy.csv" in [
        f.path for f in mlflow.tracking.MlflowClient().list_artifacts(run.info.run_id, "eval")
    ]
    # No parameter was logged twice with different values.
    assert not [w for w in recwarn if "could not log param" in str(w.message)]
