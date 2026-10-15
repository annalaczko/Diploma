"""Tests for config-driven experiment runs (pipeline/experiment.py)."""

from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest
import yaml
from PIL import Image

from pipeline.experiment import load_config, run_experiment, run_name


def _write(path: Path, data) -> Path:
    path.write_text(yaml.safe_dump(data))
    return path


def test_experiment_config_is_merged_over_defaults(tmp_path):
    config = load_config(_write(tmp_path / "exp.yaml", {"model": {"params": {"k": 16, "beta_max": 32}}}))

    assert config["model"]["params"] == {"f": 4, "n": 3, "s": 1, "k": 16, "beta_max": 32}
    assert config["clustering"]["method"] == "ward"  # untouched default


def test_overrides_are_parsed_as_yaml(tmp_path):
    config = load_config(None, ["model.params.k=32", "training.lr=1e-3", "data.labels=null"])

    assert config["model"]["params"]["k"] == 32
    assert config["training"]["lr"] == pytest.approx(1e-3)
    assert config["data"]["labels"] is None


def test_typos_and_inconsistent_settings_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="clusterng"):
        load_config(None, ["clusterng.method=ward"])
    with pytest.raises(ValueError, match="checkpoint"):
        load_config(None, ["training.train=false"])


@pytest.mark.parametrize("overrides, expected", [
    ([], "vae-k128_e100__ward-euclidean-c55__legion-no_axes-train"),
    (["model.params.beta_max=32", "model.params.k=16", "clustering.method=auto"],
     "vae-k16-beta32_e100__auto-c55__legion-no_axes-train"),
    (["model.type=byol", "model.params.tau=0.99", "data.source=raw", "model.params.f=5"],
     "byol-f5n3s1-k128-tau0.99_e100__ward-euclidean-c55__raw"),
    (["training.train=false", "model.checkpoint=models/x/vae-k16_e100__ward__legion__20261012-1200.pt"],
     "vae-k128_extract-vae-k16_e100__ward-euclidean-c55__legion-no_axes-train"),
    (["name=my run"], "my-run"),
])
def test_run_name_describes_the_config(overrides, expected):
    assert run_name(load_config(None, overrides)) == expected


@pytest.fixture
def ready_images(tmp_path):
    """Four 256x256 'preprocessed' drawings, two of them labelled."""
    folder = tmp_path / "preprocessed" / "set" / "train"
    folder.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for i in range(4):
        image = np.full((256, 256), 255, np.uint8)
        for _ in range(3):
            y = rng.integers(20, 236)
            image[y:y + 3, 30:226] = 0
        Image.fromarray(image).save(folder / f"img{i}.png")
    labels = tmp_path / "labels.csv"
    pd.DataFrame({"ID": ["img0", "img1"], "TypeRev": ["Topf", "Krug"]}).to_csv(labels, index=False)
    return folder, labels


def _config(tmp_path, ready_images, *extra):
    folder, labels = ready_images
    return load_config(None, [
        "experiment=test",
        f"data.preprocessed_path={folder}",
        f"data.labels={labels}",
        "training.epochs=2", "training.batch_size=2",
        "clustering.cluster_num=2", "clustering.min_variance=0",
        *extra,
    ])


def test_run_trains_saves_model_and_logs_everything(tmp_path, ready_images):
    result = run_experiment(config=_config(tmp_path, ready_images))
    run_id = result["run_id"]

    assert run_id.startswith("vae-k128_e2__ward-euclidean-c2__set-train__")
    # Everything in the run folder, each step in its own folder.
    run_dir = result["run_dir"]
    assert run_dir == tmp_path / "default_output" / run_id
    model_dir = run_dir / "FeatureLearningStep" / "VAE"
    assert result["model_path"] == model_dir / "model.pt"
    assert {p.name for p in model_dir.iterdir()} >= {"model.pt", "model.yaml", "features.csv"}
    info = yaml.safe_load((model_dir / "model.yaml").read_text())
    assert info["run_id"] == run_id and info["config"]["model"]["type"] == "vae"
    assert (run_dir / "config.yaml").exists() and (run_dir / "EvaluationStep" / "metrics.json").exists()
    clusters = pd.read_csv(run_dir / "ClusteringStep" / "HierarchicalClustering" / "clusters.csv")
    assert clusters["filename"].tolist() == [f"img{i}.png" for i in range(4)]
    # MLflow run.
    [run] = mlflow.search_runs(experiment_names=["test"], output_format="list")
    assert run.info.run_name == run_id
    assert info["mlflow_run_id"] == run.info.run_id
    assert run.data.tags["model_path"] == str(result["model_path"])
    assert run.data.params["cfg/model/params/k"] == "128"
    assert run.data.metrics["eval/n_labelled"] == 2
    assert "eval/acc" in run.data.metrics and "eval/silhouette" in run.data.metrics
    assert "feature/loss" in run.data.metrics


def test_saved_model_can_be_reused_for_extraction(tmp_path, ready_images):
    trained = run_experiment(config=_config(tmp_path, ready_images))

    extracted = run_experiment(config=_config(
        tmp_path, ready_images, "training.train=false", f"model.checkpoint={trained['model_path']}"))

    assert extracted["model_path"] is None  # nothing new trained
    # Named after the run that saved the checkpoint.
    assert extracted["run_id"].startswith("vae-k128_extract-vae-k128_e2__")
    np.testing.assert_allclose(np.stack(extracted["features"]["features"]),
                               np.stack(trained["features"]["features"]), rtol=1e-5, atol=1e-5)


def test_stop_after_needs_raw_data_for_preprocessing():
    with pytest.raises(ValueError, match="data.source: raw"):
        load_config(None, ["stop_after=preprocessing"])


@pytest.mark.parametrize("overrides, expected", [
    (["stop_after=preprocessing", "data.source=raw"], "prep-be-ax-norm-repair__raw"),
    (["stop_after=features"], "vae-k128_e100__legion-no_axes-train"),
])
def test_run_name_of_partial_runs(overrides, expected):
    assert run_name(load_config(None, overrides)) == expected


def test_stop_after_preprocessing_only_saves_images(tmp_path, ready_images):
    folder, _ = ready_images
    config = _config(
        tmp_path, ready_images,
        "stop_after=preprocessing", "data.source=raw", f"data.raw_path={folder}",
        # No YOLO in tests: only steps that need no model weights.
        "preprocessing.steps=[{Normalizer: {target_size: 128, background_color: [255, 255, 255]}},"
        " {Repair: {skeletonize: false, debug_images: 0}}]",
    )

    result = run_experiment(config=config)

    assert result["run_id"].startswith("prep-norm-repair__raw__")
    steps_dir = result["run_dir"] / "PreprocessingStep"
    assert result["preprocessed_path"] == steps_dir / "Repair"
    saved = sorted(p.name for p in result["preprocessed_path"].iterdir())
    assert saved == [f"img{i}.png" for i in range(4)]
    assert Image.open(result["preprocessed_path"] / "img0.png").size == (128, 128)
    assert result["model_path"] is None and result["features"] is None and result["metrics"] is None
    assert not (result["run_dir"] / "FeatureLearningStep").exists()
    [run] = mlflow.search_runs(experiment_names=["test"], output_format="list")
    assert run.data.tags["preprocessed_output"] == str(result["preprocessed_path"])
    assert run.data.tags["stop_after"] == "preprocessing"


def test_stop_after_features_skips_clustering(tmp_path, ready_images):
    result = run_experiment(config=_config(tmp_path, ready_images, "stop_after=features"))

    assert result["run_id"].startswith("vae-k128_e2__set-train__")
    assert result["model_path"].exists()
    assert len(result["features"]) == 4
    assert result["labels"] is None and result["metrics"] is None
    assert not (result["run_dir"] / "ClusteringStep").exists()
    [run] = mlflow.search_runs(experiment_names=["test"], output_format="list")
    assert not any(key.startswith(("eval/", "clustering/")) for key in run.data.metrics)
