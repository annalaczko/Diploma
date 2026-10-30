"""Tests for config-driven experiment runs (pipeline/experiment.py)."""

from pathlib import Path
from types import SimpleNamespace

import mlflow
import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from PIL import Image
from transformers import BatchEncoding

from pipeline.feature_learning import DINOv3
from pipeline.experiment import (feature_importance, feature_names, load_config, run_experiment, run_name,
                                 run_sweep, sweep_configs)


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
    (["model.type=dinov3"], "dinov3__ward-euclidean-c55__legion-no_axes-train"),
    (["model.type=dinov3", "model.params.model_name=facebook/dinov3-vitl16-pretrain-lvd1689m"],
     "dinov3-vitl16__ward-euclidean-c55__legion-no_axes-train"),
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


def test_switching_model_family_drops_the_params_that_do_not_apply(tmp_path):
    handcrafted = _write(tmp_path / "hc.yaml", {"model": {"type": "handcrafted", "params": {"efd_harmonics": 5}}})

    assert load_config(None, ["model.type=dinov3"])["model"]["params"] == {}
    assert load_config(handcrafted, ["model.type=dinov3"])["model"]["params"] == {}
    with pytest.raises(ValueError, match="not trained here"):
        load_config(None, ["model.type=dinov3", "model.checkpoint=x.pt"])


class _FakeDINOv3:
    """Stands in for the Hugging Face processor and backbone: the 'CLS
    token' is the top half's mean per channel, so no weights are downloaded."""

    def __call__(self, images=None, return_tensors=None, **batch):
        if images is not None:  # the processor
            pixels = torch.stack([torch.from_numpy(np.asarray(image, np.float32)).permute(2, 0, 1)
                                  for image in images])
            return BatchEncoding({"pixel_values": pixels})
        pooled = batch["pixel_values"][:, :, :128].mean(dim=(2, 3))
        return SimpleNamespace(last_hidden_state=torch.stack([pooled, pooled * 0], dim=1))


def test_dinov3_run_embeds_clusters_and_evaluates(tmp_path, ready_images, monkeypatch):
    fake = _FakeDINOv3()
    monkeypatch.setattr(DINOv3, "_load", lambda self: (fake, fake))
    result = run_experiment(config=_config(tmp_path, ready_images, "model.type=dinov3"))

    assert result["run_id"].startswith("dinov3__ward-euclidean-c2__set-train__")
    assert result["model_path"] is None  # nothing trained, nothing saved
    model_dir = result["run_dir"] / "FeatureLearningStep" / "DINOv3"
    features = pd.read_csv(model_dir / "features.csv")
    assert features["filename"].tolist() == [f"img{i}.png" for i in range(4)]
    assert len(features["features"][0].split()) == 3
    assert (result["run_dir"] / "EvaluationStep" / "metrics.json").exists()


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


def test_extract_path_trains_on_one_folder_and_clusters_another(tmp_path, ready_images):
    folder, _ = ready_images
    test_folder = folder.parent / "test"
    test_folder.mkdir()
    for name in ("img0.png", "img1.png", "img2.png"):
        Image.open(folder / name).transpose(Image.Transpose.FLIP_LEFT_RIGHT).save(test_folder / f"t_{name}")

    result = run_experiment(config=_config(tmp_path, ready_images, f"data.extract_path={test_folder}"))

    assert result["run_id"].startswith("vae-k128_e2__ward-euclidean-c2__set-train-to-test__")
    model_dir = result["run_dir"] / "FeatureLearningStep" / "VAE"
    assert pd.read_csv(model_dir / "features_train.csv")["filename"].tolist() == [f"img{i}.png" for i in range(4)]
    assert result["features_path"] == model_dir / "features.csv"
    assert result["features"]["filename"].tolist() == ["t_img0.png", "t_img1.png", "t_img2.png"]
    clusters = pd.read_csv(result["run_dir"] / "ClusteringStep" / "HierarchicalClustering" / "clusters.csv")
    assert clusters["filename"].tolist() == ["t_img0.png", "t_img1.png", "t_img2.png"]
    assert result["model_path"].exists()


def test_extract_path_without_training_only_reads_that_folder(tmp_path, ready_images):
    folder, _ = ready_images
    trained = run_experiment(config=_config(tmp_path, ready_images, "stop_after=features"))
    test_folder = folder.parent / "test"
    test_folder.mkdir()
    Image.open(folder / "img0.png").save(test_folder / "a.png")
    Image.open(folder / "img1.png").save(test_folder / "b.png")
    Image.open(folder / "img2.png").save(test_folder / "c.png")

    result = run_experiment(config=_config(
        tmp_path, ready_images, "training.train=false", f"model.checkpoint={trained['model_path']}",
        f"data.extract_path={test_folder}"))

    assert "__set-test__" in result["run_id"]
    assert result["features"]["filename"].tolist() == ["a.png", "b.png", "c.png"]


# --------------------------------------------------------------------- #
# Sweeps
# --------------------------------------------------------------------- #

def test_sweep_has_one_config_per_combination_model_settings_first(tmp_path):
    config = load_config(_write(tmp_path / "exp.yaml", {"sweep": {
        "clustering": {"cluster_num": [20, 30]},  # nested keys work too
        "model.params.k": [16, 32],
    }}))

    combinations, skipped = sweep_configs(config)

    assert skipped == []
    assert [swept for swept, _ in combinations] == [
        {"model.params.k": 16, "clustering.cluster_num": 20},
        {"model.params.k": 16, "clustering.cluster_num": 30},
        {"model.params.k": 32, "clustering.cluster_num": 20},
        {"model.params.k": 32, "clustering.cluster_num": 30},
    ]
    _, last = combinations[-1]
    assert last["model"]["params"]["k"] == 32 and last["clustering"]["cluster_num"] == 30
    assert last["sweep"] == {} and config["model"]["params"]["k"] == 128  # the config itself untouched


def test_sweep_leaves_out_combinations_that_cannot_run():
    config = load_config(None, [
        "sweep={clustering.method: [ward, average, auto], clustering.metric: [euclidean, cosine]}"])

    combinations, skipped = sweep_configs(config)

    assert [(swept["clustering.method"], swept["clustering.metric"]) for swept, _ in combinations] == [
        ("ward", "euclidean"), ("average", "euclidean"), ("average", "cosine"), ("auto", "euclidean")]
    assert [why.split(" ")[0] for _, why in skipped] == ["method", "same"]  # ward+cosine, auto+cosine


def test_impossible_clustering_settings_are_rejected():
    with pytest.raises(ValueError, match="needs metric euclidean"):
        load_config(None, ["clustering.metric=cosine"])  # default method: ward
    with pytest.raises(ValueError, match="clustering.method"):
        load_config(None, ["clustering.method=wardd"])
    with pytest.raises(ValueError, match="No combination"):
        load_config(None, ["sweep={clustering.metric: [cosine, cityblock]}"])
    # The sweep may fix the base config's clustering.
    load_config(None, ["clustering.metric=cosine", "sweep={clustering.method: [average, single]}"])


def test_bad_sweeps_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="clusterng"):
        load_config(None, ["sweep={clusterng.cluster_num: [20, 30]}"])
    with pytest.raises(ValueError, match="non-empty list"):
        load_config(None, ["sweep={clustering.cluster_num: 20}"])
    with pytest.raises(ValueError, match="same setting"):
        load_config(None, ["sweep={model.params.k: [16, 32]}", "model.params.k=8"])
    with pytest.raises(ValueError, match="would not change anything"):
        load_config(None, ["sweep={clustering.cluster_num: [20, 30]}", "stop_after=features"])


def test_sweep_trains_once_per_model_and_reclusters_its_features(tmp_path, ready_images):
    config = _config(tmp_path, ready_images,
                     "sweep={training.lr: [1.0e-3, 1.0e-4], clustering.method: [ward, average]}")

    rows = run_sweep(config=config)

    assert [row["status"] for row in rows] == ["ok"] * 4
    runs = [tmp_path / "default_output" / row["run_id"] for row in rows]
    trained = [run for run in runs if (run / "FeatureLearningStep").exists()]
    assert trained == [runs[0], runs[2]]  # one model per learning rate
    # lr is not in the generated name, so the sweep adds it.
    assert rows[1]["run_id"].startswith("vae-k128_e2__average-euclidean-c2__set-train__lr0.001__")
    reused = yaml.safe_load((runs[1] / "config.yaml").read_text())
    assert reused["data"]["features"] == str(runs[0] / "FeatureLearningStep" / "VAE" / "features.csv")
    for run in runs:
        assert (run / "EvaluationStep" / "metrics.json").exists()

    [sweep_dir] = (tmp_path / "default_output").glob("sweep-*")
    summary = pd.read_csv(sweep_dir / "summary.csv")
    assert summary["clustering.method"].tolist() == ["ward", "average"] * 2
    assert "silhouette" in summary.columns
    mlflow_runs = mlflow.search_runs(experiment_names=["test"], output_format="list")
    assert len(mlflow_runs) == 4
    assert {run.data.tags["sweep"] for run in mlflow_runs} == {sweep_dir.name}


def test_failed_features_run_skips_its_reclusterings(tmp_path, ready_images, monkeypatch):
    import pipeline.experiment as experiment
    real_build_model = experiment.build_model

    def build_model(config, run_id, device):
        if config["model"]["params"]["k"] == 16:
            raise RuntimeError("out of memory")
        return real_build_model(config, run_id, device)

    monkeypatch.setattr(experiment, "build_model", build_model)
    config = _config(tmp_path, ready_images,
                     "sweep={model.params.k: [16, 8], clustering.method: [ward, average]}")

    rows = run_sweep(config=config)

    assert [row["status"].split(":")[0] for row in rows] == ["failed", "skipped", "ok", "ok"]


def test_feature_importance_ranks_the_separating_dimension_first():
    rng = np.random.default_rng(0)
    labels = np.repeat([1, 2], 50)
    separating = labels * 5.0 + rng.normal(0, 0.1, 100)
    noise = rng.normal(0, 1, 100)
    table = feature_importance(np.column_stack([noise, separating]), labels, ["noise", "size"])

    assert table["feature"].tolist() == ["size", "noise"]
    assert table["eta2"][0] > 0.99 and table["eta2"][1] < 0.1


def test_feature_names_from_feature_columns_txt(tmp_path):
    csv = tmp_path / "features.csv"
    (tmp_path / "feature_columns.txt").write_text("height\nwidth\n")
    assert feature_names(csv, 2) == ["height", "width"]
    assert feature_names(csv, 3) == ["dim_0", "dim_1", "dim_2"]  # length mismatch -> generic names
