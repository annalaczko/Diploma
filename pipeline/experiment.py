"""Config-driven experiment runs: one YAML file describes a whole run
(data, preprocessing, feature learning model, clustering, evaluation), and
`run_experiment` builds the steps from it, runs them and records everything.

Start runs with run_experiment.py at the repo root; configs/default.yaml
documents every setting. Each run gets an id generated from its config,
e.g. `vae-k16-beta32_e100__ward-euclidean-c55__legion-train__20261012-153000`,
which is used everywhere:
  - MLflow run name (experiment `config["experiment"]`): params, metrics,
    tags (git commit, model file path, ...) and the resolved config.yaml.
  - data/pipeline/<run id>/: config.yaml, features, clusters.csv, metrics.json.
  - <paths.models>/<run id>.pt: the trained model's state_dict, with
    <run id>.yaml next to it (config + MLflow run id), so a model file can
    always be traced back to its run, and fed into a later run as
    `model.checkpoint` (to fine-tune it or extract features with it).
"""

from __future__ import annotations

import copy
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import yaml

import pipeline.steps.step as step_module
from pipeline import tracking
from pipeline.clustering import HierarchicalClustering
from pipeline.evaluation import EvaluationWithGroundTruth, EvaluationWithNoGroundTruth
from pipeline.feature_learning import BYOL, SimCLR, VAE
from pipeline.pipeline import Pipeline
from pipeline.preprocessing import Augmentor, BackgroundEraser, BackgroundEraser2, Normalizer, Repair

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"

MODELS = {"vae": VAE, "simclr": SimCLR, "byol": BYOL}
PREPROCESSING_STEPS = {
    cls.__name__: cls for cls in (BackgroundEraser, BackgroundEraser2, Normalizer, Repair, Augmentor)
}

# Model parameters that go into the generated run name, as (key, label):
# k=16 -> "k16", beta_max=32 -> "beta32", recon_loss="dice" -> "dice".
NAME_KEYS = {
    "vae": [("k", "k"), ("beta_max", "beta"), ("recon_loss", "")],
    "simclr": [("k", "k"), ("projection_dim", "proj")],
    "byol": [("k", "k"), ("tau", "tau")],
}
DEFAULT_FNS = (4, 3, 1)  # f, n, s; only named when changed
# Short step names for preprocessing-only run names.
STEP_LABELS = {"BackgroundEraser": "be", "BackgroundEraser2": "be2", "Normalizer": "norm",
               "Repair": "repair", "Augmentor": "aug"}
STOP_AFTER = (None, "preprocessing", "features")

# Sections whose keys are checked against configs/default.yaml, to catch
# typos. model.params, training and step parameters are passed through to
# the classes as they are, so the classes check those.
CHECKED_SECTIONS = ("data", "preprocessing", "model", "clustering", "paths")


# --------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------- #

class _ConfigLoader(yaml.SafeLoader):
    """SafeLoader that also reads 1e-4 as a float. PyYAML follows YAML 1.1,
    where a float needs a dot (1.0e-4), so 'lr: 1e-4' would be a string."""


_ConfigLoader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(r"^[-+]?(?:[0-9][0-9_]*)?(?:\.[0-9_]*)?[eE][-+]?[0-9]+$"),
    list("-+0123456789."),
)


def _parse_yaml(text: str):
    return yaml.load(text, Loader=_ConfigLoader)


def deep_merge(base: dict, override: dict) -> dict:
    """`base` updated with `override`: nested dicts are merged key by key,
    anything else (values, lists) is replaced."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def set_dotted(config: dict, key: str, value: Any) -> None:
    """Set `config["a"]["b"]` for key 'a.b', creating missing sections."""
    *parents, last = key.split(".")
    node = config
    for part in parents:
        node = node.setdefault(part, {})
        if not isinstance(node, dict):
            raise ValueError(f"Cannot set {key}: {part} is not a section")
    node[last] = value


def load_config(path=None, overrides: Iterable[str] = (), default=DEFAULT_CONFIG) -> dict:
    """configs/default.yaml, merged with the experiment config at `path`
    (if any), then with `overrides` ('key.sub=value', the value parsed as
    YAML, so 32 is an int, 1e-4 a float, null None and [a, b] a list)."""
    defaults = _read_yaml(default)
    config = deep_merge(defaults, _read_yaml(path) if path is not None else {})
    for item in overrides:
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"Overrides look like key.sub=value, got {item!r}")
        set_dotted(config, key.strip(), _parse_yaml(value))
    validate(config, defaults)
    return config


def validate(config: dict, defaults: dict) -> None:
    unknown = set(config) - set(defaults)
    for section in CHECKED_SECTIONS:
        unknown |= {f"{section}.{key}" for key in set(config.get(section) or {}) - set(defaults[section])}
    if unknown:
        raise ValueError(f"Unknown config keys (typo?): {sorted(unknown)}")
    if config["data"]["source"] not in ("preprocessed", "raw"):
        raise ValueError("data.source must be 'preprocessed' or 'raw'")
    if config["model"]["type"] not in MODELS:
        raise ValueError(f"model.type must be one of {sorted(MODELS)}")
    if config["stop_after"] not in STOP_AFTER:
        raise ValueError(f"stop_after must be one of {STOP_AFTER}")
    if config["stop_after"] == "preprocessing" and config["data"]["source"] != "raw":
        raise ValueError("stop_after: preprocessing needs data.source: raw (ready images need no preprocessing)")
    if (config["stop_after"] != "preprocessing"
            and not config["training"]["train"] and not config["model"]["checkpoint"]):
        raise ValueError("training.train: false only extracts features, so it needs model.checkpoint")
    for entry in config["preprocessing"]["steps"] or []:
        if not (isinstance(entry, dict) and len(entry) == 1 and next(iter(entry)) in PREPROCESSING_STEPS):
            raise ValueError(
                f"Each preprocessing step looks like '- StepName: {{params}}' with StepName one of "
                f"{sorted(PREPROCESSING_STEPS)}, got {entry!r}"
            )


def _read_yaml(path) -> dict:
    with open(path, encoding="utf-8") as f:
        return _parse_yaml(f.read()) or {}


def resolve_path(path) -> Path | None:
    """Relative config paths are relative to the repo root, so runs work
    from any working directory."""
    if path is None:
        return None
    path = Path(path).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


# --------------------------------------------------------------------- #
# Run names
# --------------------------------------------------------------------- #

def run_name(config: dict) -> str:
    """Readable name from the settings that tell runs apart:
    <model>-<params>_<training>__<clustering>__<data>, e.g.
    'vae-k16-beta32_e100__ward-euclidean-c55__legion-train'. Without the
    clustering part for stop_after: features, and 'prep-<steps>__raw' (e.g.
    'prep-be2-norm-repair__raw') for stop_after: preprocessing.
    `config["name"]`, if set, is used instead."""
    if config.get("name"):
        return _safe(config["name"])
    if config["stop_after"] == "preprocessing":
        steps = [STEP_LABELS[next(iter(entry))] for entry in config["preprocessing"]["steps"] or []]
        return _safe(f"prep-{'-'.join(steps)}__{_data_label(config)}")

    model = config["model"]
    params = model.get("params") or {}
    parts = [model["type"]]
    fns = tuple(params.get(key, default) for key, default in zip("fns", DEFAULT_FNS))
    if fns != DEFAULT_FNS:
        parts.append("f{}n{}s{}".format(*fns))
    parts += [f"{label}{_fmt(params[key])}" for key, label in NAME_KEYS[model["type"]] if key in params]

    training = config["training"]
    if training["train"]:
        trained = f"e{training['epochs']}" + ("-ft" if model.get("checkpoint") else "")
    else:
        # A checkpoint saved by an earlier run is named <run name>__<time>;
        # its first part is enough to say which model it was.
        trained = "extract-" + Path(model["checkpoint"]).stem.split("__")[0]

    if config["stop_after"] == "features":
        return _safe(f"{'-'.join(parts)}_{trained}__{_data_label(config)}")
    clustering = config["clustering"]
    method = "auto" if clustering["method"] == "auto" else f"{clustering['method']}-{clustering['metric']}"
    return _safe(f"{'-'.join(parts)}_{trained}__{method}-c{clustering['cluster_num']}__{_data_label(config)}")


def _data_label(config: dict) -> str:
    data = config["data"]
    if data["source"] == "raw":
        return "raw"
    path = resolve_path(data["preprocessed_path"])
    try:
        parts = path.relative_to(resolve_path(config["paths"]["preprocessed"])).parts
    except ValueError:
        parts = path.parts[-2:]
    return "-".join(parts)


def _fmt(value) -> str:
    return f"{value:g}" if isinstance(value, float) else str(value)


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", str(name)).strip("-")


# --------------------------------------------------------------------- #
# Building the steps
# --------------------------------------------------------------------- #

def build_preprocessing_steps(config: dict, run_id: str) -> list:
    """The configured preprocessing steps, in order. The last image-producing
    step (not an Augmentor) writes to <paths.preprocessed>/<run id>."""
    steps = []
    for entry in config["preprocessing"]["steps"] or []:
        (class_name, params), = entry.items()
        # YAML has no tuples; e.g. Normalizer's background_color must be one.
        params = {key: tuple(value) if isinstance(value, list) else value for key, value in (params or {}).items()}
        if "model" in params:  # BackgroundEraser's YOLO weights
            params["model"] = str(resolve_path(params["model"]))
        steps.append(PREPROCESSING_STEPS[class_name](run_id=run_id, **params))
    image_steps = [step for step in steps if not isinstance(step, Augmentor)]
    if image_steps:
        image_steps[-1].output_path = str(resolve_path(config["paths"]["preprocessed"]) / run_id)
    return steps


def build_model(config: dict, run_id: str, device) -> torch.nn.Module:
    """The feature-learning model, with `model.checkpoint` loaded if given
    (a plain state_dict, as saved by `save_model` or torch.save)."""
    model_config = config["model"]
    model = MODELS[model_config["type"]](run_id=run_id, **(model_config.get("params") or {})).to(device)
    if model_config.get("checkpoint"):
        state = torch.load(resolve_path(model_config["checkpoint"]), map_location=device, weights_only=True)
        model.load_state_dict(state)
    return model


def standardize(vectors: np.ndarray, min_variance: float) -> np.ndarray:
    """Z-score every dimension, after dropping the near-constant ones
    (inactive latent dimensions would otherwise be blown up to unit variance)."""
    keep = vectors.var(axis=0) > min_variance
    if not keep.any():
        raise ValueError(f"No feature dimension has variance above {min_variance}")
    kept = vectors[:, keep]
    return (kept - kept.mean(axis=0)) / kept.std(axis=0)


def clustering_factory(config: dict, run_id: str, built: dict):
    """Factory for `Pipeline`: builds HierarchicalClustering from the
    features CSV as configured, and keeps it in `built["step"]`."""
    settings = config["clustering"]

    def factory(features_csv):
        auto = settings["method"] == "auto"
        step = HierarchicalClustering(
            input_path=features_csv,
            cluster_num=settings["cluster_num"],
            method=None if auto else settings["method"],
            metric=settings["metric"],
            run_id=run_id,
        )
        n_features = step.vectors.shape[1]
        if settings["standardize"]:
            step.vectors = standardize(step.vectors, settings["min_variance"])
        tracking.log_metrics({"n_dims_used": step.vectors.shape[1], "n_dims_total": n_features},
                             prefix="clustering/")
        if auto:
            step.compare_all()
        built["step"] = step
        return step

    return factory


# --------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------- #

def evaluate(clustering: HierarchicalClustering, predicted: np.ndarray, labels_path) -> dict:
    """Silhouette on every clustered image, plus ACC/ARI/NMI on the images
    that have a label in `labels_path` (ID = file name without extension
    and without an Augmentor's _aug<i> suffix, TypeRev = class)."""
    metrics = {}
    if len(np.unique(predicted)) > 1:
        metrics.update(EvaluationWithNoGroundTruth(clustering.vectors, predicted).run())
    if labels_path is None or clustering.filenames is None:
        return metrics

    truth = pd.read_csv(labels_path)
    label_of = dict(zip(truth["ID"].astype(str).str.strip(), truth["TypeRev"]))
    ids = (pd.Series(clustering.filenames).astype(str)
           .str.rsplit(".", n=1).str[0].str.replace(r"_aug\d+$", "", regex=True))
    y_true = ids.map(label_of)
    labelled = y_true.notna().to_numpy()
    metrics["n_labelled"] = int(labelled.sum())
    metrics["n_classes"] = int(y_true[labelled].nunique())
    tracking.log_metrics({"n_labelled": metrics["n_labelled"], "n_classes": metrics["n_classes"]},
                         prefix="eval/")
    if labelled.any():
        metrics.update(EvaluationWithGroundTruth(y_true[labelled].to_numpy(), predicted[labelled]).run())
    else:
        print(f"No clustered image has a label in {labels_path}; skipping ACC/ARI/NMI")
    return metrics


# --------------------------------------------------------------------- #
# Running
# --------------------------------------------------------------------- #

def save_model(model, config: dict, run_id: str, mlflow_run_id: str | None) -> Path:
    """Save the state_dict to <paths.models>/<run id>.pt, and the resolved
    config plus the MLflow run id to <run id>.yaml next to it."""
    folder = resolve_path(config["paths"]["models"])
    folder.mkdir(parents=True, exist_ok=True)
    model_path = folder / f"{run_id}.pt"
    torch.save(model.state_dict(), model_path)
    info = {
        "run_id": run_id,
        "mlflow_experiment": config["experiment"],
        "mlflow_run_id": mlflow_run_id,
        "created": datetime.now().isoformat(timespec="seconds"),
        "model_type": config["model"]["type"],
        "model_params": config["model"].get("params") or {},
        "started_from": config["model"].get("checkpoint"),
        "config": config,
    }
    _write_yaml(info, folder / f"{run_id}.yaml")
    return model_path


def run_experiment(config_path=None, overrides: Iterable[str] = (), dry_run: bool = False,
                   config: dict | None = None) -> dict:
    """Run one experiment end to end (see the module docstring).

    Args:
        config_path: experiment YAML (merged over configs/default.yaml).
        overrides: 'key.sub=value' strings applied on top.
        dry_run: only print the resolved config and the run id.
        config: an already loaded config, instead of `config_path`.

    `config["stop_after"]` ends the run early: after 'preprocessing'
    (only the preprocessed images are produced) or after 'features' (model
    trained/saved and features extracted, no clustering or evaluation).

    Returns:
        dict with 'run_id', 'config', and unless `dry_run`: 'run_dir',
        'preprocessed_path' (None unless data.source is raw), 'model_path'
        (None if not trained), 'features', 'labels', 'metrics' (the last
        ones None when the run stopped before them).
    """
    config = config if config is not None else load_config(config_path, overrides)
    run_id = f"{run_name(config)}__{datetime.now():%Y%m%d-%H%M%S}"
    if dry_run:
        print(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
        print(f"Run id: {run_id}")
        return {"run_id": run_id, "config": config}

    run_dir = Path(step_module.DEFAULT_OUTPUT_ROOT) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_yaml(config, run_dir / "config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = config["data"]
    raw = data["source"] == "raw"
    print(f"Run {run_id}\n  outputs: {run_dir}")

    stop_after = config["stop_after"]
    with tracking.start_run(config["experiment"], run_name=run_id) as mlflow_run:
        _log_run_info(config, config_path, run_id, run_dir)

        model = build_model(config, run_id, device) if stop_after != "preprocessing" else None
        built = {}
        steps = build_preprocessing_steps(config, run_id) if raw else []
        pipe = Pipeline(steps, feature_learning_step=model,
                        clustering_factory=clustering_factory(config, run_id, built) if stop_after is None else None)

        training = dict(config["training"])
        train = training.pop("train") and model is not None
        feature_kwargs = training if train else {"batch_size": training.get("batch_size", 32)}
        result = pipe.run(
            str(resolve_path(data["raw_path"] if raw else data["preprocessed_path"])),
            # Ready images: disk mode reads the folder directly, no copy in RAM.
            mode=config["preprocessing"]["mode"] if raw else "disk",
            preprocess_batch_size=config["preprocessing"]["batch_size"],
            feature_mode="train_model" if train else "extract_features",
            save_preprocessed="last" if raw else None,
            **(feature_kwargs if model is not None else {}),
        )
        image_steps = [step for step in steps if not isinstance(step, Augmentor)]
        preprocessed_path = Path(image_steps[-1].output_path) if image_steps else None
        tracking.set_tags({"preprocessed_output": preprocessed_path})

        model_path = save_model(model, config, run_id, mlflow_run.info.run_id) if train else None
        tracking.set_tags({"model_path": model_path})

        predicted, metrics = result.get("labels"), None
        if stop_after is None:
            clustering = built["step"]
            metrics = evaluate(clustering, predicted, resolve_path(data["labels"]))
            names = clustering.filenames if clustering.filenames is not None else range(len(predicted))
            pd.DataFrame({"filename": names, "cluster": predicted}).to_csv(run_dir / "clusters.csv", index=False)
            (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))

    print(f"Done: {run_id}" + (f" (stopped after {stop_after})" if stop_after else ""))
    if preprocessed_path:
        print(f"  preprocessed images: {preprocessed_path}")
    if model_path:
        print(f"  model:   {model_path}")
    if metrics:
        print("  metrics: " + ", ".join(f"{key}={value:.4g}" for key, value in metrics.items()))
    return {"run_id": run_id, "config": config, "run_dir": run_dir, "preprocessed_path": preprocessed_path,
            "model_path": model_path, "features": result.get("features"), "labels": predicted,
            "metrics": metrics}


def _log_run_info(config: dict, config_path, run_id: str, run_dir: Path) -> None:
    """Tags that locate everything belonging to the run, the whole config
    as 'cfg/...' params, and the config file itself."""
    commit, dirty = _git_state()
    tracking.set_tags({
        "run_id": run_id,
        "config_file": config_path,
        "output_dir": run_dir,
        "stop_after": config["stop_after"],
        **({} if config["stop_after"] == "preprocessing" else {
            "model_type": config["model"]["type"],
            "started_from": config["model"].get("checkpoint"),
        }),
        "data": resolve_path(config["data"]["raw_path" if config["data"]["source"] == "raw" else "preprocessed_path"]),
        "git_commit": commit,
        "git_dirty": dirty,
    })
    tracking.log_params(_flatten(config), prefix="cfg/")
    tracking.log_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), "config.yaml")


def _flatten(config: dict, prefix: str = "") -> dict:
    """{'a': {'b': 1}} -> {'a/b': 1}; lists (preprocessing steps) as JSON."""
    flat = {}
    for key, value in config.items():
        if isinstance(value, dict) and value:
            flat.update(_flatten(value, f"{prefix}{key}/"))
        elif isinstance(value, list):
            flat[f"{prefix}{key}"] = json.dumps(value)
        else:
            flat[f"{prefix}{key}"] = value
    return flat


def _git_state() -> tuple[str | None, bool | None]:
    """Current commit and whether there are uncommitted changes."""
    try:
        commit = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                                capture_output=True, text=True, check=True).stdout.strip()
        status = subprocess.run(["git", "-C", str(REPO_ROOT), "status", "--porcelain"],
                                capture_output=True, text=True, check=True).stdout
        return commit, bool(status.strip())
    except (OSError, subprocess.CalledProcessError):
        return None, None


def _write_yaml(data, path: Path) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
