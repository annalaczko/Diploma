"""Config-driven experiment runs: one YAML file describes a whole run
(data, preprocessing, feature learning model, clustering, evaluation), and
`run_experiment` builds the steps from it, runs them and records everything.

Start runs with run_experiment.py at the repo root; configs/default.yaml
documents every setting. Each run gets an id generated from its config,
e.g. `vae-k16-beta32_e100__ward-euclidean-c55__legion-no_axes-train__20261012-153000`,
which is used everywhere:
  - MLflow run name (experiment `config["experiment"]`): params, metrics,
    tags (git commit, model file path, ...) and the resolved config.yaml.
  - data/pipeline/<run id>/: everything the run writes, each step into its
    own folder <Stage>/<Step>/ (see `Step`):
      config.yaml                                   the resolved config
      PreprocessingStep/<Step>/                     each step's images (raw runs:
                                                    the last one, in memory mode),
                                                    extracted_metadata.csv, _debug/, _svg/
      FeatureLearningStep/<Model>/model.pt          the trained state_dict, with
                                  model.yaml        config + MLflow run id, and
                                  features.csv      the features
      ClusteringStep/HierarchicalClustering/clusters.csv
      EvaluationStep/metrics.json
    A model file can thus always be traced back to its run, and fed into a
    later run as `model.checkpoint` (to fine-tune it or extract features).

A config with a `sweep` (settings -> lists of values) is run by `run_sweep`:
one such run per combination, training each model only once - combinations
that differ only in clustering settings cluster its features.csv again
(`data.features`). data/pipeline/<sweep id>/summary.csv compares them.
"""

from __future__ import annotations

import copy
import itertools
import json
import re
import subprocess
import traceback
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
from pipeline.feature_learning import BYOL, DINOv3, HandcraftedFeatures, SimCLR, VAE
from pipeline.feature_learning.dinov3 import short_model_name
from pipeline.feature_learning.handcrafted_features import FEATURE_GROUPS
from pipeline.pipeline import Pipeline
from pipeline.preprocessing import (
    Augmentor, AxisRenderer, BackgroundEraser, Normalizer, Repair, Vectorizer,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"
# Ready images are usually copied here by hand; their path below it names
# the data in run names (data/processed/legion/no_axes/train -> legion-no_axes-train).
PROCESSED_ROOT = REPO_ROOT / "data" / "processed"

MODELS = {"vae": VAE, "simclr": SimCLR, "byol": BYOL, "handcrafted": HandcraftedFeatures, "dinov3": DINOv3}
# Feature steps with their own params, not the learned models' shared
# encoder ones, and not trained here - measuring (handcrafted) or a frozen
# pretrained backbone (dinov3) - except dinov3 with `finetune: true`, which
# trains like a learned model (see `_learned`).
MEASURED = {"handcrafted", "dinov3"}
PREPROCESSING_STEPS = {
    cls.__name__: cls
    for cls in (BackgroundEraser, Normalizer, Repair, Vectorizer, AxisRenderer, Augmentor)
}

# Model parameters that go into the generated run name, as (key, label):
# k=16 -> "k16", beta_max=32 -> "beta32", recon_loss="dice" -> "dice".
NAME_KEYS = {
    "vae": [("k", "k"), ("beta_max", "beta"), ("recon_loss", "")],
    "simclr": [("k", "k"), ("projection_dim", "proj")],
    "byol": [("k", "k"), ("tau", "tau")],
    "handcrafted": [("efd_harmonics", "efd")],
    "dinov3": [("model_name", ""), ("finetune_method", ""), ("unfreeze_blocks", "u")],
}
# How a parameter's value is written in run names, where not as it is.
NAME_FORMATS = {"model_name": short_model_name}
DEFAULT_FNS = (4, 3, 1)  # f, n, s; only named when changed
# Short step names for preprocessing-only run names.
STEP_LABELS = {"BackgroundEraser": "be", "Normalizer": "norm",
               "Repair": "repair", "Vectorizer": "vec", "AxisRenderer": "axes", "Augmentor": "aug"}
STOP_AFTER = (None, "preprocessing", "features")

# Sections whose keys are checked against configs/default.yaml, to catch
# typos. model.params, training and step parameters are passed through to
# the classes as they are, so the classes check those.
CHECKED_SECTIONS = ("data", "preprocessing", "model", "clustering")


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


def _flatten_sweep(sweep: dict, prefix: str = "") -> dict:
    """The sweep as {'a.b': [values]}, whether written with dotted keys
    ('model.params.k: [16, 32]') or nested ones ('model: {params: {k: [16, 32]}}')."""
    if not isinstance(sweep, dict):
        raise ValueError(f"sweep must map settings to lists of values, got {sweep!r}")
    flat = {}
    for key, value in sweep.items():
        if isinstance(value, dict):
            flat.update(_flatten_sweep(value, f"{prefix}{key}."))
        elif isinstance(value, list) and value:
            flat[f"{prefix}{key}"] = value
        else:
            raise ValueError(f"sweep {prefix}{key} needs a non-empty list of values, got {value!r}"
                             " (a list-valued setting is a list of lists: [[0, 0], [5, 5]])")
    if any(key.split(".")[0] == "sweep" for key in flat):
        raise ValueError("A sweep cannot change the sweep")
    return flat


def sweep_configs(config: dict) -> tuple[list[tuple[dict, dict]], list[tuple[dict, str]]]:
    """Every combination of the sweep's values, as (the swept settings,
    the config with them set and without the sweep). Combinations sharing
    everything but clustering settings come one after the other.

    Combinations that cannot run (e.g. method ward with metric cosine) or
    would repeat an earlier one (method auto ignores the metric) are left
    out, and returned as (the swept settings, why) second."""
    sweep = config.get("sweep") or {}
    keys = sorted(sweep, key=_is_downstream)  # stable: upstream keys first, as written
    base = copy.deepcopy(config)
    base["sweep"] = {}
    combinations, skipped, seen = [], [], {}
    for values in itertools.product(*(sweep[key] for key in keys)):
        swept = dict(zip(keys, values))
        combination = copy.deepcopy(base)
        for key, value in swept.items():
            set_dotted(combination, key, copy.deepcopy(value))
        problem = clustering_problem(combination["clustering"])
        if problem:
            skipped.append((swept, problem))
            continue
        effective = {key: value for key, value in swept.items()
                     if not (key == "clustering.metric" and combination["clustering"]["method"] == "auto")}
        same = json.dumps(effective, sort_keys=True, default=str)
        if same in seen:
            skipped.append((swept, f"same run as {_describe(seen[same])} (method auto ignores the metric)"))
            continue
        seen[same] = swept
        combinations.append((swept, combination))
    return combinations, skipped


def clustering_problem(clustering: dict) -> str | None:
    """Why these clustering settings cannot run, or None if they can."""
    method, metric = clustering["method"], clustering["metric"]
    if method != "auto" and method not in HierarchicalClustering.METHODS:
        return f"clustering.method must be auto or one of {list(HierarchicalClustering.METHODS)}, got {method!r}"
    if method != "auto" and metric not in HierarchicalClustering.METRICS:
        return f"clustering.metric must be one of {list(HierarchicalClustering.METRICS)}, got {metric!r}"
    if method in HierarchicalClustering.EUCLIDEAN_ONLY and metric != "euclidean":
        return f"method {method} needs metric euclidean, got {metric}"
    cluster_num = clustering["cluster_num"]
    if isinstance(cluster_num, bool) or not isinstance(cluster_num, int) or cluster_num < 1:
        return f"clustering.cluster_num must be a positive whole number, got {cluster_num!r}"
    return None


def _describe(swept: dict) -> str:
    return ", ".join(f"{key}={_fmt(value)}" for key, value in swept.items())


def _is_downstream(key: str) -> bool:
    """Settings that need no new features: clustering (and evaluation)."""
    return key.split(".")[0] == "clustering"


def load_config(path=None, overrides: Iterable[str] = (), default=DEFAULT_CONFIG) -> dict:
    """configs/default.yaml, merged with the experiment config at `path`
    (if any), then with `overrides` ('key.sub=value', the value parsed as
    YAML, so 32 is an int, 1e-4 a float, null None and [a, b] a list)."""
    defaults = _read_yaml(default)
    own = _read_yaml(path) if path is not None else {}
    own_type = (own.get("model") or {}).get("type")
    if own_type is not None and _other_kind(defaults["model"]["type"], own_type):
        # The default model.params (f/n/s/k of the shared encoder) don't
        # apply to a feature step not trained here, nor the other way round.
        defaults["model"]["params"] = {}
    config = deep_merge(defaults, own)
    parsed = []
    for item in overrides:
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"Overrides look like key.sub=value, got {item!r}")
        parsed.append((key.strip(), _parse_yaml(value)))
    # A new model.type first: switching to a model with other params
    # (learned <-> handcrafted <-> dinov3), the params set so far don't apply.
    for key, value in parsed:
        if key == "model.type" and _other_kind(config["model"]["type"], value):
            config["model"]["params"] = {}
            config["model"]["type"] = value
    for key, value in parsed:
        set_dotted(config, key, value)
    config["sweep"] = _flatten_sweep(config.get("sweep") or {})
    for key, _ in parsed:
        if key.split(".")[0] == "sweep":
            continue
        # The same setting, or one containing the other (model.params vs model.params.k).
        clash = [swept for swept in config["sweep"]
                 if (swept + ".").startswith(key + ".") or key.startswith(swept + ".")]
        if clash:
            raise ValueError(f"--set {key} and sweep {clash[0]} set the same setting")
    if config["sweep"]:
        # The base config's own clustering may be one the sweep replaces.
        validate(config, defaults, check_clustering=False)
        combinations, skipped = sweep_configs(config)
        if not combinations:
            raise ValueError("No combination of the sweep can run: "
                             + "; ".join(f"{_describe(swept)}: {why}" for swept, why in skipped))
        for _, combination in combinations:
            validate(combination, defaults)
    else:
        validate(config, defaults)
    return config


def _other_kind(old_type, new_type) -> bool:
    """Whether a model type switch changes which params apply: the learned
    models share the encoder params f/n/s/k, each MEASURED one has its own."""
    def family(model_type):
        return model_type if model_type in MEASURED else "learned"
    return family(old_type) != family(new_type)


def validate(config: dict, defaults: dict, check_clustering: bool = True) -> None:
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
    problem = clustering_problem(config["clustering"]) if check_clustering else None
    if problem:
        raise ValueError(problem)
    features = config["data"].get("features")
    if features and config["stop_after"]:
        raise ValueError("data.features only clusters existing features: remove stop_after")
    if features and config["data"].get("extract_path"):
        raise ValueError("data.features already are features: remove data.extract_path")
    if config["data"].get("extract_path") and config["stop_after"] == "preprocessing":
        raise ValueError("data.extract_path is where features come from: stop_after: preprocessing extracts none")
    # Sections the run never gets to (or, with data.features, skips).
    unused = {"preprocessing": ("model", "training", "clustering"), "features": ("clustering",)}.get(
        config["stop_after"], ("model", "training") if features else ())
    pointless = [key for key in config.get("sweep") or {} if key.split(".")[0] in unused]
    if pointless:
        raise ValueError(f"sweep {pointless} would not change anything: this run does not get to "
                         f"{'/'.join(unused)}")
    measured = not _learned(config["model"])
    if measured and config["model"]["checkpoint"]:
        raise ValueError(f"model.type: {config['model']['type']} is not trained here: remove model.checkpoint")
    if (config["stop_after"] != "preprocessing" and not measured and not features
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
    'vae-k16-beta32_e100__ward-euclidean-c55__legion-no_axes-train'. Without the
    clustering part for stop_after: features, and 'prep-<steps>__raw' (e.g.
    'prep-be-ax-norm-repair__raw') for stop_after: preprocessing.
    `config["name"]`, if set, is used instead."""
    if config.get("name"):
        return _safe(config["name"])
    if config["stop_after"] == "preprocessing":
        steps = []
        for entry in config["preprocessing"]["steps"] or []:
            (class_name, params), = entry.items()
            params = params or {}
            label = STEP_LABELS[class_name]
            if params.get("record_axes"):  # BackgroundEraser recording the axes
                label += "-ax"
            if params.get("axes") == "draw":  # Vectorizer drawing them
                label += "-axes"
            steps.append(label)
        return _safe(f"prep-{'-'.join(steps)}__{_data_label(config)}")

    clustering = config["clustering"]
    method = "auto" if clustering["method"] == "auto" else f"{clustering['method']}-{clustering['metric']}"
    if config["data"].get("features"):
        model_part, data_part = _features_label(config["data"]["features"])
        return _safe(f"{model_part}__{method}-c{clustering['cluster_num']}__{data_part or _data_label(config)}")

    model = config["model"]
    params = model.get("params") or {}
    parts = [model["type"]]
    fns = tuple(params.get(key, default) for key, default in zip("fns", DEFAULT_FNS))
    if fns != DEFAULT_FNS:
        parts.append("f{}n{}s{}".format(*fns))
    parts += [f"{label}{NAME_FORMATS.get(key, _fmt)(params[key])}"
              for key, label in NAME_KEYS[model["type"]] if key in params]

    if not _learned(model):
        features = params.get("features")
        if features is not None and tuple(features) != tuple(FEATURE_GROUPS):
            parts.append("-".join(features))
        trained = ""
    elif config["training"]["train"]:
        trained = f"e{config['training']['epochs']}" + ("-ft" if model.get("checkpoint") else "")
    else:
        trained = "extract-" + _checkpoint_label(model["checkpoint"])

    model_part = "-".join(parts) + (f"_{trained}" if trained else "")
    if config["stop_after"] == "features":
        return _safe(f"{model_part}__{_data_label(config)}")
    return _safe(f"{model_part}__{method}-c{clustering['cluster_num']}__{_data_label(config)}")


def _checkpoint_label(checkpoint) -> str:
    """Which model a checkpoint is, for run names: the name of the run that
    saved it - data/pipeline/<run name>__<time>/FeatureLearningStep/<Model>/model.pt,
    or <run name>__<time>.pt for older ones - without its time."""
    path = Path(checkpoint)
    run = path.parents[2].name if path.name == "model.pt" and len(path.parents) > 2 else path.stem
    return run.split("__")[0]


def _features_label(features) -> tuple[str, str | None]:
    """Model and data part of the name of the run that wrote a features.csv
    (data/pipeline/<model>__[<clustering>__]<data>__<time>/FeatureLearningStep/<Model>/features.csv);
    the data part is None if the path does not look like that."""
    path = Path(features)
    parts = (path.parents[2].name if len(path.parents) > 2 else path.stem).split("__")
    return parts[0], parts[-2] if len(parts) >= 3 else None


def _data_label(config: dict) -> str:
    """The images the run uses: 'raw', the preprocessed_path below
    data/processed (legion/no_axes/train -> legion-no_axes-train), and with
    data.extract_path the folder features come from: trained on one and
    extracting from the other, '<train>-to-<extract>' without their common
    start (legion-no_axes-train-to-test)."""
    data = config["data"]
    extract = data.get("extract_path")
    if extract and config["stop_after"] != "preprocessing" and not _trains(config):
        return "-".join(_path_parts(extract))
    trained = ["raw"] if data["source"] == "raw" else _path_parts(data["preprocessed_path"])
    if not extract or config["stop_after"] == "preprocessing":
        return "-".join(trained)
    extracted = _path_parts(extract)
    common = 0
    while common < min(len(trained), len(extracted)) - 1 and trained[common] == extracted[common]:
        common += 1
    return "-".join(trained) + "-to-" + "-".join(extracted[common:])


def _path_parts(path) -> list[str]:
    path = resolve_path(path)
    try:
        return list(path.relative_to(PROCESSED_ROOT).parts)
    except ValueError:
        return list(path.parts[-2:])


def _learned(model_config: dict) -> bool:
    """Whether the feature step is trained (or loaded from a checkpoint)
    here: the learned models, and DINOv3 with `finetune: true`."""
    return (model_config["type"] not in MEASURED
            or bool((model_config.get("params") or {}).get("finetune")))


def _trains(config: dict) -> bool:
    return bool(config["training"]["train"]) and _learned(config["model"])


def _fmt(value) -> str:
    return f"{value:g}" if isinstance(value, float) else str(value)


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", str(name)).strip("-")


# --------------------------------------------------------------------- #
# Building the steps
# --------------------------------------------------------------------- #

def build_preprocessing_steps(config: dict, run_id: str) -> list:
    """The configured preprocessing steps, in order, each writing to its own
    folder data/pipeline/<run id>/PreprocessingStep/<Step>/."""
    steps = []
    for entry in config["preprocessing"]["steps"] or []:
        (class_name, params), = entry.items()
        # YAML has no tuples; e.g. Normalizer's background_color must be one.
        params = {key: tuple(value) if isinstance(value, list) else value for key, value in (params or {}).items()}
        if "model" in params:  # BackgroundEraser's YOLO weights
            params["model"] = str(resolve_path(params["model"]))
        steps.append(PREPROCESSING_STEPS[class_name](run_id=run_id, **params))
    return steps


def build_model(config: dict, run_id: str, device):
    """The feature step: a learned model (on `device`, with `model.checkpoint`
    loaded if given - a plain state_dict, as saved by `save_model` or
    torch.save), or one not trained here (HandcraftedFeatures, frozen
    DINOv3). A fine-tuned DINOv3 is not an nn.Module (it picks its device
    itself) but has state_dict/load_state_dict for its trained parts."""
    model_config = config["model"]
    model = MODELS[model_config["type"]](run_id=run_id, **(model_config.get("params") or {}))
    if not _learned(model_config):
        return model
    if isinstance(model, torch.nn.Module):
        model = model.to(device)
    if model_config.get("checkpoint"):
        state = torch.load(resolve_path(model_config["checkpoint"]), map_location=device, weights_only=True)
        model.load_state_dict(state)
    return model


def standardize(vectors: np.ndarray, min_variance: float) -> tuple[np.ndarray, np.ndarray]:
    """Z-score every dimension, after dropping the near-constant ones
    (inactive latent dimensions would otherwise be blown up to unit variance).
    Returns the standardized vectors and the mask of the kept dimensions."""
    keep = vectors.var(axis=0) > min_variance
    if not keep.any():
        raise ValueError(f"No feature dimension has variance above {min_variance}")
    kept = vectors[:, keep]
    return (kept - kept.mean(axis=0)) / kept.std(axis=0), keep


def feature_names(features_csv, n_dims: int) -> list[str]:
    """Names of the feature vector's dimensions: feature_columns.txt next to
    the features.csv (written by the handcrafted features), else dim_<i>."""
    columns = Path(features_csv).with_name("feature_columns.txt")
    if columns.exists():
        names = columns.read_text(encoding="utf-8").split()
        if len(names) == n_dims:
            return names
    return [f"dim_{i}" for i in range(n_dims)]


def feature_importance(vectors: np.ndarray, labels: np.ndarray, names: list[str]) -> pd.DataFrame:
    """How strongly each dimension separates the clusters, most important
    first: eta2 = between-cluster / total sum of squares (the share of the
    dimension's variance the clusters explain, 0..1) and the ANOVA F
    statistic it gives."""
    clusters = np.unique(labels)
    centered = vectors - vectors.mean(axis=0)
    total = (centered ** 2).sum(axis=0)
    between = sum((labels == c).sum() * centered[labels == c].mean(axis=0) ** 2 for c in clusters)
    within = total - between
    k, n = len(clusters), len(vectors)
    with np.errstate(divide="ignore", invalid="ignore"):
        eta2 = np.where(total > 0, between / total, 0.0)
        f = (between / max(k - 1, 1)) / (within / max(n - k, 1))
    table = pd.DataFrame({"feature": names, "eta2": eta2, "f_statistic": f})
    return table.sort_values("eta2", ascending=False, ignore_index=True)


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
        step.feature_names = feature_names(features_csv, n_features)
        if settings["standardize"]:
            step.vectors, keep = standardize(step.vectors, settings["min_variance"])
            step.feature_names = [name for name, kept in zip(step.feature_names, keep) if kept]
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
    """Save the state_dict to model.pt in the model step's folder
    (data/pipeline/<run id>/FeatureLearningStep/<Model>/), and the resolved
    config plus the MLflow run id to model.yaml next to it."""
    folder = Path(model.output_path)
    folder.mkdir(parents=True, exist_ok=True)
    model_path = folder / "model.pt"
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
    _write_yaml(info, folder / "model.yaml")
    return model_path


def run_experiment(config_path=None, overrides: Iterable[str] = (), dry_run: bool = False,
                   config: dict | None = None, tags: dict | None = None) -> dict:
    """Run one experiment end to end (see the module docstring).

    Args:
        config_path: experiment YAML (merged over configs/default.yaml).
        overrides: 'key.sub=value' strings applied on top.
        dry_run: only print the resolved config and the run id.
        config: an already loaded config, instead of `config_path`.
        tags: extra MLflow tags for the run.

    `config["stop_after"]` ends the run early: after 'preprocessing'
    (only the preprocessed images are produced) or after 'features' (model
    trained/saved and features extracted, no clustering or evaluation).
    With `config["data"]["features"]` the run starts late instead: it only
    clusters and evaluates that features.csv. A config with a `sweep` is
    run with `run_sweep`.

    Returns:
        dict with 'run_id', 'config', and unless `dry_run`: 'run_dir',
        'preprocessed_path' (None unless data.source is raw), 'model_path'
        (None if not trained), 'features_path' (the features.csv clustered,
        None after stop_after: preprocessing), 'features' (None with
        data.features), 'labels', 'metrics' (the last ones None when the
        run stopped before them).
    """
    config = config if config is not None else load_config(config_path, overrides)
    if config.get("sweep"):
        raise ValueError("The config has a sweep: run it with run_sweep")
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
        tracking.set_tags(tags or {})
        built = {}  # clustering_factory puts the clustering step here

        # 1. Get cluster labels. Either from the features of an earlier run
        #    (data.features), or by running preprocessing (raw images only)
        #    and the feature step through a Pipeline, which then clusters.
        if data.get("features"):
            # Only clustering and evaluation, on the features of an earlier run.
            features_path = resolve_path(data["features"])
            result = {"features": None,
                      "labels": clustering_factory(config, run_id, built)(str(features_path)).run()}
            preprocessed_path = model_path = None
        else:
            model = build_model(config, run_id, device) if stop_after != "preprocessing" else None
            training = dict(config["training"])
            training.pop("train")
            train = model is not None and _trains(config)
            extract_path = resolve_path(data.get("extract_path")) if model is not None else None
            # Training on one folder, features from another: cluster only after the second extraction.
            separate = extract_path is not None and train
            if extract_path is not None and not train:
                raw = False  # nothing to train: only the images to extract from are needed
            steps = build_preprocessing_steps(config, run_id) if raw else []
            pipe = Pipeline(steps, feature_learning_step=model,
                            clustering_factory=clustering_factory(config, run_id, built)
                            if stop_after is None and not separate else None)

            feature_kwargs = training if train else {"batch_size": training.get("batch_size", 32)}
            source = data["raw_path"] if raw else (extract_path if not separate and extract_path
                                                   else data["preprocessed_path"])
            result = pipe.run(
                str(resolve_path(source)),
                # Ready images: disk mode reads the folder directly, no copy in RAM.
                mode=config["preprocessing"]["mode"] if raw else "disk",
                preprocess_batch_size=config["preprocessing"]["batch_size"],
                feature_mode="train_model" if train else "extract_features",
                save_preprocessed="last" if raw else None,
                **(feature_kwargs if model is not None else {}),
            )
            # 2. Note where everything went; save the trained model.
            features_path = Path(model.output_path) / "features.csv" if model is not None else None
            image_steps = [step for step in steps if not isinstance(step, Augmentor)]
            preprocessed_path = Path(image_steps[-1].output_path) if image_steps else None
            tracking.set_tags({"preprocessed_output": preprocessed_path})

            model_path = save_model(model, config, run_id, mlflow_run.info.run_id) if train else None
            tracking.set_tags({"model_path": model_path})

            if separate:
                # 3. data.extract_path: extract features from the other folder
                #    with the trained model, then cluster those. The training
                #    images' features are kept as features_train.csv.
                features_path.replace(features_path.with_name("features_train.csv"))
                model.set_images(None)
                model.input_path = str(extract_path)
                result["features"] = model.run(mode="extract_features", batch_size=feature_kwargs["batch_size"])
                if stop_after is None:
                    result["labels"] = clustering_factory(config, run_id, built)(str(features_path)).run()

        # 4. Evaluate the clusters and write clusters.csv, the feature
        #    importance and metrics.json.
        predicted, metrics = result.get("labels"), None
        if stop_after is None:
            clustering = built["step"]
            metrics = evaluate(clustering, predicted, resolve_path(data["labels"]))
            names = clustering.filenames if clustering.filenames is not None else range(len(predicted))
            clusters_csv = Path(clustering.output_path) / "clusters.csv"
            clusters_csv.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"filename": names, "cluster": predicted}).to_csv(clusters_csv, index=False)
            importance = feature_importance(clustering.vectors, predicted, clustering.feature_names)
            importance.to_csv(clusters_csv.with_name("feature_importance.csv"), index=False)
            tracking.log_table(importance, "clustering/feature_importance.csv")
            # The evaluation steps write nothing themselves; their stage folder holds the metrics.
            evaluation_dir = run_dir / "EvaluationStep"
            evaluation_dir.mkdir(exist_ok=True)
            (evaluation_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))

    print(f"Done: {run_id}" + (f" (stopped after {stop_after})" if stop_after else ""))
    if preprocessed_path:
        print(f"  preprocessed images: {preprocessed_path}")
    if model_path:
        print(f"  model:   {model_path}")
    if metrics:
        print("  metrics: " + ", ".join(f"{key}={value:.4g}" for key, value in metrics.items()))
    top = config["clustering"]["feature_importance"]
    if stop_after is None and top:
        print(f"  feature importance (eta2, top {min(top, len(importance))} of {len(importance)}; "
              f"all in {clusters_csv.with_name('feature_importance.csv')}):")
        for row in importance.head(top).itertuples():
            print(f"    {row.feature:<30} {row.eta2:.3f}")
    return {"run_id": run_id, "config": config, "run_dir": run_dir, "preprocessed_path": preprocessed_path,
            "model_path": model_path, "features_path": features_path, "features": result.get("features"),
            "labels": predicted, "metrics": metrics}


def run_sweep(config_path=None, overrides: Iterable[str] = (), dry_run: bool = False,
              config: dict | None = None) -> list[dict]:
    """One run (`run_experiment`) per combination of the config's sweep.

    The model is trained once for each combination of the other settings;
    the combinations that differ from it only in clustering settings then
    cluster its features.csv again (`data.features`). If that first run
    fails, they are skipped; any other failed run is reported and the sweep
    goes on. Every run gets its own folder and MLflow run (tagged with the
    sweep id) as usual, and data/pipeline/<sweep id>/ gets the config
    (sweep.yaml) and summary.csv: the swept settings, run id, status and
    metrics of every run, rewritten after each one.

    Returns:
        The rows of summary.csv, as dicts.
    """
    config = config if config is not None else load_config(config_path, overrides)
    combinations, skipped = sweep_configs(config)
    _name_sweep_runs(combinations)
    label = config["name"] or (Path(config_path).stem if config_path else "default")
    sweep_id = _safe(f"sweep-{label}__{datetime.now():%Y%m%d-%H%M%S}")
    groups = [_upstream(swept) for swept, _ in combinations]
    first_of = {}
    for index, group in enumerate(groups):
        first_of.setdefault(group, index)

    print(f"Sweep {sweep_id}: {len(combinations)} runs, {len(first_of)} of them computing features")
    for index, (swept, combination) in enumerate(combinations):
        source = "" if first_of[groups[index]] == index else (
            f"  (features of {combinations[first_of[groups[index]]][1]['name']})")
        print(f"  {index + 1:>3}. {combination['name']}{source}")
    if skipped:
        print(f"Left out {len(skipped)} combinations:")
        for swept, why in skipped:
            print(f"       {_describe(swept)}: {why}")
    if dry_run:
        return [{**swept, "run_id": combination["name"], "status": "dry run"} for swept, combination in combinations]

    sweep_dir = Path(step_module.DEFAULT_OUTPUT_ROOT) / sweep_id
    sweep_dir.mkdir(parents=True, exist_ok=True)
    _write_yaml(config, sweep_dir / "sweep.yaml")
    features_of, rows = {}, []
    for index, (swept, combination) in enumerate(combinations):
        group = groups[index]
        first = first_of[group] == index
        print(f"\n=== Sweep run {index + 1}/{len(combinations)}: {_describe(swept)}")
        row = {**swept, "run_id": None}
        if not first:
            if features_of.get(group) is None:
                rows.append({**row, "status": "skipped: its features run failed"})
                _write_summary(rows, sweep_dir)
                continue
            combination["data"]["features"] = str(features_of[group])
            combination["data"]["extract_path"] = None
        try:
            result = run_experiment(config_path, config=combination, tags={"sweep": sweep_id})
        except Exception as error:  # report it, go on with the other runs
            traceback.print_exc()
            rows.append({**row, "status": f"failed: {type(error).__name__}: {error}"})
        else:
            if first:
                features_of[group] = result["features_path"]
            rows.append({**row, "run_id": result["run_id"], "status": "ok", **(result["metrics"] or {})})
        _write_summary(rows, sweep_dir)

    print(f"\nSweep {sweep_id} done: {sum(row['status'] == 'ok' for row in rows)}/{len(rows)} runs ok")
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(pd.DataFrame(rows).drop(columns="run_id").to_string(index=False))
    print(f"  summary: {sweep_dir / 'summary.csv'}")
    return rows


def _upstream(swept: dict) -> str:
    """What the features depend on: the swept settings other than clustering."""
    return json.dumps({key: value for key, value in swept.items() if not _is_downstream(key)},
                      sort_keys=True, default=str)


def _name_sweep_runs(combinations: list[tuple[dict, dict]]) -> None:
    """Set each combination's `name`: its generated run name (or the
    config's own name), followed by the swept settings that name does not
    show, e.g. 'vae-k16_e100__ward-euclidean-c30__legion-train__lr0.001'."""
    swept, first = combinations[0]
    hidden = []
    for key, value in swept.items():
        values = {json.dumps(other[key], sort_keys=True, default=str): other[key] for other, _ in combinations}
        names = set()
        for other in values.values():
            changed = copy.deepcopy(first)
            set_dotted(changed, key, other)
            names.add(run_name(changed))
        if len(names) < len(values):
            hidden.append(key)
    for swept, combination in combinations:
        suffix = "-".join(f"{key.split('.')[-1]}{_fmt(swept[key])}" for key in hidden)
        combination["name"] = _safe(run_name(combination) + (f"__{suffix}" if suffix else ""))
    names = [combination["name"] for _, combination in combinations]
    if len(set(names)) < len(names):  # e.g. clustering.metric, which method: auto ignores
        for index, (_, combination) in enumerate(combinations):
            combination["name"] += f"__run{index + 1}"


def _write_summary(rows: list[dict], sweep_dir: Path) -> None:
    pd.DataFrame(rows).to_csv(sweep_dir / "summary.csv", index=False)


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
        "data": resolve_path(config["data"].get("features") or config["data"][
            "raw_path" if config["data"]["source"] == "raw" else "preprocessed_path"]),
        "extract_data": resolve_path(config["data"].get("extract_path")),
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
