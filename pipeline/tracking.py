"""Optional MLflow experiment tracking for the whole pipeline.

Every step logs through the helpers below, and they only do something while
an MLflow run is active (`start_run`, or `Pipeline.run(experiment_name=...)`).
Without an active run, or without mlflow installed, they are no-ops, so the
steps keep working unchanged in notebooks and tests.

Runs are stored in the repository, independent of the working directory (a
notebook in notebooks/ and a script at the repo root share the same store):
  - run metadata: <repo>/mlflow.db (SQLite)
  - artifacts:    <repo>/mlartifacts/<experiment>/...
Set the MLFLOW_TRACKING_URI environment variable to use another server.
Browse them with:
    mlflow ui --backend-store-uri sqlite:///mlflow.db    (from the repo root)
"""

from __future__ import annotations

import math
import os
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator

# MLflow prints a hint for coding agents on import; it is noise here.
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

try:
    import mlflow
except ImportError:  # tracking is optional
    mlflow = None

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACKING_URI = f"sqlite:///{REPO_ROOT / 'mlflow.db'}"

# MLflow rejects longer parameter values.
_MAX_PARAM_LENGTH = 6000


def is_active() -> bool:
    """True when there is an MLflow run to log into."""
    return mlflow is not None and mlflow.active_run() is not None


def setup(experiment_name: str, tracking_uri: str | None = None) -> str:
    """Point MLflow at the repo's store (unless `tracking_uri` or
    MLFLOW_TRACKING_URI says otherwise) and select `experiment_name`,
    creating it with its artifacts under <repo>/mlartifacts if needed.

    Returns:
        The experiment id.
    """
    if mlflow is None:
        raise ImportError("mlflow is not installed: pip install mlflow")
    uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI") or DEFAULT_TRACKING_URI
    mlflow.set_tracking_uri(uri)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        # With a local SQLite store, keep the artifacts next to the database
        # file (MLflow's default would be ./mlruns under the working
        # directory). A tracking server decides on its own.
        artifact_location = None
        if uri.startswith("sqlite:///"):
            db_dir = Path(uri[len("sqlite:///"):]).resolve().parent
            artifact_location = (db_dir / "mlartifacts" / experiment_name).as_uri()
        experiment_id = mlflow.create_experiment(experiment_name, artifact_location=artifact_location)
    else:
        experiment_id = experiment.experiment_id
    mlflow.set_experiment(experiment_id=experiment_id)
    return experiment_id


@contextmanager
def start_run(
    experiment_name: str = "pottery",
    run_name: str | None = None,
    tags: dict[str, Any] | None = None,
    tracking_uri: str | None = None,
) -> Iterator[Any]:
    """Context manager that opens an MLflow run in `experiment_name`. Inside
    an already active run it opens a nested child run instead.

    Example (notebook, steps called one by one):
        with tracking.start_run("vae_experiments", run_name="vae_k128"):
            features = vae.run(mode="train_model", epochs=100)
            labels = HierarchicalClustering(input_path=csv, cluster_num=55).run()
    """
    nested = is_active()
    if not nested:
        setup(experiment_name, tracking_uri)
    with mlflow.start_run(run_name=run_name, nested=nested, tags=tags) as run:
        yield run


def _scalar(value: Any) -> Any:
    """`value` if it can be a parameter (scalar, path, tuple of scalars),
    else None. Lists are left out on purpose: in steps they hold results
    (e.g. names of skipped images), not configuration."""
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple) and all(isinstance(v, (bool, int, float, str)) for v in value):
        return str(list(value))
    return None


def params_of(obj: Any, prefix: str = "") -> dict[str, Any]:
    """The scalar public attributes of `obj` (e.g. a step's configuration
    such as `cluster_num` or `mask_margin`), keyed `<prefix><name>`.
    Arrays, tensors, models and other objects are skipped, and so is
    `input_path`: `Pipeline` rewires it between steps, and MLflow does not
    allow a parameter to change within a run."""
    params = {}
    for name, value in vars(obj).items():
        # nn.Module's train/eval flag is state, not configuration.
        if name.startswith("_") or name in ("training", "input_path"):
            continue
        scalar = _scalar(value)
        if scalar is not None or value is None:
            params[f"{prefix}{name}"] = scalar
    return params


def log_params(params: dict[str, Any], prefix: str = "") -> None:
    """Log parameters; non-scalar values are skipped. A parameter that was
    already logged with a different value in this run (MLflow forbids
    changing it) only produces a warning."""
    if not is_active():
        return
    for key, value in params.items():
        if _scalar(value) is None and value is not None:
            continue
        text = str(_scalar(value))[:_MAX_PARAM_LENGTH]
        try:
            mlflow.log_param(f"{prefix}{key}", text)
        except Exception as exc:  # e.g. same key, different value
            warnings.warn(f"MLflow: could not log param {prefix}{key}: {exc}")


def log_metrics(metrics: dict[str, Any], step: int | None = None, prefix: str = "") -> None:
    """Log numeric metrics, skipping None and non-finite values."""
    if not is_active():
        return
    clean = {}
    for key, value in metrics.items():
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            clean[f"{prefix}{key}"] = number
    if clean:
        mlflow.log_metrics(clean, step=step)


def log_artifact(path: str | Path, artifact_path: str | None = None) -> None:
    """Upload a local file (e.g. features.csv) to the run."""
    if is_active() and Path(path).exists():
        mlflow.log_artifact(str(path), artifact_path=artifact_path)


def log_text(text: str, artifact_file: str) -> None:
    """Store `text` as a file in the run, e.g. 'preprocessing/no_shard.txt'."""
    if is_active():
        mlflow.log_text(text, artifact_file)


def log_lines(lines: Iterable[str], artifact_file: str) -> None:
    """Store one item per line (e.g. image names) as a text file in the run."""
    log_text("".join(f"{line}\n" for line in lines), artifact_file)


def log_table(df, artifact_file: str) -> None:
    """Store a DataFrame as a CSV file in the run."""
    if is_active():
        mlflow.log_text(df.to_csv(index=False), artifact_file)
