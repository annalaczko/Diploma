import pytest

import pipeline.steps.step as step_module


@pytest.fixture(autouse=True)
def _isolated_default_output(tmp_path, monkeypatch):
    # Steps without an explicit output_path create <repo>/data/pipeline/...;
    # keep tests from writing into the real data folder.
    monkeypatch.setattr(step_module, "DEFAULT_OUTPUT_ROOT", tmp_path / "default_output")


@pytest.fixture(autouse=True)
def _isolated_mlflow_store(tmp_path, monkeypatch):
    # Pipeline.run(experiment_name=...) would otherwise log into the repo's
    # mlflow.db; artifacts follow the database into tmp_path/mlartifacts.
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
