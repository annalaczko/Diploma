import pytest

import pipeline.steps.step as step_module


@pytest.fixture(autouse=True)
def _isolated_default_output(tmp_path, monkeypatch):
    # Steps without an explicit output_path create <repo>/data/pipeline/...;
    # keep tests from writing into the real data folder.
    monkeypatch.setattr(step_module, "DEFAULT_OUTPUT_ROOT", tmp_path / "default_output")
