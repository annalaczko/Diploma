"""VAE.beta_max is set at construction and can be overridden per run."""

import numpy as np
import pytest
from PIL import Image

from pipeline.feature_learning.vae import VAE


@pytest.fixture
def images(tmp_path):
    folder = tmp_path / "images"
    folder.mkdir()
    for i in range(2):
        Image.fromarray(np.full((256, 256), 255 - 50 * i, np.uint8)).save(folder / f"{i}.png")
    return folder


def _betas(vae, monkeypatch, **run_kwargs):
    """Train one epoch at full beta and return the beta values it logged."""
    betas = []
    monkeypatch.setattr(VAE, "_cyclical_beta", staticmethod(lambda *args, **kwargs: 1.0))
    monkeypatch.setattr(vae, "_log_epoch", lambda epoch, **metrics: betas.append(metrics["beta"]))
    vae.run(mode="train_model", epochs=1, batch_size=2, lr=1e-4, **run_kwargs)
    return betas


def test_beta_max_from_constructor(tmp_path, images, monkeypatch):
    vae = VAE(input_path=str(images), beta_max=4.0, output_path=str(tmp_path / "out"))

    assert _betas(vae, monkeypatch) == [4.0]
    assert vae.hparams["beta_max"] == 4.0


def test_run_argument_overrides_constructor(tmp_path, images, monkeypatch):
    vae = VAE(input_path=str(images), beta_max=4.0, output_path=str(tmp_path / "out"))

    assert _betas(vae, monkeypatch, beta_max=2.0) == [2.0]
    assert vae.beta_max == 4.0  # the override is for that run only
