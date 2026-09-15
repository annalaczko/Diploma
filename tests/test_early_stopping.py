import os

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image

from pipeline.feature_learning.early_stopping import EarlyStopping
import pipeline.feature_learning.vae as vae_module
import pipeline.feature_learning.simclr as simclr_module
import pipeline.feature_learning.byol as byol_module
from pipeline.feature_learning.vae import VAE
from pipeline.feature_learning.simclr import SimCLR
from pipeline.feature_learning.byol import BYOL


def _random_images(dir_path, count=4, size=256):
    for i in range(count):
        arr = (np.random.rand(size, size) * 255).astype("uint8")
        Image.fromarray(arr, mode="L").save(os.path.join(dir_path, f"img{i}.png"))


class _ImmediateStop:
    """Stand-in for `EarlyStopping` that always reports "stop now" -
    lets these tests assert the training loops actually break on the
    signal, without waiting for real loss convergence."""

    def __init__(self, *args, **kwargs):
        self.best_smoothed = 0.0
        self.best_epoch = 1
        self.restored = False

    def step(self, value, epoch, model=None):
        return value, True

    def restore_best(self, model):
        self.restored = True


def test_rejects_invalid_mode():
    with pytest.raises(ValueError):
        EarlyStopping(mode="not_min_or_max")


def test_min_mode_stops_after_patience_epochs_without_improvement():
    stopper = EarlyStopping(patience=3, smoothing_window=1, mode="min")

    # Improving, then flat for exactly `patience` epochs.
    results = [stopper.step(value, epoch) for epoch, value in enumerate([5, 4, 3, 3, 3, 3], start=1)]
    should_stop = [r[1] for r in results]

    assert should_stop == [False, False, False, False, False, True]
    assert stopper.best_epoch == 3


def test_max_mode_prefers_higher_values():
    stopper = EarlyStopping(patience=2, smoothing_window=1, mode="max")

    _, stop1 = stopper.step(0.5, 1)
    _, stop2 = stopper.step(0.8, 2)  # improvement
    _, stop3 = stopper.step(0.6, 3)  # worse
    _, stop4 = stopper.step(0.6, 4)  # still worse -> patience exhausted

    assert (stop1, stop2, stop3, stop4) == (False, False, False, True)
    assert stopper.best_epoch == 2
    assert stopper.best_smoothed == 0.8


def test_min_delta_requires_a_meaningful_improvement():
    stopper = EarlyStopping(patience=2, min_delta=0.5, smoothing_window=1, mode="min")

    stopper.step(10.0, 1)
    # A tiny decrease that doesn't clear min_delta shouldn't count as an
    # improvement, so patience should keep counting down.
    _, stop = stopper.step(9.9, 2)
    _, stop2 = stopper.step(9.9, 3)

    assert stop is False
    assert stop2 is True
    assert stopper.best_epoch == 1


def test_smoothing_window_averages_trailing_values():
    stopper = EarlyStopping(patience=10, smoothing_window=2, mode="min")

    smoothed1, _ = stopper.step(10.0, 1)
    smoothed2, _ = stopper.step(20.0, 2)
    smoothed3, _ = stopper.step(0.0, 3)

    assert smoothed1 == 10.0
    assert smoothed2 == 15.0  # mean of [10, 20]
    assert smoothed3 == 10.0  # mean of [20, 0], the last 2 values only


def test_restore_best_loads_the_snapshotted_weights():
    model = nn.Linear(2, 1)
    stopper = EarlyStopping(patience=5, smoothing_window=1, mode="min")

    stopper.step(1.0, 1, model=model)
    best_weight = model.weight.detach().clone()

    with torch.no_grad():
        model.weight.add_(100.0)  # simulate further (worse) training

    stopper.restore_best(model)

    assert torch.equal(model.weight, best_weight)


def test_restore_best_is_a_noop_without_any_snapshot():
    model = nn.Linear(2, 1)
    original_weight = model.weight.detach().clone()
    stopper = EarlyStopping()

    stopper.restore_best(model)

    assert torch.equal(model.weight, original_weight)


# ---------------------------------------------------------------------------
# Integration: each unsupervised model's train_model actually breaks its
# epoch loop as soon as EarlyStopping reports "stop now", instead of
# always running the full requested `epochs`. A fake, always-stop
# EarlyStopping is monkeypatched in so this doesn't depend on the model's
# loss ever actually plateauing on tiny random-noise test data.
# ---------------------------------------------------------------------------

def _count_epoch_lines(capsys):
    printed = capsys.readouterr().out
    return len([line for line in printed.splitlines() if line.startswith("[epoch")])


def test_vae_train_model_stops_early_when_loss_plateaus(tmp_path, monkeypatch, capsys):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    _random_images(source_dir, count=4)

    # Default k=128: Decoder's reshape after `fc` is hardcoded to the
    # default architecture's channel width, so a custom k would break
    # reconstruction shape here independently of anything early-stopping
    # related - not something this test is meant to exercise.
    model = VAE(input_path=str(source_dir), output_path=str(tmp_path / "out"))
    monkeypatch.setattr(vae_module, "EarlyStopping", _ImmediateStop)

    model.train_model(str(source_dir), epochs=5, batch_size=2)

    assert _count_epoch_lines(capsys) == 1


def test_simclr_train_model_stops_early_when_loss_plateaus(tmp_path, monkeypatch, capsys):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    _random_images(source_dir, count=4)

    model = SimCLR(input_path=str(source_dir), output_path=str(tmp_path / "out"), k=16, projection_dim=8)
    monkeypatch.setattr(simclr_module, "EarlyStopping", _ImmediateStop)

    model.train_model(str(source_dir), epochs=5, batch_size=2)

    assert _count_epoch_lines(capsys) == 1


def test_byol_train_model_stops_early_when_loss_plateaus(tmp_path, monkeypatch, capsys):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    _random_images(source_dir, count=4)

    model = BYOL(
        input_path=str(source_dir),
        output_path=str(tmp_path / "out"),
        k=16,
        hidden_dim=32,
        projection_dim=8,
    )
    monkeypatch.setattr(byol_module, "EarlyStopping", _ImmediateStop)

    model.train_model(str(source_dir), epochs=5, batch_size=2)

    assert _count_epoch_lines(capsys) == 1
