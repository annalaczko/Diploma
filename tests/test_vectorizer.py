"""Tests for the Vectorizer preprocessing step (pipeline/preprocessing/vectorizer.py)."""

import mlflow
import numpy as np
import pytest
from PIL import Image

from pipeline import tracking
from pipeline.preprocessing import ImageRecord, Vectorizer, collect_image_paths


def _drawing(noise=False, size=256):
    """A shard-like outline: two curved walls joined by a rim, optionally
    with paper speckles and a small gap in one wall."""
    img = np.full((size, size), 255, np.uint8)
    ys = np.arange(40, 220)
    for offset in (0, 30):
        xs = (80 + offset + 25 * np.sin(ys / 40)).astype(int)
        img[ys, xs] = img[ys, xs + 1] = 30
    img[40:42, 80:111] = 30  # rim
    if noise:
        rng = np.random.default_rng(0)
        y, x = rng.integers(0, size, 60), rng.integers(0, size, 60)
        img[y, x] = 60  # isolated speckles
        img[120:124] = np.where(img[120:124] < 128, 255, img[120:124])  # gap in the walls
    return Image.fromarray(img)


def _blank(size=256):
    return Image.fromarray(np.full((size, size), 255, np.uint8))


def test_output_is_clean_redrawn_lines():
    out = np.asarray(Vectorizer(debug_images=0).process_image(_drawing(noise=True)))

    assert out.shape == (256, 256)
    assert set(np.unique(out)) <= {0, 255}  # crisp black on white
    # Speckles are gone: nothing dark far away from the drawing.
    assert not (out[:, :50] < 128).any() and not (out[:, 200:] < 128).any()
    # The gap in the walls is bridged.
    assert (out[118:126] < 128).any(axis=1).all()


@pytest.mark.parametrize("on_failure, kept", [("keep", 2), ("drop", 1)])
def test_untraceable_image_is_kept_or_dropped_and_reported(tmp_path, on_failure, kept):
    step = Vectorizer(output_path=str(tmp_path / "out"), on_failure=on_failure, debug_images=0)
    step.begin_run()

    out = step.process_batch([ImageRecord("ok.png", _drawing()), ImageRecord("blank.png", _blank())])

    assert len(out) == kept
    assert step.no_trace == ["blank.png"]
    if on_failure == "keep":
        np.testing.assert_array_equal(np.asarray(out[1].image), np.asarray(_blank()))


def test_disk_run_writes_images_svgs_and_logs_failures(tmp_path):
    source = tmp_path / "raw"
    (source / "sub").mkdir(parents=True)
    _drawing().save(source / "sub" / "a.png")
    _blank().save(source / "sub" / "b.png")
    step = Vectorizer(input_path=str(source), output_path=str(tmp_path / "out"), save_svg=True, debug_images=1)

    with tracking.start_run("vec"):
        step.run()

    # Images in the output folder, side outputs in its _svg / _debug folders.
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == ["_debug", "_svg", "sub__a.png", "sub__b.png"]
    assert [p.name for p in (tmp_path / "out" / "_svg").iterdir()] == ["sub__a.svg"]
    assert "<path d=" in (tmp_path / "out" / "_svg" / "sub__a.svg").read_text()
    assert len(list((tmp_path / "out" / "_debug").iterdir())) == 1
    assert [p.name for p in collect_image_paths(tmp_path / "out")] == ["sub__a.png", "sub__b.png"]
    [run] = mlflow.search_runs(experiment_names=["vec"], output_format="list")
    assert run.data.metrics["preprocessing/Vectorizer/n_images"] == 2
    assert run.data.metrics["preprocessing/Vectorizer/no_trace_count"] == 1
