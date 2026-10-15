"""Tests for the hand-crafted profile features
(pipeline/feature_learning/handcrafted_features.py) on a synthetic drawing
with known measures."""

import numpy as np
import pandas as pd
import pytest

from pipeline.feature_learning import handcrafted_features as hf
from pipeline.preprocessing.axes import METADATA_CSV, METADATA_COLUMNS

PX_PER_CM = 20.0
AXIS_X, LINE_Y = 200.0, 40.0  # vertical axis at x=200, rim line at y=40


def _folder(tmp_path, paths):
    """A Vectorizer-like folder: _svg/img.svg with `paths`, and a metadata row
    with the axes (axis also drawn into the SVG as a remnant)."""
    (tmp_path / "_svg").mkdir()
    d = "".join(f'<path d="M {" L ".join(f"{x} {y}" for x, y in p)}"/>' for p in paths)
    (tmp_path / "_svg" / "img.svg").write_text(
        f'<svg><g id="drawing">{d}</g><g id="axes"></g></svg>', encoding="utf-8")
    row = {c: np.nan for c in METADATA_COLUMNS}
    row.update(filename="img.png", found=True, frame_w=256, frame_h=256, px_per_cm=PX_PER_CM,
               v_x1=AXIS_X, v_y1=LINE_Y, v_x2=AXIS_X, v_y2=250, h_x1=20, h_y1=LINE_Y, h_x2=AXIS_X, h_y2=LINE_Y,
               origin_x=AXIS_X, origin_y=LINE_Y)
    pd.DataFrame([row], columns=METADATA_COLUMNS).to_csv(tmp_path / METADATA_CSV, index=False)
    return tmp_path


def test_wall_measures_in_cm(tmp_path):
    # A wall hanging below the rim line, left of the axis: outer side at
    # x=100 (5 cm from the axis), inner side at x=116 (0.8 cm thick), 3 cm high,
    # closed at the top by the rim; plus a stub of the vertical axis.
    wall = [(100, 100), (100, 40), (116, 40), (116, 100)]
    axis_stub = [(200, 60), (200, 120)]
    df = hf.extract_features(_folder(tmp_path, [wall, axis_stub]))
    row = df.loc["img.png"]

    assert row["position"] == "below"
    assert row["rim_height_cm"] == pytest.approx(3.0, abs=0.1)
    assert row["rim_width_cm"] == pytest.approx(0.8, abs=0.1)
    assert row["rim_axis_dist_max_cm"] == pytest.approx(5.0, abs=0.1)  # axis stub removed
    assert row["rim_axis_dist_min_cm"] == pytest.approx(4.2, abs=0.1)
    assert row["rim_thickness_mean_cm"] == pytest.approx(0.8, abs=0.1)
    efd = row[[c for c in df.columns if c.startswith("efd_")]].to_numpy(float)
    assert len(efd) == 40 and np.isfinite(efd).all()


def test_base_fragment_is_above_and_missing_axes_give_nan(tmp_path):
    base = [(60, 40), (140, 40), (150, 0), (130, 0), (125, 25), (60, 25)]  # standing on the line
    folder = _folder(tmp_path, [base])
    meta = pd.read_csv(folder / METADATA_CSV)
    meta = pd.concat([meta, meta.assign(filename="none.png", found=False)])
    meta.to_csv(folder / METADATA_CSV, index=False)

    df = hf.extract_features(folder)

    assert df.loc["img.png", "position"] == "above"
    assert df.loc["none.png"].drop("position").isna().all()


# --------------------------------------------------------------------- the pipeline step

def _ready_images(tmp_path):
    """A copied Vectorizer folder after split_by_labels: images in train/,
    _svg/ and extracted_metadata.csv in the parent. Four walls of two kinds."""
    from PIL import Image

    walls = {"a0": 16, "a1": 17, "b0": 40, "b1": 42}  # thin and thick walls
    rows = []
    (tmp_path / "train").mkdir()
    (tmp_path / "_svg").mkdir()
    for name, thick in walls.items():
        wall = [(100, 100), (100, 40), (100 + thick, 40), (100 + thick, 100)]
        d = f'<path d="M {" L ".join(f"{x} {y}" for x, y in wall)}"/>'
        (tmp_path / "_svg" / f"{name}.svg").write_text(f'<svg><g id="drawing">{d}</g></svg>', encoding="utf-8")
        Image.new("L", (256, 256), 255).save(tmp_path / "train" / f"{name}.png")
        row = {c: np.nan for c in METADATA_COLUMNS}
        row.update(filename=f"{name}.png", found=True, frame_w=256, frame_h=256, px_per_cm=PX_PER_CM,
                   v_x1=AXIS_X, v_y1=LINE_Y, v_x2=AXIS_X, v_y2=250, h_x1=20, h_y1=LINE_Y, h_x2=AXIS_X,
                   h_y2=LINE_Y, origin_x=AXIS_X, origin_y=LINE_Y)
        rows.append(row)
    pd.DataFrame(rows, columns=METADATA_COLUMNS).to_csv(tmp_path / METADATA_CSV, index=False)
    labels = tmp_path / "labels.csv"
    pd.DataFrame({"ID": list(walls), "TypeRev": ["thin", "thin", "thick", "thick"]}).to_csv(labels, index=False)
    return tmp_path / "train", labels


def test_run_experiment_with_handcrafted_features(tmp_path):
    from pipeline.experiment import load_config, run_experiment

    images, labels = _ready_images(tmp_path)
    config = load_config(None, [
        "experiment=test", "model.type=handcrafted", "model.params={features: [size, thickness, fourier]}",
        f"data.preprocessed_path={images}", f"data.labels={labels}", "clustering.cluster_num=2",
    ])
    assert config["model"]["params"] == {"features": ["size", "thickness", "fourier"]}  # no VAE defaults

    result = run_experiment(config=config)

    assert result["run_id"].startswith("handcrafted-size-thickness-fourier__ward-euclidean-c2__")
    assert result["model_path"] is None
    step_dir = result["run_dir"] / "FeatureLearningStep" / "HandcraftedFeatures"
    assert {"features.csv", "handcrafted_features.csv", "feature_columns.txt"} <= {p.name for p in step_dir.iterdir()}
    table = pd.read_csv(step_dir / "handcrafted_features.csv", index_col=0)
    assert table.loc["b0.png", "rim_thickness_mean_cm"] == pytest.approx(2.0, abs=0.1)
    # The thin and the thick walls end up in different clusters.
    assert result["metrics"]["acc"] == 1.0


def test_handcrafted_rejects_a_checkpoint_and_unknown_groups():
    from pipeline.experiment import load_config

    with pytest.raises(ValueError, match="no weights"):
        load_config(None, ["model.type=handcrafted", "model.checkpoint=x.pt"])
    with pytest.raises(ValueError, match="Unknown feature groups"):
        hf.HandcraftedFeatures(features=["size", "colour"])


def test_switching_the_model_type_drops_the_other_types_params():
    from pipeline.experiment import load_config

    assert load_config(None, ["model.type=handcrafted"])["model"]["params"] == {}
    assert load_config(None, ["model.params.efd_harmonics=6", "model.type=handcrafted"])["model"]["params"] \
        == {"efd_harmonics": 6}  # the new type's own params are kept, in any order
