"""Tests for the coordinate system as data: axes rows (pipeline/preprocessing/axes.py),
drawing them back (AxisRenderer, Vectorizer(axes="draw")) and the
extracted_metadata.csv that Pipeline writes next to the preprocessed images."""

from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from pipeline.pipeline import Pipeline
from pipeline.preprocessing import AxisRenderer, BackgroundEraser, ImageRecord, Normalizer, Vectorizer
from pipeline.preprocessing import axes as axes_lib
from pipeline.preprocessing.axis_detection import AxisDetection, Segment
from pipeline.preprocessing.vectorization import TraceResult


def _row(name="img.png", frame=100, v=(150, 20, 150, 90), h=(40, 20, 150, 20)):
    """Axes row in a frame x frame image: vertical x=150 lies outside it."""
    detection = AxisDetection(origin_x=v[0], origin_y=h[1], vertical=Segment(*v), horizontal=Segment(*h),
                              gap_px=0, confidence=1.0)
    return axes_lib.axis_row(name, detection, None, (frame, frame), scan_size=(frame, frame), dpi=300)


def _square(size=100):
    img = np.full((size, size), 255, np.uint8)
    img[30:70, 30:70] = 0
    return Image.fromarray(img)


# --------------------------------------------------------------------- geometry

def test_extended_frame_is_the_image_when_axes_fit():
    assert axes_lib.extended_frame(100, 100, [(10, 10, 10, 90), (10, 50, 90, 50)], 0.03) == (0, 0, 100, 100)


def test_extended_frame_covers_axes_and_keeps_aspect():
    x0, y0, w, h = axes_lib.extended_frame(100, 50, [(150, 10, 150, 40)], 0.0)

    assert x0 <= 0 and x0 + w >= 150 and y0 <= 0 and y0 + h >= 50
    assert w / h == pytest.approx(2.0)


def test_metadata_csv_round_trip_and_final_coordinates(tmp_path):
    rows = {"img.png": _row(), "other.png": _row("other.png")}
    axes_lib.write_metadata_csv(rows, tmp_path / "meta.csv", sizes={"img.png": (200, 200)})

    [(name, row)] = axes_lib.read_metadata_csv(tmp_path / "meta.csv").items()
    assert name == "img.png"  # only images with a final size
    assert (row["frame_w"], row["v_x1"], row["origin_y"]) == (200, 300, 40)  # scaled x2
    assert row["scale"] == 2 and row["scan_dpi"] == 300
    assert row["px_per_cm"] == pytest.approx(2 * 300 / 2.54)


def _scan_point(row, x, y):
    """Where pixel (x, y) of the row's frame lies on the raw scan."""
    return row["offset_x"] + x / row["scale"], row["offset_y"] + y / row["scale"]


def test_scale_and_offset_map_drawn_axes_back_to_the_scan():
    # Crop at (1000, 2000) on a 300 dpi scan; vertical axis at x=150 outside the 100 px crop.
    detection = AxisDetection(origin_x=1150, origin_y=2020, vertical=Segment(1150, 2020, 1150, 2090),
                              horizontal=Segment(1040, 2020, 1150, 2020), gap_px=0, confidence=1.0)
    row = axes_lib.axis_row("img.png", detection, (1000, 2000, 1100, 2100), (100, 100),
                            scan_size=(2480, 3507), dpi=300)
    _, rendered = AxisRenderer(metadata_csv="unused.csv").render(_square(), row)
    vectorizer = Vectorizer(axes="draw", debug_images=0)
    vectorizer.axes_source = SimpleNamespace(axes_table={"img.png": row})
    vectorizer._with_axes(TraceResult([], (0, 0, 0, 0), 0.0, 0.0), (100, 100), "img.png")

    for moved in (rendered, vectorizer.axes_table["img.png"]):
        assert moved["scale"] < 1  # the frame was extended, so the content shrank
        # Every moved point still lands on the same scan pixel.
        assert _scan_point(moved, moved["v_x1"], moved["v_y1"]) == pytest.approx((1150, 2020), abs=1e-6)
        assert _scan_point(moved, moved["h_x1"], moved["h_y1"]) == pytest.approx((1040, 2020), abs=1e-6)
        assert moved["px_per_cm"] == pytest.approx(moved["scale"] * 300 / 2.54)


# --------------------------------------------------------------------- AxisRenderer

def test_axis_renderer_extends_the_frame_and_draws_the_axes():
    renderer = AxisRenderer(metadata_csv="unused.csv")
    out, row = renderer.render(_square(), _row())
    out = np.asarray(out)

    assert out.shape == (100, 100)
    # The drawing shrank (frame extended to fit x=150) ...
    assert (out < 128).sum() < 40 * 40
    # ... and the axes are drawn where the moved row says.
    x = int(round(row["v_x1"]))
    assert row["frame_w"] == 100 and 0 < x < 100
    ys = np.arange(int(row["v_y1"]) + 1, int(row["v_y2"]))
    assert (out[ys, x] < 128).all()


def test_axis_renderer_passes_images_without_axes_through(tmp_path):
    renderer = AxisRenderer(output_path=str(tmp_path / "out"))
    renderer.axes_source = SimpleNamespace(axes_table={})
    renderer.begin_run()

    [record] = renderer.process_batch([ImageRecord("img.png", _square())])

    np.testing.assert_array_equal(np.asarray(record.image), np.asarray(_square()))
    assert renderer.no_axes == ["img.png"]


def test_axis_renderer_needs_axes_data():
    with pytest.raises(RuntimeError, match="metadata_csv"):
        AxisRenderer().process_batch([ImageRecord("img.png", _square())])


# --------------------------------------------------------------------- Vectorizer(axes="draw")

def _drawing(size=256):
    img = np.full((size, size), 255, np.uint8)
    cv2.ellipse(img, (128, 128), (60, 80), 0, 0, 360, 0, 2)
    return Image.fromarray(img)


def test_vectorizer_draws_axes_from_vectors(tmp_path):
    row = _row(frame=256, v=(300, 40, 300, 220), h=(100, 40, 300, 40))
    vectorizer = Vectorizer(output_path=str(tmp_path / "out"), axes="draw", save_svg=True, debug_images=0)
    vectorizer.axes_source = SimpleNamespace(axes_table={"img.png": row})
    vectorizer.begin_run()

    [record] = vectorizer.process_batch([ImageRecord("img.png", _drawing())])
    out = np.asarray(record.image)
    moved = vectorizer.axes_table["img.png"]

    assert out.shape == (256, 256) and set(np.unique(out)) <= {0, 255}
    x = int(round(moved["v_x1"]))
    assert 0 < x < 256 and (out[int(moved["v_y1"]) + 2:int(moved["v_y2"]) - 2, x] == 0).all()
    assert 'id="axes"' in (tmp_path / "out" / "_svg" / "img.svg").read_text()


# --------------------------------------------------------------------- Pipeline

H, W = 1400, 1000
SHARD_BOX = [400.0, 520.0, 520.0, 640.0]


def _sheet():
    img = np.full((H, W, 3), 255, np.uint8)
    cv2.line(img, (350, 500), (700, 500), (150, 150, 150), 2)
    cv2.line(img, (700, 500), (700, 900), (150, 150, 150), 2)
    cv2.ellipse(img, (460, 580), (40, 50), 0, 0, 360, (0, 0, 0), 3)
    return img


def _mocked_eraser(tmp_path):
    eraser = BackgroundEraser(record_axes=True, mode="detection", box_margin=0.0, debug_images=0,
                               output_path=str(tmp_path / "erased"))
    boxes = SimpleNamespace(xyxy=torch.tensor([SHARD_BOX]))

    def predict(source, **_):
        if isinstance(source, str):
            return [SimpleNamespace(path=str(p), boxes=boxes) for p in sorted(Path(source).iterdir())]
        return [SimpleNamespace(path=None, boxes=boxes) for _ in source]

    eraser._get_model = lambda: SimpleNamespace(predict=predict)
    return eraser


@pytest.mark.parametrize("mode", ["memory", "disk"])
def test_pipeline_draws_axes_and_writes_matching_metadata_csv(tmp_path, mode):
    source = tmp_path / "raw"
    source.mkdir()
    Image.fromarray(_sheet()).save(source / "sheet.png")
    steps = [
        _mocked_eraser(tmp_path),
        Normalizer(output_path=str(tmp_path / "normalized"), background_color=(255, 255, 255)),
        Vectorizer(output_path=str(tmp_path / "vectorized"), axes="draw", debug_images=0),
    ]

    result = Pipeline(steps).run(str(source), mode=mode, save_preprocessed="last")

    folder = tmp_path / "vectorized"
    table = pd.read_csv(folder / "extracted_metadata.csv")
    pd.testing.assert_frame_equal(table, result["metadata"].reset_index(drop=True), check_dtype=False)
    [row] = table.to_dict("records")
    image = np.asarray(Image.open(folder / "sheet.png"))
    assert row["found"] and (row["frame_w"], row["frame_h"]) == image.shape[::-1]
    # The vertical axis is drawn exactly where extracted_metadata.csv puts it.
    x = int(round(row["v_x1"]))
    ys = np.arange(int(row["v_y1"]) + 2, int(row["v_y2"]) - 2)
    assert (image[ys, x] == 0).all()
