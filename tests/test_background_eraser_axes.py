"""Tests for BackgroundEraser(record_axes=True): the output shows the shard
only, and the coordinate system is recorded as data (metadata rows /
extracted_metadata.csv),
in both disk and in-memory mode."""

from functools import partial
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from pipeline.preprocessing import BackgroundEraser, ImageRecord
from pipeline.preprocessing.axis_detection import detect_axes

AxesEraser = partial(BackgroundEraser, record_axes=True)

# A4-like sheet (portrait), pencil-gray axes forming a ⅂ corner at
# (AXIS_X, AXIS_Y) and a dark "shard" blob to the lower left of it.
H, W = 1400, 1000
AXIS_X, AXIS_Y = 700, 500
SHARD_BOX = (400.0, 520.0, 520.0, 640.0)


def _sheet(with_axes=True):
    img = np.full((H, W, 3), 255, np.uint8)
    if with_axes:
        cv2.line(img, (350, AXIS_Y), (AXIS_X, AXIS_Y), (150, 150, 150), 2)  # horizontal, going left
        cv2.line(img, (AXIS_X, AXIS_Y), (AXIS_X, 900), (150, 150, 150), 2)  # vertical, going down
    x1, y1, x2, y2 = map(int, SHARD_BOX)
    cv2.ellipse(img, ((x1 + x2) // 2, (y1 + y2) // 2), (40, 50), 0, 0, 360, (0, 0, 0), 3)
    return img


def _mocked_eraser(cls, tmp_path, **kwargs):
    eraser = cls(mode="detection", box_margin=0.0, debug_images=0,
                 output_path=str(tmp_path / "out"), **kwargs)
    found = SimpleNamespace(
        path=None, boxes=SimpleNamespace(xyxy=torch.tensor([list(SHARD_BOX)]))
    )

    def predict(source, **_):
        if isinstance(source, str):  # disk mode: flat symlink folder
            paths = sorted(Path(source).iterdir())
            return [SimpleNamespace(path=str(p), boxes=found.boxes) for p in paths]
        return [found for _ in source]

    eraser._get_model = lambda: SimpleNamespace(predict=predict)
    return eraser


def test_detect_axes_finds_corner():
    axes = detect_axes(cv2.cvtColor(_sheet(), cv2.COLOR_BGR2GRAY))

    assert axes is not None
    assert abs(axes.origin_x - AXIS_X) < 4
    assert abs(axes.origin_y - AXIS_Y) < 4
    assert axes.vertical.y2 > 850  # runs down to the end of the line
    assert axes.horizontal.x1 < 400  # runs left past the shard


def test_detect_axes_returns_none_without_lines():
    assert detect_axes(cv2.cvtColor(_sheet(with_axes=False), cv2.COLOR_BGR2GRAY)) is None


def test_output_is_the_shard_only_with_axes_recorded(tmp_path):
    eraser = _mocked_eraser(AxesEraser, tmp_path)
    plain = _mocked_eraser(BackgroundEraser, tmp_path)
    eraser.begin_run()

    [record] = eraser.process_batch([ImageRecord("sheet.png", Image.fromarray(_sheet()))])
    [plain_record] = plain.process_batch([ImageRecord("sheet.png", Image.fromarray(_sheet()))])

    # Same pixels as the plain eraser: cropped to the shard, axes not drawn.
    np.testing.assert_array_equal(np.asarray(record.image), np.asarray(plain_record.image))
    assert record.image.size == (120, 120)
    # The axes are recorded in the crop's coordinates (outside the crop here).
    row = eraser.axes_table["sheet.png"]
    assert row["found"] and (row["frame_w"], row["frame_h"]) == (120, 120)
    assert (row["crop_x"], row["crop_y"]) == (400, 520)
    assert abs(row["v_x1"] - (AXIS_X - 400)) < 4 and abs(row["origin_y"] - (AXIS_Y - 520)) < 4
    assert abs(row["scan_v_x1"] - AXIS_X) < 4


def test_missing_axes_are_recorded(tmp_path):
    eraser = _mocked_eraser(AxesEraser, tmp_path)
    eraser.begin_run()

    eraser.process_batch([ImageRecord("sheet.png", Image.fromarray(_sheet(with_axes=False)))])

    assert not eraser.axes_table["sheet.png"]["found"]
    assert eraser.no_axes == ["sheet.png"]


def test_disk_run_matches_process_batch_and_writes_metadata_csv(tmp_path):
    source = tmp_path / "raw"
    source.mkdir()
    Image.fromarray(_sheet()).save(source / "sheet.png", dpi=(150, 150))

    eraser = _mocked_eraser(AxesEraser, tmp_path)
    eraser.input_path = str(source)
    eraser.run()
    on_disk = np.asarray(Image.open(tmp_path / "out" / "sheet.png"))
    table = pd.read_csv(tmp_path / "out" / "extracted_metadata.csv")

    [record] = eraser.process_batch([ImageRecord("sheet.png", Image.open(source / "sheet.png").copy())])

    np.testing.assert_array_equal(np.asarray(record.image), on_disk)
    assert table["filename"].tolist() == ["sheet.png"] and table["found"].tolist() == [True]
    [row] = table.to_dict("records")
    assert (row["scan_w"], row["scan_h"], row["scan_dpi"], row["scale"]) == (W, H, 150, 1)
    assert (row["offset_x"], row["offset_y"]) == (row["crop_x"], row["crop_y"])


def test_without_record_axes_only_the_metadata_is_recorded(tmp_path):
    eraser = _mocked_eraser(BackgroundEraser, tmp_path)
    eraser.begin_run()
    image = Image.fromarray(_sheet())
    image.info["dpi"] = (300, 300)

    eraser.process_batch([ImageRecord("sheet.png", image)])

    row = eraser.axes_table["sheet.png"]
    assert not row["found"] and np.isnan(row["v_x1"])  # axes not searched
    assert (row["scan_w"], row["scan_dpi"]) == (W, 300) and row["px_per_cm"] == pytest.approx(300 / 2.54)
    assert not eraser.provides_axes and eraser.provides_metadata
    assert eraser.failure_lists() == {"no_shard": []} and eraser.no_axes == []


# --------------------------------------------------------------------- speed-ups
# The parallel / crop-only / grayscale code paths must not change the output.

def _segmenting_eraser(tmp_path, **kwargs):
    """Segmentation-mode eraser with a fixed YOLO mask around the shard."""
    eraser = BackgroundEraser(mode="segmentation", mask_margin=0.2, debug_images=0, record_axes=True,
                              output_path=str(tmp_path / "out"), **kwargs)
    x1, y1, x2, y2 = SHARD_BOX
    polygon = np.array([[x1, y1], [x2, y1 + 30], [x2 - 10, y2], [x1 + 20, y2 - 5]], np.float32)
    found = SimpleNamespace(masks=SimpleNamespace(xy=[polygon]),
                            boxes=SimpleNamespace(xyxy=torch.tensor([list(SHARD_BOX)])))
    eraser._get_model = lambda: SimpleNamespace(predict=lambda source, **_: [found for _ in source])
    return eraser


def test_crop_only_mask_matches_whole_sheet_mask(tmp_path):
    img = _sheet()
    eraser = _segmenting_eraser(tmp_path)
    [record] = eraser.process_batch([ImageRecord("sheet.png", Image.fromarray(img))])

    # Reference: rasterize and grow the mask on the whole sheet, then crop.
    x1, y1, x2, y2 = SHARD_BOX
    polygon = np.array([[x1, y1], [x2, y1 + 30], [x2 - 10, y2], [x1 + 20, y2 - 5]], np.int32)
    mask = np.zeros((H, W), np.uint8)
    cv2.fillPoly(mask, [polygon], 255)
    radius = eraser._dilate_radius(np.ptp(polygon[:, 0]) + 1, np.ptp(polygon[:, 1]) + 1, 0.2)
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2))
    ys, xs = np.nonzero(mask)
    side = max(xs.max() + 1 - xs.min(), ys.max() + 1 - ys.min())
    assert record.image.size == (2 * round(side / 2),) * 2
    whitened = np.where(mask[..., None] > 0, img, 255)
    sq_x1 = round((xs.min() + xs.max() + 1) / 2) - round(side / 2)
    sq_y1 = round((ys.min() + ys.max() + 1) / 2) - round(side / 2)
    size = record.image.size[0]
    np.testing.assert_array_equal(np.asarray(record.image), whitened[sq_y1:sq_y1 + size, sq_x1:sq_x1 + size])


def test_parallel_and_grayscale_give_the_same_result(tmp_path):
    sheets = [ImageRecord(f"sheet{i}.png", Image.fromarray(_sheet(with_axes=i % 2 == 0))) for i in range(6)]
    serial = _segmenting_eraser(tmp_path, num_workers=1)
    parallel = _segmenting_eraser(tmp_path, num_workers=4)
    gray = _segmenting_eraser(tmp_path, num_workers=4)

    out_serial = serial.process_batch(sheets)
    out_parallel = parallel.process_batch(sheets)
    out_gray = gray.process_batch([ImageRecord(r.name, r.image.convert("L")) for r in sheets])

    assert all(r.image.mode == "L" for r in out_gray)  # grayscale stays one channel
    for a, b, g in zip(out_serial, out_parallel, out_gray):
        np.testing.assert_array_equal(np.asarray(a.image), np.asarray(b.image))
        np.testing.assert_array_equal(np.asarray(a.image.convert("L")), np.asarray(g.image))
    assert serial.axes_table == parallel.axes_table == gray.axes_table
    assert serial.no_axes == parallel.no_axes == gray.no_axes == ["sheet1.png", "sheet3.png", "sheet5.png"]
