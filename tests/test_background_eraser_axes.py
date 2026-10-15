"""Tests for BackgroundEraser2: the detected coordinate system is added to
the kept area of the shard mask, in both disk and in-memory mode."""

from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from PIL import Image

from pipeline.preprocessing import BackgroundEraser, BackgroundEraser2, ImageRecord
from pipeline.preprocessing.axis_detection import detect_axes

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


def test_axis_mask_dilates_both_segments():
    axes = detect_axes(cv2.cvtColor(_sheet(), cv2.COLOR_BGR2GRAY))
    mask = BackgroundEraser2.axis_mask(axes, (H, W), margin=0.01)  # radius 10 px

    assert mask[AXIS_Y, 500] == 255 and mask[AXIS_Y + 9, 500] == 255
    assert mask[AXIS_Y + 15, 500] == 0
    assert mask[800, AXIS_X - 9] == 255 and mask[800, AXIS_X - 15] == 0


def test_output_contains_shard_and_axes(tmp_path):
    eraser = _mocked_eraser(BackgroundEraser2, tmp_path)
    [record] = eraser.process_batch([ImageRecord("sheet.png", Image.fromarray(_sheet()))])
    out = np.asarray(record.image.convert("L"))

    plain = _mocked_eraser(BackgroundEraser, tmp_path)
    [plain_record] = plain.process_batch([ImageRecord("sheet.png", Image.fromarray(_sheet()))])

    # The crop now spans the axes (vertical line ~400 px long), not just
    # the 120 px shard box.
    assert plain_record.image.size == (120, 120)
    assert out.shape[0] >= 400 and out.shape[0] == out.shape[1]
    # Pencil-gray axis pixels survive the whiting-out.
    assert np.count_nonzero((out > 100) & (out < 200)) > 500
    assert eraser.axes["sheet.png"] is not None


def test_falls_back_to_shard_only_without_axes(tmp_path):
    sheet = Image.fromarray(_sheet(with_axes=False))
    eraser = _mocked_eraser(BackgroundEraser2, tmp_path)
    plain = _mocked_eraser(BackgroundEraser, tmp_path)

    [with_axes] = eraser.process_batch([ImageRecord("sheet.png", sheet)])
    [without] = plain.process_batch([ImageRecord("sheet.png", sheet)])

    np.testing.assert_array_equal(np.asarray(with_axes.image), np.asarray(without.image))
    assert eraser.axes["sheet.png"] is None


def test_disk_run_matches_process_batch(tmp_path):
    source = tmp_path / "raw"
    source.mkdir()
    Image.fromarray(_sheet()).save(source / "sheet.png")

    eraser = _mocked_eraser(BackgroundEraser2, tmp_path)
    eraser.input_path = str(source)
    eraser.run()
    on_disk = np.asarray(Image.open(tmp_path / "out" / "sheet.png"))

    [record] = eraser.process_batch([ImageRecord("sheet.png", Image.open(source / "sheet.png").copy())])

    np.testing.assert_array_equal(np.asarray(record.image), on_disk)


def test_default_output_is_grouped_under_preprocessing_stage(tmp_path):
    eraser = BackgroundEraser2(run_id="r", debug_images=0)

    assert Path(eraser.output_path).parts[-3:] == ("r", "PreprocessingStep", "BackgroundEraser2")
