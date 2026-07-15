"""Tests for preprocessing steps.

BackgroundEraser uses YOLO + OpenCV, so heavy external calls are mocked.
"""

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pipeline.preprocessing.base import PreprocessingStep
from pipeline.preprocessing.steps import BackgroundEraser


# --- abstract base enforcement ---

def test_cannot_instantiate_preprocessing_step_directly():
    with pytest.raises(TypeError):
        PreprocessingStep()  # type: ignore[abstract]


# --- BackgroundEraser: construction ---

class TestBackgroundEraserInit:
    def test_default_mode(self):
        be = BackgroundEraser(image_paths=[])
        assert be.mode == "segmentation"

    def test_custom_mode_stored(self):
        be = BackgroundEraser(image_paths=[], mode="detection")
        assert be.mode == "detection"

    def test_image_paths_stored(self):
        paths = ["/img/a.jpg", "/img/b.jpg"]
        be = BackgroundEraser(image_paths=paths)
        assert be.image_paths == paths

    def test_model_path_stored(self):
        be = BackgroundEraser(image_paths=[], model="/some/model.pt")
        assert be.model == "/some/model.pt"


# --- BackgroundEraser: _detect_shards ---

class TestDetectShards:
    def _make_mock_result(self, mask_xy):
        r = MagicMock()
        r.masks.xy = mask_xy
        return r

    def test_returns_list_with_path_and_masks(self):
        paths = ["/img/a.jpg"]
        mask = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)
        mock_result = self._make_mock_result([mask])

        be = BackgroundEraser(image_paths=paths, model="fake.pt")

        with patch.object(be, "model", create=True):
            mock_model = MagicMock()
            mock_model.predict.return_value = [mock_result]
            results = be._detect_shards(paths, mock_model)

        assert len(results) == 1
        assert results[0]["path"] == paths[0]
        assert results[0]["masks"] is mock_result.masks.xy

    def test_detection_mode_raises_not_implemented(self):
        be = BackgroundEraser(image_paths=[], mode="detection", model="fake.pt")
        mock_model = MagicMock()
        mock_model.predict.return_value = [MagicMock()]

        with pytest.raises(NotImplementedError):
            be._detect_shards(["img.jpg"], mock_model)


# --- BackgroundEraser: _clear_images ---

class TestClearImages:
    def _make_item(self, shape=(50, 50, 3)):
        """Create a fake image dict entry with a simple rectangular mask."""
        h, w = shape[:2]
        polygon = np.array([[5, 5], [w - 5, 5], [w - 5, h - 5], [5, h - 5]], dtype=np.float32)
        img = np.zeros(shape, dtype=np.uint8)
        return {"path": "/fake/img.jpg", "masks": [polygon]}, img

    def test_returns_list_of_dicts(self):
        be = BackgroundEraser(image_paths=[], model="fake.pt")
        item, fake_img = self._make_item()

        with patch("pipeline.preprocessing.steps.cv2.imread", return_value=fake_img):
            results = be._clear_images([item])

        assert isinstance(results, list)
        assert results[0]["path"] == item["path"]
        assert "image" in results[0]

    def test_output_image_same_shape_as_input(self):
        be = BackgroundEraser(image_paths=[], model="fake.pt")
        item, fake_img = self._make_item((40, 60, 3))

        with patch("pipeline.preprocessing.steps.cv2.imread", return_value=fake_img):
            results = be._clear_images([item])

        assert results[0]["image"].shape == fake_img.shape

    def test_skips_unreadable_image(self):
        be = BackgroundEraser(image_paths=[], model="fake.pt")
        item, _ = self._make_item()

        with patch("pipeline.preprocessing.steps.cv2.imread", return_value=None):
            results = be._clear_images([item])

        assert results == []

    def test_saves_to_output_dir(self, tmp_path):
        be = BackgroundEraser(image_paths=[], model="fake.pt")
        item, fake_img = self._make_item()

        with patch("pipeline.preprocessing.steps.cv2.imread", return_value=fake_img), \
             patch("pipeline.preprocessing.steps.cv2.imwrite") as mock_write:
            be._clear_images([item], output_dir=str(tmp_path))

        mock_write.assert_called_once()

    def test_no_write_when_output_dir_is_none(self):
        be = BackgroundEraser(image_paths=[], model="fake.pt")
        item, fake_img = self._make_item()

        with patch("pipeline.preprocessing.steps.cv2.imread", return_value=fake_img), \
             patch("pipeline.preprocessing.steps.cv2.imwrite") as mock_write:
            be._clear_images([item], output_dir=None)

        mock_write.assert_not_called()
