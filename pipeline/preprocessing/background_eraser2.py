"""BackgroundEraser variant that also keeps the drawing's coordinate system."""

import cv2
import numpy as np

from pipeline.preprocessing.axis_detection import AxisDetection, detect_axes
from pipeline.preprocessing.background_eraser import BackgroundEraser


class BackgroundEraser2(BackgroundEraser):
    """Like `BackgroundEraser`, but also finds the construction coordinate
    system (vertical rotation axis + horizontal rim/base line, see
    `axis_detection.detect_axes`) on the original scan and adds it to the
    kept area: both detected segments are drawn into the shard mask and
    dilated in every direction by `axis_margin`. The square crop then
    covers the shard and the axes together, so the output image shows the
    sherd with its coordinate system.

    If no coordinate system is found on an image, it is processed exactly
    like `BackgroundEraser` would (shard only), and its name is added to
    `no_axes` (logged to MLflow as `no_axes.txt`, see `log_summary`).
    """

    def __init__(self, *args, axis_margin=0.005, **kwargs):
        """
        Args:
            axis_margin: how far the axis lines are dilated in every
                direction, as a fraction of the image's shorter side.
                0.005 (the default) is ~1 mm on an A4 scan (~12 px at
                300 dpi), enough to cover the pencil line's width and
                the small error of the straight-line fit.
            *args, **kwargs: passed to `BackgroundEraser`.
        """
        super().__init__(*args, **kwargs)
        self.axis_margin = axis_margin
        # Flat image name -> AxisDetection (or None if not found), filled
        # as images are processed, so the axes can be inspected afterwards.
        self.axes: dict[str, AxisDetection | None] = {}
        self.no_axes: list[str] = []

    def begin_run(self):
        super().begin_run()
        self.axes = {}
        self.no_axes = []

    def failure_lists(self):
        return {**super().failure_lists(), "no_axes": self.no_axes}

    def _extend_mask(self, img, item, mask, region):
        name = self._flatten_output_name(item.get("path", ""))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        axes = detect_axes(gray)
        self.axes[name] = axes
        if axes is None:
            print(f"No coordinate system found, keeping shard only: {name}")
            self.no_axes.append(name)
            return mask, region

        axis_mask = self.axis_mask(axes, img.shape[:2], self.axis_margin)
        ys, xs = np.nonzero(axis_mask)
        x1, y1, x2, y2 = region
        region = (min(x1, xs.min()), min(y1, ys.min()), max(x2, xs.max() + 1), max(y2, ys.max() + 1))
        return cv2.bitwise_or(mask, axis_mask), region

    @staticmethod
    def axis_mask(axes: AxisDetection, shape, margin) -> np.ndarray:
        """Rasterize both axis segments as 1 px lines and dilate them by
        `margin` (fraction of the shorter image side) in every direction.

        Returns:
            (H, W) uint8 mask, 255 on the dilated axes.
        """
        h, w = shape
        mask = np.zeros((h, w), dtype=np.uint8)
        for seg in (axes.vertical, axes.horizontal):
            cv2.line(
                mask,
                (int(round(seg.x1)), int(round(seg.y1))),
                (int(round(seg.x2)), int(round(seg.y2))),
                255,
                1,
            )
        radius = int(round(margin * min(h, w)))
        if radius >= 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
            mask = cv2.dilate(mask, kernel)
        return mask
