"""Per-image metadata extracted during preprocessing - the scan's size and
resolution, where the final image lies on the scan and at what scale, and
the drawing's coordinate system (rotation axis + rim/base line) - as data
that travels next to the images, instead of pixels burnt into them.

`BackgroundEraser` records one row per image (`axis_row`); with
`record_axes=True` it also finds the axes, once, on the raw scan. Later
steps that draw the axes back (`Vectorizer` with axes='draw',
`AxisRenderer`) look the row up by image name. A row holds the two segments
in the pixel coordinates of a frame of size (frame_w, frame_h): the image the
row was made for. Steps that only rescale the image uniformly (Normalizer)
need no update - a later step converts the coordinates to the image it
actually gets (`scaled_points`).

Scale: a point p (in pixels of the row's frame) lies on the raw scan at
`(offset_x, offset_y) + p / scale`; `px_per_cm = scale * scan_dpi / 2.54`
converts frame pixels to centimetres on the paper (so to real size for 1:1
drawings).

`extracted_metadata.csv` (`METADATA_CSV`, written by `write_metadata_csv`,
see `METADATA_COLUMNS`) stores the rows; `Pipeline` writes the final one
next to the preprocessed images, in their pixel coordinates.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from pipeline.preprocessing.axis_detection import AxisDetection

METADATA_CSV = "extracted_metadata.csv"

SEGMENT_COLUMNS = ["v_x1", "v_y1", "v_x2", "v_y2", "h_x1", "h_y1", "h_x2", "h_y2"]
METADATA_COLUMNS = [
    "filename",
    "frame_w", "frame_h",              # image size the coordinates below refer to
    # The raw scan: size (px) and resolution (dots per inch, from the file header).
    "scan_w", "scan_h", "scan_dpi",
    # Frame pixels per scan pixel (< 1: the image was shrunk), where the
    # frame's (0, 0) lies on the scan, and frame pixels per cm of paper.
    "scale", "offset_x", "offset_y", "px_per_cm",
    # The coordinate system (found: False also when it was not searched).
    "found", "confidence",
    *SEGMENT_COLUMNS, "origin_x", "origin_y",
    # Traceability: the same axes on the raw scan, and the square crop taken from it.
    "scan_v_x1", "scan_v_y1", "scan_v_x2", "scan_v_y2",
    "scan_h_x1", "scan_h_y1", "scan_h_x2", "scan_h_y2",
    "crop_x", "crop_y", "crop_side",
]

Segment = tuple[float, float, float, float]  # x1, y1, x2, y2


def axis_row(name: str, detection: AxisDetection | None, crop_box, frame_size,
             scan_size=None, dpi=None) -> dict:
    """Row for one image: its scan metadata, and the detected axes moved
    into the crop's frame.

    Args:
        name: image name (as the image is saved / named in memory mode).
        detection: `detect_axes` result on the raw scan, or None (not found
            or not searched).
        crop_box: (x1, y1, x2, y2) of the crop on the scan, or None (no crop).
        frame_size: (width, height) of the cropped image.
        scan_size: (width, height) of the raw scan, if known.
        dpi: the scan's resolution in dots per inch (`scan_dpi`), if known.
    """
    row = {column: np.nan for column in METADATA_COLUMNS}
    width, height = frame_size
    x0, y0 = (crop_box[0], crop_box[1]) if crop_box is not None else (0, 0)
    row.update(filename=name, found=detection is not None, frame_w=width, frame_h=height,
               crop_x=x0, crop_y=y0, crop_side=width,
               scale=1.0, offset_x=float(x0), offset_y=float(y0))
    if scan_size is not None:
        row["scan_w"], row["scan_h"] = scan_size
    if dpi:
        row["scan_dpi"] = float(dpi)
        row["px_per_cm"] = float(dpi) / 2.54
    if detection is None:
        return row
    v, h = detection.vertical, detection.horizontal
    scan = [v.x1, v.y1, v.x2, v.y2, h.x1, h.y1, h.x2, h.y2]
    row.update(zip(["scan_" + c for c in SEGMENT_COLUMNS], scan))
    row.update(zip(SEGMENT_COLUMNS, [c - (x0 if i % 2 == 0 else y0) for i, c in enumerate(scan)]))
    row.update(confidence=detection.confidence,
               origin_x=detection.origin_x - x0, origin_y=detection.origin_y - y0)
    return row


def has_axes(row: dict | None) -> bool:
    return row is not None and bool(row.get("found"))


def row_points(row: dict) -> np.ndarray:
    """(5, 2) array: vertical start/end, horizontal start/end, origin."""
    return np.array([[row["v_x1"], row["v_y1"]], [row["v_x2"], row["v_y2"]],
                     [row["h_x1"], row["h_y1"]], [row["h_x2"], row["h_y2"]],
                     [row["origin_x"], row["origin_y"]]], dtype=np.float64)


def scaled_points(row: dict, width: int, height: int) -> np.ndarray:
    """`row_points` in the pixel coordinates of a (width, height) image
    derived from the row's frame by uniform scaling."""
    return row_points(row) * (width / row["frame_w"], height / row["frame_h"])


def segments_of(points: np.ndarray) -> list[Segment]:
    """The two segments from `row_points`-shaped points."""
    return [tuple(points[0]) + tuple(points[1]), tuple(points[2]) + tuple(points[3])]


def extended_frame(width: int, height: int, segments: list[Segment], margin: float) -> tuple[float, float, float, float]:
    """The smallest window (x0, y0, w, h) - in the image's pixel coordinates,
    with the image's aspect ratio - that contains the whole image and the
    segments plus `margin` (fraction of the image size) around them. Equal
    to the image itself when the segments already lie inside it."""
    pad = margin * max(width, height)
    xs = [x for seg in segments for x in (seg[0], seg[2])]
    ys = [y for seg in segments for y in (seg[1], seg[3])]
    x0, x1 = min(0.0, min(xs) - pad), max(float(width), max(xs) + pad)
    y0, y1 = min(0.0, min(ys) - pad), max(float(height), max(ys) + pad)
    w, h = x1 - x0, y1 - y0
    aspect = width / height
    if w / h < aspect:  # too narrow: widen around the centre
        x0 -= (h * aspect - w) / 2
        w = h * aspect
    else:
        y0 -= (w / aspect - h) / 2
        h = w / aspect
    return x0, y0, w, h


def to_output(points: np.ndarray, frame, width: int) -> np.ndarray:
    """Map (N, 2) points from input-image coordinates into an output image of
    width `width` showing the window `frame` = (x0, y0, w, h)."""
    x0, y0, w, _ = frame
    return (np.asarray(points, dtype=np.float64) - (x0, y0)) * (width / w)


def moved_row(row: dict, points: np.ndarray | None, frame_size, factor: float, shift=(0.0, 0.0)) -> dict:
    """Copy of `row` for a new frame of `frame_size`, whose pixel coordinates
    are `(p + shift) * factor` for a point p in the row's frame: the
    segments and origin are replaced by `points` (`row_points` layout,
    already moved that way; None for a row without axes) and the scale and
    offset are updated to match."""
    new = dict(row)
    new["frame_w"], new["frame_h"] = frame_size
    scale = row["scale"]
    new["scale"] = scale * factor
    new["offset_x"] = row["offset_x"] - shift[0] / scale
    new["offset_y"] = row["offset_y"] - shift[1] / scale
    new["px_per_cm"] = row["px_per_cm"] * factor
    if points is not None:
        (vx1, vy1), (vx2, vy2), (hx1, hy1), (hx2, hy2), (ox, oy) = points
        new.update(v_x1=vx1, v_y1=vy1, v_x2=vx2, v_y2=vy2, h_x1=hx1, h_y1=hy1, h_x2=hx2, h_y2=hy2,
                   origin_x=ox, origin_y=oy)
    return new


def draw_segments(canvas: np.ndarray, segments: list[Segment], thickness: int, antialias: bool = False) -> None:
    """Draw segments in black onto a white uint8 canvas, in place."""
    line_type = cv2.LINE_AA if antialias else cv2.LINE_8
    for x1, y1, x2, y2 in segments:
        cv2.line(canvas, (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2))),
                 0, max(1, int(thickness)), line_type)


def write_metadata_csv(rows: dict[str, dict], path, sizes: dict[str, tuple[int, int]] | None = None) -> Path:
    """Write `metadata_dataframe(rows, sizes)` as a CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata_dataframe(rows, sizes).to_csv(path, index=False)
    return path


def metadata_dataframe(rows: dict[str, dict], sizes: dict[str, tuple[int, int]] | None = None) -> pd.DataFrame:
    """The rows as a table. With `sizes` (name -> final image size), only
    those images, converted to their final pixel coordinates (the images
    are uniform rescales of the rows' frames)."""
    out = []
    for name, row in rows.items():
        if sizes is not None:
            if name not in sizes:
                continue
            width, height = sizes[name]
            points = scaled_points(row, width, height) if has_axes(row) else None
            row = moved_row(row, points, (width, height), width / row["frame_w"])
        out.append(row)
    return pd.DataFrame(out, columns=METADATA_COLUMNS)


def read_metadata_csv(path) -> dict[str, dict]:
    """Rows of an extracted_metadata.csv, by image name."""
    table = pd.read_csv(path)
    table["found"] = table["found"].astype(bool)
    return {row["filename"]: row for row in table.to_dict("records")}
