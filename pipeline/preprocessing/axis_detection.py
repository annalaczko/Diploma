"""Finds the construction coordinate system on a raw A4 pottery scan: the
long vertical rotation axis, the horizontal rim/base line and their
intersection. Ported from giappetto/pottery_axes_batch.py (which also
reports quadrants and writes a CSV); used by `BackgroundEraser(record_axes=True)`.

The lines are found with long morphological openings rather than Hough
segments: an opening with a long, thin kernel keeps only straight ruler
lines, so handwriting and the curved profile drop out, and faint pencil
lines are recovered as one component instead of many Hough pieces.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

# Detection runs on a copy with this long side; results are mapped back.
WORK_LONG_SIDE = 1600

# Threshold on the 0-255 normalized blackhat image for pencil line pixels.
# Deliberately low: Otsu is pulled up by dark ink handwriting and then drops
# faint pencil axes (2012 survey sheets). Paper noise that passes it is
# removed by the long line openings.
LINE_THRESHOLD = 15

# Score penalty per unit of ink fraction beside a vertical candidate.
CLUTTER_WEIGHT = 1.5


@dataclass
class Segment:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def length(self) -> float:
        return math.hypot(self.x2 - self.x1, self.y2 - self.y1)

    def scaled(self, s: float) -> "Segment":
        return Segment(self.x1 * s, self.y1 * s, self.x2 * s, self.y2 * s)


@dataclass
class AxisDetection:
    """Pixel coordinates in the original image."""
    origin_x: float
    origin_y: float
    vertical: Segment  # y1 <= y2
    horizontal: Segment  # x1 <= x2
    gap_px: float  # distance between the two segments at the origin
    confidence: float


def blackhat_edges(gray: np.ndarray) -> np.ndarray:
    """Enhance thin, dark pencil or ink lines on a light paper background."""
    k = max(15, int(round(min(gray.shape[:2]) * 0.018)))
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    return cv2.normalize(blackhat, None, 0, 255, cv2.NORM_MINMAX)


def _extract_lines(binary: np.ndarray, orientation: str, min_len: int) -> List[Segment]:
    """Keep only long straight strokes of one orientation as segments."""
    if orientation == "h":
        open_k = cv2.getStructuringElement(cv2.MORPH_RECT, (min_len, 1))
        # Bridge small breaks in faint pencil lines and allow a little
        # scanner skew (the line may wander by a pixel or two).
        close_k = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
        thick_k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    else:
        open_k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, min_len))
        close_k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
        thick_k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))

    work = cv2.dilate(binary, thick_k)
    work = cv2.morphologyEx(work, cv2.MORPH_CLOSE, close_k)
    lines = cv2.morphologyEx(work, cv2.MORPH_OPEN, open_k)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(lines, 8)
    segments: List[Segment] = []
    for i in range(1, n):
        x, y, w, h, _area = map(int, stats[i])
        ys, xs = np.where(labels[y:y + h, x:x + w] == i)
        xs = xs + x
        ys = ys + y
        # Fit the line as a linear function to tolerate a slight skew.
        if orientation == "h":
            if w < min_len:
                continue
            a, b = np.polyfit(xs, ys, 1) if np.ptp(xs) > 0 else (0.0, float(ys.mean()))
            x1, x2 = float(xs.min()), float(xs.max())
            segments.append(Segment(x1, a * x1 + b, x2, a * x2 + b))
        else:
            if h < min_len:
                continue
            a, b = np.polyfit(ys, xs, 1) if np.ptp(ys) > 0 else (0.0, float(xs.mean()))
            y1, y2 = float(ys.min()), float(ys.max())
            segments.append(Segment(a * y1 + b, y1, a * y2 + b, y2))
    return segments


def _x_at(v: Segment, y: float) -> float:
    if v.y2 == v.y1:
        return 0.5 * (v.x1 + v.x2)
    t = (y - v.y1) / (v.y2 - v.y1)
    return v.x1 + t * (v.x2 - v.x1)


def _y_at(h: Segment, x: float) -> float:
    if h.x2 == h.x1:
        return 0.5 * (h.y1 + h.y2)
    t = (x - h.x1) / (h.x2 - h.x1)
    return h.y1 + t * (h.y2 - h.y1)


def _intersection(v: Segment, h: Segment) -> Tuple[float, float]:
    # Two fixed-point steps are exact enough for near-axis-parallel lines.
    y = 0.5 * (h.y1 + h.y2)
    x = _x_at(v, y)
    y = _y_at(h, x)
    x = _x_at(v, y)
    return x, y


def _outside(value: float, lo: float, hi: float) -> float:
    """How far value lies outside [lo, hi] (0 if inside)."""
    return max(0.0, lo - value, value - hi)


def _vertical_clutter(binary: np.ndarray, v: Segment) -> float:
    """Fraction of ink pixels in a band beside a vertical line.

    The rotation axis runs through empty paper, while a vertical drawn along
    the profile (e.g. the rim edge on 2012 survey sheets) has the profile
    strokes right next to it. The line itself is excluded from the band.
    """
    h, w = binary.shape
    band = max(6, int(0.025 * w))
    core = 4
    y0, y1 = max(0, int(v.y1)), min(h, int(v.y2) + 1)
    xc = int(round(0.5 * (v.x1 + v.x2)))
    left = binary[y0:y1, max(0, xc - band):max(0, xc - core)]
    right = binary[y0:y1, min(w, xc + core + 1):min(w, xc + band + 1)]
    area = left.size + right.size
    if area == 0:
        return 0.0
    return float((np.count_nonzero(left) + np.count_nonzero(right)) / area)


def detect_axes(gray: np.ndarray) -> Optional[AxisDetection]:
    """Find the vertical axis, the horizontal line and their intersection
    on a grayscale scan. Returns None when no meeting pair of lines exists.
    """
    H, W = gray.shape[:2]
    scale = WORK_LONG_SIDE / float(max(H, W))
    small = cv2.resize(gray, (int(round(W * scale)), int(round(H * scale))), interpolation=cv2.INTER_AREA)
    h, w = small.shape

    _, binary = cv2.threshold(blackhat_edges(small), LINE_THRESHOLD, 255, cv2.THRESH_BINARY)

    # Blank out the sheet border: scanner edges, shadows and punch holes.
    bx, by = int(0.05 * w), int(0.02 * h)
    binary[:by, :] = 0
    binary[h - by:, :] = 0
    binary[:, :bx] = 0
    binary[:, w - int(0.02 * w):] = 0

    min_len = int(0.06 * min(h, w))
    verticals = _extract_lines(binary, "v", min_len)
    horizontals = _extract_lines(binary, "h", min_len)
    if not verticals or not horizontals:
        return None

    # Score every vertical/horizontal pair: long lines that actually meet
    # (or nearly meet) win. Handwriting underlines rarely meet a long
    # perpendicular line.
    diag = math.hypot(h, w)
    max_gap = 0.06 * min(h, w)
    clutter = {id(v): _vertical_clutter(binary, v) for v in verticals}
    best = None
    for v in verticals:
        for hz in horizontals:
            x, y = _intersection(v, hz)
            gap = math.hypot(_outside(x, hz.x1, hz.x2), _outside(y, v.y1, v.y2))
            if gap > max_gap:
                continue
            score = (v.length + hz.length) / diag - 3.0 * gap / diag - CLUTTER_WEIGHT * clutter[id(v)]
            if best is None or score > best[0]:
                best = (score, v, hz, x, y, gap)
    if best is None:
        return None

    _, v, hz, x, y, gap = best
    # Confidence: both lines reasonably long and meeting cleanly.
    v_norm = min(1.0, v.length / (0.15 * h))
    h_norm = min(1.0, hz.length / (0.15 * w))
    gap_norm = max(0.0, 1.0 - gap / max_gap)
    confidence = float(0.4 * v_norm + 0.4 * h_norm + 0.2 * gap_norm)

    inv = 1.0 / scale
    return AxisDetection(
        origin_x=x * inv,
        origin_y=y * inv,
        vertical=v.scaled(inv),
        horizontal=hz.scaled(inv),
        gap_px=gap * inv,
        confidence=confidence,
    )
