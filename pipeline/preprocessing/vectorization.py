"""Vectorization of cropped, normalized pottery shard drawings: the line
work is binarized, skeletonized and traced into smoothed polylines (no OCR
or learned model). Ported from giappetto/shard_svg_batch.py; used by the
`Vectorizer` preprocessing step.

Main entry point: `trace_image(gray) -> TraceResult` (raises RuntimeError
when nothing can be traced); `render_polylines` draws the result back to a
clean raster, `write_svg` / `write_debug` save it for inspection.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import cv2
import numpy as np
from skimage.morphology import skeletonize

Point = Tuple[int, int]  # x, y in pixels


@dataclass
class TraceResult:
    polylines: List[np.ndarray]
    drawing_bbox_px: Tuple[int, int, int, int]
    threshold_value: float
    ink_fraction: float  # foreground pixels / image pixels after cleaning


def blackhat_edges(gray: np.ndarray) -> np.ndarray:
    """Enhance thin, dark pencil or ink lines on a light paper background.

    The kernel only has to be wider than a stroke. On a 256 px image strokes
    are 1-3 px wide, so a much smaller kernel than on the A4 scans is enough
    and avoids merging the two walls of thin shards.
    """
    min_dim = min(gray.shape[:2])
    k = max(7, int(round(min_dim * 0.035)))
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    blackhat = cv2.normalize(blackhat, None, 0, 255, cv2.NORM_MINMAX)
    return blackhat


def segment_drawing(
    gray: np.ndarray,
    border_px: int = 1,
    min_threshold: float = 25.0,
    low_frac: float = 0.3,
) -> Tuple[np.ndarray, float]:
    """Binarize the profile strokes and remove noise.

    Noise handling:
    * a light median blur removes salt-and-pepper speckles before thresholding;
    * Otsu's threshold is floored at `min_threshold`, because on a nearly blank
      or very clean image Otsu would otherwise split the paper texture itself;
    * hysteresis thresholding keeps faint strokes only where they connect to
      a clearly dark one;
    * connected components smaller than a size threshold are dropped.
    """
    h, w = gray.shape
    den = cv2.medianBlur(gray, 3)
    bh = blackhat_edges(den)
    otsu, _ = cv2.threshold(bh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    threshold_value = max(float(otsu), min_threshold)

    # Hysteresis: Otsu alone keeps only the dark strokes and breaks faint
    # pencil lines into pieces. Weak pixels are kept when they are connected
    # to a strong one; isolated weak noise is still rejected. The low
    # threshold is set from the background noise level (median + k * MAD),
    # but never below a fraction of the high one.
    bg = bh[bh < threshold_value]
    med = float(np.median(bg)) if bg.size else 0.0
    mad = float(np.median(np.abs(bg - med))) if bg.size else 0.0
    low = min(threshold_value, max(low_frac * threshold_value, med + 6.0 * mad + 1.0))
    weak = (bh >= low).astype(np.uint8)
    strong = bh >= threshold_value
    n, labels = cv2.connectedComponents(weak, connectivity=8)
    keep = np.zeros(n, dtype=bool)
    keep[np.unique(labels[strong])] = True
    keep[0] = False
    binary = (keep[labels] * 255).astype(np.uint8)

    # Scanner/crop artefacts often sit exactly on the image edge.
    if border_px > 0:
        binary[:border_px, :] = 0
        binary[-border_px:, :] = 0
        binary[:, :border_px] = 0
        binary[:, -border_px:] = 0

    # A light close bridges tiny gaps caused by faint pencil strokes.
    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )

    # Reject small specks; the profile is one or a few long strokes.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    clean = np.zeros_like(binary)
    min_area = max(12, int(0.0004 * h * w))
    min_extent = max(6, int(0.04 * min(h, w)))
    for i in range(1, n):
        _, _, cw, ch, area = map(int, stats[i])
        if area < min_area:
            continue
        if max(cw, ch) < min_extent:
            continue
        clean[labels == i] = 255

    return clean, threshold_value


def pixel_neighbors(p: Point, pixels: set[Point]) -> Iterable[Tuple[Point, float]]:
    x, y = p
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            q = (x + dx, y + dy)
            if q in pixels:
                yield q, (math.sqrt(2.0) if dx and dy else 1.0)


def dijkstra_farthest(start: Point, pixels: set[Point]) -> Tuple[Point, dict[Point, Point], dict[Point, float]]:
    dist = {start: 0.0}
    prev: dict[Point, Point] = {}
    heap = [(0.0, start)]
    while heap:
        d, p = heapq.heappop(heap)
        if d != dist.get(p):
            continue
        for q, cost in pixel_neighbors(p, pixels):
            nd = d + cost
            if nd < dist.get(q, float("inf")):
                dist[q] = nd
                prev[q] = p
                heapq.heappush(heap, (nd, q))
    farthest = max(dist, key=dist.get)
    return farthest, prev, dist


def longest_path(pixels: set[Point]) -> List[Point]:
    """Geodesic diameter of a connected pixel set (two Dijkstra sweeps)."""
    seed = next(iter(pixels))
    a, _, _ = dijkstra_farthest(seed, pixels)
    b, prev, _ = dijkstra_farthest(a, pixels)
    path = [b]
    p = b
    while p != a and p in prev:
        p = prev[p]
        path.append(p)
    path.reverse()
    return path


def component_paths(component_mask: np.ndarray, min_branch_px: int) -> List[np.ndarray]:
    """Decompose one skeleton component into paths.

    Unlike the A4 script, which keeps only the longest path per component, a
    cropped shard often has real branches (e.g. a rim line meeting a wall).
    The longest path is taken first, then the longest remaining branches are
    taken repeatedly. Branches shorter than `min_branch_px` are treated as
    skeleton spurs caused by noise and discarded.
    """
    ys, xs = np.where(component_mask > 0)
    remaining = set(zip(xs.tolist(), ys.tolist()))
    paths: List[np.ndarray] = []
    taken: set[Point] = set()
    while len(remaining) >= min_branch_px:
        # Work per connected piece of what is left.
        seed = next(iter(remaining))
        piece = {seed}
        stack = [seed]
        while stack:
            p = stack.pop()
            for q, _ in pixel_neighbors(p, remaining):
                if q not in piece:
                    piece.add(q)
                    stack.append(q)
        remaining -= piece
        if len(piece) < min_branch_px:
            continue
        path = longest_path(piece)
        if len(path) < min_branch_px:
            continue
        # Reattach a branch to the path it grew out of, so there is no gap.
        for end in (0, -1):
            x, y = path[end]
            for q, _ in pixel_neighbors((x, y), taken):
                if end == 0:
                    path.insert(0, q)
                else:
                    path.append(q)
                break
        paths.append(np.asarray(path, dtype=np.float32))
        taken.update(path)
        rest = piece.difference(path)
        remaining |= rest
    return paths


def smooth_polyline(points: np.ndarray, window: int = 5) -> np.ndarray:
    """Moving-average smoothing that removes the pixel staircase of the
    skeleton while keeping both endpoints fixed."""
    if len(points) < window + 2:
        return points
    pad = window // 2
    padded = np.pad(points, ((pad, pad), (0, 0)), mode="edge")
    kernel = np.ones(window, dtype=np.float32) / window
    out = np.stack(
        [np.convolve(padded[:, k], kernel, mode="valid") for k in range(2)], axis=1
    )
    out[0], out[-1] = points[0], points[-1]
    return out.astype(np.float32)


def _end_direction(p: np.ndarray, at_start: bool, k: int = 6) -> np.ndarray:
    """Unit vector pointing outwards from one end of a polyline."""
    if at_start:
        a, b = p[min(k, len(p) - 1)], p[0]
    else:
        a, b = p[max(0, len(p) - 1 - k)], p[-1]
    d = b - a
    n = float(np.linalg.norm(d))
    return d / n if n > 0 else d


def bridge_gaps(lines: List[np.ndarray], max_gap_px: float, max_angle_deg: float = 50.0) -> List[np.ndarray]:
    """Join polylines whose ends are close and point towards each other.

    Pencil strokes on the scans are often interrupted for a few pixels; the
    two halves of such a line should become a single path. Two ends are joined
    when they are within `max_gap_px` and the gap continues both strokes'
    directions (within `max_angle_deg`), so parallel walls of a thin shard are
    not accidentally connected.
    """
    lines = [p.copy() for p in lines]
    cos_lim = math.cos(math.radians(max_angle_deg))
    while True:
        best = None
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                for si in (True, False):
                    for sj in (True, False):
                        pi = lines[i][0] if si else lines[i][-1]
                        pj = lines[j][0] if sj else lines[j][-1]
                        gap = pj - pi
                        dist = float(np.linalg.norm(gap))
                        if dist > max_gap_px:
                            continue
                        if dist > 1.5:
                            u = gap / dist
                            if np.dot(_end_direction(lines[i], si), u) < cos_lim:
                                continue
                            if np.dot(_end_direction(lines[j], sj), -u) < cos_lim:
                                continue
                        if best is None or dist < best[0]:
                            best = (dist, i, j, si, sj)
        if best is None:
            return lines
        _, i, j, si, sj = best
        a = lines[i][::-1] if si else lines[i]  # a ends at the joined end
        b = lines[j] if sj else lines[j][::-1]  # b starts at the joined end
        merged = np.concatenate([a, b], axis=0)
        lines = [p for k, p in enumerate(lines) if k not in (i, j)] + [merged]


def simplify_polyline(points: np.ndarray, epsilon_px: float = 0.6) -> np.ndarray:
    if len(points) < 3:
        return points
    curve = points.reshape(-1, 1, 2).astype(np.float32)
    approx = cv2.approxPolyDP(curve, epsilon_px, False)
    return approx[:, 0, :]


def polyline_length(p: np.ndarray) -> float:
    return float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)))


def _is_straight(p: np.ndarray, max_dev: float) -> bool:
    """All points of `p` lie within `max_dev` px of the chord p[0] -> p[-1]."""
    chord = p[-1] - p[0]
    chord_len = float(np.linalg.norm(chord))
    if chord_len == 0:
        return True
    rel = p - p[0]
    dev = np.abs(rel[:, 0] * chord[1] - rel[:, 1] * chord[0]) / chord_len
    return float(dev.max()) <= max_dev


def _off_axis_deg(v: np.ndarray) -> float:
    angle = abs(math.degrees(math.atan2(v[1], v[0]))) % 180.0
    return min(angle, abs(angle - 90.0), 180.0 - angle)


def _off_horizontal_deg(v: np.ndarray) -> float:
    angle = abs(math.degrees(math.atan2(v[1], v[0]))) % 180.0
    return min(angle, 180.0 - angle)


def _rms_residual(p: np.ndarray) -> float:
    """RMS distance of the points from their best-fit (total least squares) line."""
    c = p - p.mean(axis=0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    return float(np.sqrt(np.mean((c @ vt[1]) ** 2)))


def _straight_runs(
    p: np.ndarray, min_len: float, max_dev: float, max_angle_deg: float, max_rms_px: float
) -> List[Tuple[int, int]]:
    """(i, j) index ranges of `p` that are long, straight, close to
    horizontal/vertical and ruler-even: candidate construction-axis runs."""
    n = len(p)
    runs = []
    i = 0
    while i < n - 1:
        # Longest straight run starting at i (binary search; straightness is
        # monotone enough along a path for this to be reliable).
        lo, hi = i + 1, n - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if _is_straight(p[i : mid + 1], max_dev):
                lo = mid
            else:
                hi = mid - 1
        j = lo
        chord = p[j] - p[i]
        if (
            float(np.linalg.norm(chord)) >= min_len
            and _off_axis_deg(chord) <= max_angle_deg
            and _rms_residual(p[i : j + 1]) <= max_rms_px
        ):
            runs.append((i, j))
            i = j + 1
        else:
            i += 1
    return runs


def _split_out(p: np.ndarray, ranges: List[Tuple[int, int]]) -> List[np.ndarray]:
    """Remove the index ranges from `p`, returning the remaining pieces."""
    remove = np.zeros(len(p), dtype=bool)
    for i, j in ranges:
        remove[i : j + 1] = True
    pieces = []
    start = None
    for k in range(len(p) + 1):
        keep = k < len(p) and not remove[k]
        if keep and start is None:
            start = k
        elif not keep and start is not None:
            pieces.append(p[start:k])
            start = None
    return pieces


def remove_axis_lines(
    lines: List[np.ndarray],
    shape: Tuple[int, int],
    main_len_frac: float = 0.5,
    side_len_frac: float = 0.25,
    max_dev_frac: float = 0.012,
    max_angle_deg: float = 3.0,
    side_angle_deg: float = 5.0,
    max_rms_px: float = 0.6,
    join_px: float = 6.0,
) -> List[np.ndarray]:
    """Cut construction-axis remnants (baseline / vertical axis) out of paths.

    Where the axes touch the shard, the skeleton merges them with the contour,
    so an axis is often only one part of a path; such runs are removed and the
    path is split around them.

    The normalized shards are often oriented so that walls and rims are close
    to vertical/horizontal as well, so straightness and orientation alone are
    not enough (thresholds below were measured on samples, axes vs. walls):
    * every candidate must be straight (all points within `max_dev_frac` of the
      chord) and ruler-even (RMS residual <= `max_rms_px`; axes 0.35-0.54 px,
      freehand walls 0.56-0.97 px);
    * main axes: chord >= `main_len_frac` of the image (axes 0.51-0.90, most
      walls <= 0.40) and within `max_angle_deg` of HORIZONTAL. Only the
      baseline qualifies: the normalized crops keep the sheet orientation, so
      the baseline is always horizontal, while tall rims have long, straight,
      near-vertical walls (measured 0.50-0.57 long, RMS 0.60) that would
      otherwise be cut out;
    * secondary axes: the vertical axis (often short, e.g. starting
      mid-image, 0.38). A run (>= `side_len_frac`, within `side_angle_deg` of
      horizontal/vertical) is removed only if it is perpendicular to a main
      axis and one of its ends lies on it (within `join_px`), i.e. the two
      form the L/T of the construction cross;
    * leftovers: short straight pieces lying on the extension of a main axis
      (e.g. the bit of baseline on the far side of the shard) are dropped too.

    `lines` must be dense pixel paths (before simplification).
    """
    h, w = shape
    size = min(h, w)
    max_dev = max(2.0, max_dev_frac * size)
    cands = [
        _straight_runs(p, side_len_frac * size, max_dev, side_angle_deg, max_rms_px)
        for p in lines
    ]

    def chord(k: int, r: Tuple[int, int]) -> np.ndarray:
        return lines[k][r[1]] - lines[k][r[0]]

    main = [
        (k, r)
        for k, rs in enumerate(cands)
        for r in rs
        if float(np.linalg.norm(chord(k, r))) >= main_len_frac * size
        and _off_horizontal_deg(chord(k, r)) <= max_angle_deg
    ]
    removed = {k: [] for k in range(len(lines))}
    for k, r in main:
        removed[k].append(r)

    def on_main(pt: np.ndarray, direction: np.ndarray) -> bool:
        for k, r in main:
            a, b = lines[k][r[0]], lines[k][r[1]]
            d = b - a
            length = float(np.linalg.norm(d))
            u = d / length
            # Perpendicular to this main axis?
            if abs(float(np.dot(u, direction))) > math.sin(math.radians(side_angle_deg + max_angle_deg)):
                continue
            t = float(np.dot(pt - a, u))
            dist = abs(float(u[0] * (pt - a)[1] - u[1] * (pt - a)[0]))
            if -join_px <= t <= length + join_px and dist <= join_px:
                return True
        return False

    for k, rs in enumerate(cands):
        for r in rs:
            if r in removed[k]:
                continue
            c = chord(k, r)
            u = c / float(np.linalg.norm(c))
            if on_main(lines[k][r[0]], u) or on_main(lines[k][r[1]], u):
                removed[k].append(r)

    def on_main_line(q: np.ndarray) -> bool:
        if not _is_straight(q, max_dev):
            return False
        for k, r in main:
            a, b = lines[k][r[0]], lines[k][r[1]]
            u = (b - a) / float(np.linalg.norm(b - a))
            rel = q - a
            if float(np.abs(rel[:, 0] * u[1] - rel[:, 1] * u[0]).max()) <= max_dev:
                return True
        return False

    out = []
    for k, p in enumerate(lines):
        pieces = _split_out(p, removed[k]) if removed[k] else [p]
        out.extend(q for q in pieces if not (main and on_main_line(q)))
    return out


def trace_polylines(
    clean: np.ndarray,
    max_paths: int = 12,
    max_gap_frac: float = 0.05,
    min_keep_frac: float = 0.06,
    remove_axes: bool = False,
) -> List[np.ndarray]:
    """Skeletonize the binary drawing and turn it into smoothed, simplified
    polylines (longest first): branches are split into separate paths,
    small gaps (< `max_gap_frac` of the image) bridged, and paths shorter
    than `min_keep_frac` of the longest dropped. `remove_axes` cuts
    construction-axis remnants out (see `remove_axis_lines`)."""
    h, w = clean.shape
    min_branch_px = max(6, int(round(0.04 * min(h, w))))
    skel = skeletonize(clean > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(skel, 8)
    lines: List[np.ndarray] = []
    for i in range(1, n):
        if int(stats[i, cv2.CC_STAT_AREA]) < min_branch_px:
            continue
        component = (labels == i).astype(np.uint8)
        lines.extend(component_paths(component, min_branch_px))

    if not lines:
        raise RuntimeError("No pottery contour could be traced")

    lines = bridge_gaps(lines, max_gap_px=max_gap_frac * min(h, w))
    # Off by default: remove_axis_lines can also cut out straight shard walls.
    if remove_axes:
        lines = remove_axis_lines(lines, (h, w))
    lines = [p for p in lines if len(p) >= min_branch_px]
    if not lines:
        raise RuntimeError("No pottery contour could be traced")
    lines = [simplify_polyline(smooth_polyline(p)) for p in lines]
    lines = [p for p in lines if len(p) >= 2]

    lines.sort(key=polyline_length, reverse=True)
    longest = polyline_length(lines[0])
    # Shorter strokes are kept only if they are not negligible compared to the
    # main contour; this drops leftover hatching or annotation fragments.
    kept = [p for p in lines if polyline_length(p) >= min_keep_frac * longest]
    return kept[:max_paths]


def trace_image(gray: np.ndarray, **trace_kwargs) -> TraceResult:
    """Vectorize one grayscale drawing: `segment_drawing`, then
    `trace_polylines` (which takes `trace_kwargs`). Raises RuntimeError if
    no contour can be traced (e.g. a blank image)."""
    clean, threshold_value = segment_drawing(gray)
    polylines = trace_polylines(clean, **trace_kwargs)
    allpts = np.concatenate(polylines, axis=0)
    bx0 = int(np.floor(allpts[:, 0].min()))
    by0 = int(np.floor(allpts[:, 1].min()))
    bx1 = int(np.ceil(allpts[:, 0].max()))
    by1 = int(np.ceil(allpts[:, 1].max()))
    return TraceResult(
        polylines=polylines,
        drawing_bbox_px=(bx0, by0, bx1, by1),
        threshold_value=threshold_value,
        ink_fraction=float(np.count_nonzero(clean)) / clean.size,
    )


def fmt(v: float) -> str:
    return f"{v:.3f}".rstrip("0").rstrip(".")


def path_d(points: np.ndarray) -> str:
    if len(points) == 0:
        return ""
    parts = [f"M {fmt(points[0,0])} {fmt(points[0,1])}"]
    for x, y in points[1:]:
        parts.append(f"L {fmt(x)} {fmt(y)}")
    return " ".join(parts)


def svg_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def write_svg(
    result: TraceResult,
    image_shape: Tuple[int, int],
    input_name: str,
    out_svg: Path,
    stroke_px: float = 1.0,
    axes_segments: List[Tuple[float, float, float, float]] | None = None,
) -> None:
    """Write the polylines as an SVG in pixel coordinates (viewBox = image),
    with `axes_segments` (x1, y1, x2, y2) in a separate "axes" group."""
    h, w = image_shape
    source = svg_escape(input_name)
    svg = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        f'  <metadata>Source image: {source}. Pixel coordinates, no physical scale.</metadata>',
        f'  <g id="drawing" fill="none" stroke="#111" stroke-width="{fmt(stroke_px)}" stroke-linecap="round" stroke-linejoin="round">',
    ]
    for p in result.polylines:
        svg.append(f'    <path d="{path_d(p)}"/>')
    svg.append('  </g>')
    if axes_segments:
        svg.append(f'  <g id="axes" fill="none" stroke="#777" stroke-width="{fmt(stroke_px)}" stroke-linecap="round">')
        for x1, y1, x2, y2 in axes_segments:
            svg.append(f'    <path d="M {fmt(x1)} {fmt(y1)} L {fmt(x2)} {fmt(y2)}"/>')
        svg.append('  </g>')
    svg.append('</svg>')
    out_svg.write_text("\n".join(svg), encoding="utf-8")


def write_debug(gray: np.ndarray, result: TraceResult, output: Path, scale: int = 3) -> None:
    """Original image (left) and traced paths over it (right), upscaled."""
    big = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    vis = cv2.cvtColor(big, cv2.COLOR_GRAY2BGR)
    colors = [(0, 0, 255), (255, 0, 0), (0, 180, 0), (180, 0, 180), (0, 140, 255), (200, 200, 0)]
    for i, p in enumerate(result.polylines):
        pts = np.round((p + 0.5) * scale).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis, [pts], False, colors[i % len(colors)], 2, cv2.LINE_AA)
    left = cv2.cvtColor(big, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(str(output), np.hstack([left, vis]))


def render_polylines(
    polylines: List[np.ndarray],
    shape: Tuple[int, int],
    thickness: int = 1,
    antialias: bool = False,
) -> np.ndarray:
    """Draw the polylines in black on a white (H, W) uint8 canvas."""
    canvas = np.full(shape, 255, dtype=np.uint8)
    line_type = cv2.LINE_AA if antialias else cv2.LINE_8
    for p in polylines:
        pts = np.round(p).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [pts], False, 0, max(1, int(thickness)), line_type)
    return canvas
