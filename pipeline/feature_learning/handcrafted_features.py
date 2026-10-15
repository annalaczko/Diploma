"""Hand-crafted shape features of the vectorized profile drawings.

Input: a Vectorizer output folder - `_svg/<name>.svg` (the traced lines) and
`extracted_metadata.csv` (per image the coordinate system and the scale, see
`pipeline/preprocessing/axes.py`). Every drawing is moved into the
coordinate system of its vessel, in centimetres:

    x = distance from the vertical (rotation) axis, positive on the side the
        profile is drawn on (drawings are mirrored to that side);
    y = distance from the horizontal (rim/base) line, positive downwards
        (in the image's direction), so a rim fragment hanging below its rim
        line has y > 0 and a base fragment standing on its base line y < 0.

Each `add_*` function computes one kind of feature for every profile and adds
its columns to the feature DataFrame (one row per image, indexed by the
image's file name); `extract_features` runs the chosen groups
(`FEATURE_GROUPS`). Images without a coordinate system get NaN.

`HandcraftedFeatures` is the pipeline step (`model.type: handcrafted` in the
configs): it writes the table and a `features.csv` for clustering, like the
learned models. On its own:

    python -m pipeline.feature_learning.handcrafted_features <Vectorizer folder> [--out features.csv]
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from pipeline import tracking
from pipeline.feature_learning.featurelearning_step import FeatureLearningStep
from pipeline.preprocessing.axes import METADATA_CSV, has_axes
from pipeline.preprocessing.dataset import collect_image_paths

# Vertical-axis remnants: ink this close to the axis (px) and this parallel
# to it (degrees) is the axis, not the profile. The horizontal line is kept:
# on the drawings the rim top / the base lies on it.
AXIS_TOL_PX = 2.5
AXIS_ANGLE_DEG = 15
MIN_PIECE_PX = 6  # shorter pieces left after removing the axis are dropped

ON_LINE_CM = 0.1          # ink this close to the horizontal line lies on it (rim top, base)
THICKNESS_MAX_CM = 3.0    # longest wall-to-wall distance searched for
THICKNESS_SKIP_PX = 3.0   # start the search this far from the line itself
EFD_HARMONICS = 10
TOUCH_PX = 3.0            # a line end this close to other ink is attached, not free
BREAK_MAX_CM = 3.0        # longest undrawn break line closed between two free ends


@dataclass
class Profile:
    """One drawing's lines, in pixels of its image and in the vessel's
    coordinate system (cm, see the module docstring)."""
    name: str
    pieces_px: list[np.ndarray]   # (N, 2) polylines, 1 px point spacing
    pieces_cm: list[np.ndarray]   # the same points as (x, y) in cm
    px_per_cm: float
    row: dict                     # its extracted_metadata.csv row
    side: float                   # +1 / -1: which side of the axis the profile is drawn on

    @property
    def points_cm(self) -> np.ndarray:
        return np.vstack(self.pieces_cm)

    @property
    def thickness_px(self) -> np.ndarray:
        """`wall_thickness_px` of the drawing (computed once)."""
        if not hasattr(self, "_thickness_px"):
            self._thickness_px = wall_thickness_px(self.pieces_px, THICKNESS_MAX_CM * self.px_per_cm)
        return self._thickness_px


# --------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------- #

def read_svg_paths(svg_path) -> list[np.ndarray]:
    """The polylines of the SVG's "drawing" group, as (N, 2) pixel arrays."""
    text = Path(svg_path).read_text(encoding="utf-8")
    drawing = text.split('id="drawing"', 1)[1].split("</g>", 1)[0]
    paths = []
    for d in re.findall(r'd="([^"]+)"', drawing):
        numbers = [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", d)]
        if len(numbers) >= 4:
            paths.append(np.array(numbers).reshape(-1, 2))
    return paths


def densify(points: np.ndarray, step: float = 1.0) -> np.ndarray:
    """Resample a polyline so consecutive points are at most `step` apart."""
    out = [points[0]]
    for a, b in zip(points[:-1], points[1:]):
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / step)))
        out.extend(a + (b - a) * t for t in np.arange(1, n + 1) / n)
    return np.array(out)


def _axis(row, key) -> tuple[np.ndarray, np.ndarray]:
    """(a point, unit direction) of the vertical ('v') or horizontal ('h') axis.
    The vertical one points down (y grows), the horizontal one right."""
    a = np.array([row[f"{key}_x1"], row[f"{key}_y1"]], dtype=float)
    b = np.array([row[f"{key}_x2"], row[f"{key}_y2"]], dtype=float)
    u = (b - a) / np.linalg.norm(b - a)
    if (key == "v" and u[1] < 0) or (key == "h" and u[0] < 0):
        u = -u
    return a, u


def remove_axis_remnants(paths: list[np.ndarray], row) -> list[np.ndarray]:
    """Cut out the stretches that run along the vertical axis; split the
    paths there and drop the short leftovers."""
    a, u = _axis(row, "v")
    pieces = []
    for path in paths:
        p = densify(path)
        if len(p) < 2:
            continue
        direction = np.diff(p, axis=0)
        direction = np.vstack([direction, direction[-1:]])
        direction /= np.linalg.norm(direction, axis=1, keepdims=True) + 1e-9
        rel = p - a
        close = np.abs(rel[:, 0] * u[1] - rel[:, 1] * u[0]) < AXIS_TOL_PX
        parallel = np.abs(direction @ u) > np.cos(np.radians(AXIS_ANGLE_DEG))
        remnant = np.convolve((close & parallel).astype(float), np.ones(5) / 5, "same") > 0.5
        edges = np.flatnonzero(np.diff(np.r_[False, ~remnant, False].astype(int)))
        for start, end in zip(edges[::2], edges[1::2]):
            piece = p[start:end]
            if len(piece) > 1 and np.linalg.norm(np.diff(piece, axis=0), axis=1).sum() >= MIN_PIECE_PX:
                pieces.append(piece)
    return pieces


def to_vessel_frame(points_px: np.ndarray, row, side: float) -> np.ndarray:
    """Pixel points -> (x, y) in cm: distance from the vertical axis (times
    `side`, so the profile is on the positive side) and from the horizontal
    line, positive downwards."""
    va, vu = _axis(row, "v")
    ha, hu = _axis(row, "h")
    rel = points_px - va
    x = (rel[:, 1] * vu[0] - rel[:, 0] * vu[1]) * side  # perpendicular to the axis
    normal = np.array([-hu[1], hu[0]])                  # points down
    y = (points_px - ha) @ normal
    return np.c_[x, y] / row["px_per_cm"]


def locate_sources(folder) -> tuple[Path, Path]:
    """(SVG folder, extracted_metadata.csv) of a Vectorizer output folder -
    in `folder` itself, or in its parent: `split_by_labels.py` moves the
    images into train/ and test/ subfolders and leaves these in place."""
    folder = Path(folder)
    for base in (folder, folder.parent):
        if (base / METADATA_CSV).is_file() and (base / "_svg").is_dir():
            return base / "_svg", base / METADATA_CSV
    raise FileNotFoundError(
        f"No _svg/ and {METADATA_CSV} in {folder} or its parent: hand-crafted features need the "
        "output folder of a Vectorizer with save_svg: true"
    )


def load_profiles(folder, names=None) -> dict[str, Profile | None]:
    """The images of a Vectorizer output folder (all in its metadata, or only
    `names`): each one's Profile, or None when it has no coordinate system,
    no SVG or no lines left."""
    svg_dir, metadata_csv = locate_sources(folder)
    metadata = pd.read_csv(metadata_csv)
    metadata["found"] = metadata["found"].astype(bool)
    rows = {row["filename"]: row for row in metadata.to_dict("records")}
    profiles = {}
    for name in tqdm(list(rows if names is None else names), desc="Handcrafted: loading SVGs", unit="image"):
        row = rows.get(name)
        if row is None:
            profiles[name] = None
            continue
        svg = svg_dir / f"{Path(name).stem}.svg"
        if not has_axes(row) or not svg.exists() or not row.get("px_per_cm") > 0:
            profiles[name] = None
            continue
        pieces = remove_axis_remnants(read_svg_paths(svg), row)
        if not pieces:
            profiles[name] = None
            continue
        # Which side of the axis the profile is drawn on: the majority of the ink.
        side = 1.0 if np.median(to_vessel_frame(np.vstack(pieces), row, 1.0)[:, 0]) >= 0 else -1.0
        profiles[name] = Profile(name, pieces, [to_vessel_frame(p, row, side) for p in pieces],
                                 float(row["px_per_cm"]), row, side)
    return profiles


# --------------------------------------------------------------------- #
# Features: each adds one kind of columns to the DataFrame
# --------------------------------------------------------------------- #

def _apply(df: pd.DataFrame, profiles: dict, compute, columns: list[str]) -> pd.DataFrame:
    """Add `columns`, filled per image by `compute(profile) -> dict` (NaN
    for images without a profile)."""
    # Progress bar named after the add_* function, e.g. "thickness".
    group = compute.__qualname__.split(".")[0].removeprefix("add_").removesuffix("_features")
    rows = {name: (compute(p) if p is not None else {})
            for name, p in tqdm(profiles.items(), desc=f"Handcrafted: {group}", unit="image")}
    added = pd.DataFrame.from_dict(rows, orient="index").reindex(index=df.index, columns=columns)
    return pd.concat([df, added], axis=1)


def add_position_features(df: pd.DataFrame, profiles: dict) -> pd.DataFrame:
    """Where the profile lies relative to the horizontal line: the share of
    its ink below / above the line, and 'below' (rim fragment), 'above'
    (base fragment) or 'both' as `position` (>= 90% on one side). Ink on
    the line itself (within ON_LINE_CM: the rim top, the base) is not
    counted."""
    def compute(p: Profile):
        y = p.points_cm[:, 1]
        y = y[np.abs(y) > ON_LINE_CM]
        if not len(y):
            return {"below_frac": np.nan, "above_frac": np.nan, "position": "on_line"}
        below = float(np.mean(y > 0))
        position = "below" if below >= 0.9 else "above" if below <= 0.1 else "both"
        return {"below_frac": below, "above_frac": 1.0 - below, "position": position}
    return _apply(df, profiles, compute, ["position", "below_frac", "above_frac"])


def add_size_features(df: pd.DataFrame, profiles: dict) -> pd.DataFrame:
    """Height (along the axis) and width (across it) of the profile, cm."""
    def compute(p: Profile):
        x, y = p.points_cm.T
        return {"rim_height_cm": float(np.ptp(y)), "rim_width_cm": float(np.ptp(x))}
    return _apply(df, profiles, compute, ["rim_height_cm", "rim_width_cm"])


def add_axis_distance_features(df: pd.DataFrame, profiles: dict) -> pd.DataFrame:
    """Distance of the profile from the vertical (rotation) axis, cm: mean
    (over the line length, the points are 1 px apart), min and max. The max
    is about the vessel's radius at its widest; min < 0 means the drawing
    crosses the axis."""
    def compute(p: Profile):
        x = p.points_cm[:, 0]
        return {"rim_axis_dist_mean_cm": float(x.mean()), "rim_axis_dist_min_cm": float(x.min()),
                "rim_axis_dist_max_cm": float(x.max())}
    return _apply(df, profiles, compute,
                  ["rim_axis_dist_mean_cm", "rim_axis_dist_min_cm", "rim_axis_dist_max_cm"])


def wall_thickness_px(pieces: list[np.ndarray], max_px: float, step: int = 2) -> np.ndarray:
    """Distances (px) from points of the lines to the nearest other line
    along the line's normal, in either direction - the wall thickness where
    the two sides of the wall are drawn. The lines need not be joined or
    ordered: from every `step`-th point, rays are cast both ways across the
    rasterized drawing and the first ink they hit (past THICKNESS_SKIP_PX,
    i.e. off the line itself) counts. Points whose rays hit nothing within
    `max_px` (the outer side of a single line) give no value."""
    scale = 2.0  # rasterize at twice the resolution for sub-pixel distances
    all_points = np.vstack(pieces)
    origin = all_points.min(axis=0) - max_px - 2
    size = np.ceil((all_points.max(axis=0) - origin + max_px + 2) * scale).astype(int)
    canvas = np.zeros((size[1], size[0]), np.uint8)
    for p in pieces:
        cv2.polylines(canvas, [np.round((p - origin) * scale).astype(np.int32)], False, 1, 1)

    starts, normals = [], []
    for p in pieces:
        if len(p) < 3:
            continue
        tangent = np.gradient(p, axis=0)
        tangent /= np.linalg.norm(tangent, axis=1, keepdims=True) + 1e-9
        starts.append(p[::step])
        normals.append(np.c_[-tangent[::step, 1], tangent[::step, 0]])
    if not starts:
        return np.array([])
    starts, normals = np.vstack(starts), np.vstack(normals)

    distances = np.arange(THICKNESS_SKIP_PX, max_px, 0.5)
    best = np.full(len(starts), np.inf)
    for sign in (1.0, -1.0):
        ray = starts[:, None, :] + sign * normals[:, None, :] * distances[None, :, None]
        ix = np.round((ray - origin) * scale).astype(int)
        inside = (ix[..., 0] >= 0) & (ix[..., 0] < size[0]) & (ix[..., 1] >= 0) & (ix[..., 1] < size[1])
        hit = np.zeros(inside.shape, bool)
        hit[inside] = canvas[ix[..., 1][inside], ix[..., 0][inside]] > 0
        first = np.where(hit.any(axis=1), distances[hit.argmax(axis=1)], np.inf)
        best = np.minimum(best, first)
    return best[np.isfinite(best)]


def add_thickness_features(df: pd.DataFrame, profiles: dict) -> pd.DataFrame:
    """Mean wall thickness, cm (see `wall_thickness_px`); NaN when the two
    sides of the wall are nowhere drawn within THICKNESS_MAX_CM."""
    def compute(p: Profile):
        thickness = p.thickness_px
        return {"rim_thickness_mean_cm": float(thickness.mean() / p.px_per_cm) if len(thickness) else np.nan}
    return _apply(df, profiles, compute, ["rim_thickness_mean_cm"])


def free_ends(pieces: list[np.ndarray]) -> np.ndarray:
    """The line ends that touch no other ink (within TOUCH_PX): where the
    drawing is open. An end touching its own line further along (a loop)
    counts as attached too."""
    ends = []
    for i, piece in enumerate(pieces):
        arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(piece, axis=0), axis=1))]
        for end, along in ((piece[0], arc), (piece[-1], arc[-1] - arc)):
            others = [q for j, q in enumerate(pieces) if j != i] + [piece[along > 4 * TOUCH_PX]]
            others = np.vstack([q for q in others if len(q)]) if any(len(q) for q in others) else np.empty((0, 2))
            if not len(others) or np.linalg.norm(others - end, axis=1).min() > TOUCH_PX:
                ends.append(end)
    return np.array(ends).reshape(-1, 2)


def break_lines(ends: np.ndarray, max_px: float) -> list[tuple[np.ndarray, np.ndarray]]:
    """Pairs of free ends to join with a straight line - the break edges the
    drawings leave out: the closest pairs first, each end used once, up to
    `max_px` apart."""
    pairs = sorted((np.linalg.norm(ends[i] - ends[j]), i, j)
                   for i in range(len(ends)) for j in range(i + 1, len(ends)))
    used, lines = set(), []
    for distance, i, j in pairs:
        if distance > max_px:
            break
        if i not in used and j not in used:
            used |= {i, j}
            lines.append((ends[i], ends[j]))
    return lines


def outline_contour(p: Profile, n_points: int = 256) -> np.ndarray | None:
    """A closed outline of the cross-section in the vessel frame (cm),
    resampled to `n_points` points evenly spaced along it.

    The drawings leave the break edges out, so the section is open there:
    its free line ends are joined pairwise (`free_ends`, `break_lines`,
    closest first, up to BREAK_MAX_CM) and the enclosed area is filled.
    When that encloses nothing (e.g. a single line), the ink is closed
    morphologically instead, wider than the wall (1.5 x its median
    thickness, at least 5% of the drawing)."""
    scale = 4.0  # canvas px per image px
    pts = np.vstack(p.pieces_px)
    thickness = p.thickness_px
    wall = 1.5 * float(np.median(thickness)) if len(thickness) else 0.0
    gap = max(wall, 0.05 * max(np.ptp(pts, axis=0)))  # image px
    pad = gap + 8  # room for the closing at the canvas edge
    origin = pts.min(axis=0) - pad
    size = np.ceil((pts.max(axis=0) - origin + pad) * scale).astype(int)

    def to_canvas(points):
        return np.round((np.asarray(points) - origin) * scale).astype(np.int32)

    ink = np.zeros((size[1], size[0]), np.uint8)
    for piece in p.pieces_px:
        cv2.polylines(ink, [to_canvas(piece)], False, 255, int(scale))

    # 1) Close the section along its undrawn break edges and fill it.
    closed = ink.copy()
    for a, b in break_lines(free_ends(p.pieces_px), BREAK_MAX_CM * p.px_per_cm):
        cv2.line(closed, tuple(to_canvas(a)), tuple(to_canvas(b)), 255, int(scale))
    outside = closed.copy()
    flood = np.zeros((size[1] + 2, size[0] + 2), np.uint8)
    cv2.floodFill(outside, flood, (0, 0), 128)
    filled = np.where(outside == 0, 255, closed).astype(np.uint8)  # not reached from outside = enclosed
    enclosed = int(np.count_nonzero(filled)) > 1.5 * int(np.count_nonzero(closed))

    # 2) Otherwise: closing with a disk wider than the wall, via distance
    #    transforms (the cost doesn't grow with the radius).
    if not enclosed:
        filled = ink
        radius = gap * scale / 2
        if radius > 0.5:
            dilated = cv2.distanceTransform((ink == 0).astype(np.uint8), cv2.DIST_L2, 5) <= radius
            filled = (cv2.distanceTransform(dilated.astype(np.uint8), cv2.DIST_L2, 5) > radius).astype(np.uint8) * 255

    contours, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)[:, 0, :].astype(float) / scale + origin
    ring = np.vstack([contour, contour[:1]])
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(ring, axis=0), axis=1))]
    if arc[-1] <= 0:
        return None
    t = np.linspace(0, arc[-1], n_points, endpoint=False)
    resampled = np.c_[np.interp(t, arc, ring[:, 0]), np.interp(t, arc, ring[:, 1])]
    return to_vessel_frame(resampled, p.row, p.side)


def elliptic_fourier_descriptors(contour: np.ndarray, harmonics: int) -> np.ndarray:
    """Elliptic Fourier descriptors (Kuhl & Giardina 1982) of a closed
    contour, (harmonics, 4) as (a, b, c, d) per harmonic. Normalized for the
    starting point and the size (the first harmonic's ellipse gets semi-major
    axis 1), but not for rotation: the vessel frame already fixes the
    orientation, and it carries meaning (which way the rim points)."""
    d = np.diff(np.vstack([contour, contour[:1]]), axis=0)
    dt = np.linalg.norm(d, axis=1)
    keep = dt > 0
    d, dt = d[keep], dt[keep]
    t = np.r_[0, np.cumsum(dt)]
    T = t[-1]
    n = np.arange(1, harmonics + 1)[:, None]
    phi = 2 * np.pi * t[None, :] / T
    dcos = np.cos(n * phi[:, 1:]) - np.cos(n * phi[:, :-1])
    dsin = np.sin(n * phi[:, 1:]) - np.sin(n * phi[:, :-1])
    k = T / (2 * n ** 2 * np.pi ** 2)
    a = k[:, 0] * (d[:, 0] / dt * dcos).sum(axis=1)
    b = k[:, 0] * (d[:, 0] / dt * dsin).sum(axis=1)
    c = k[:, 0] * (d[:, 1] / dt * dcos).sum(axis=1)
    e = k[:, 0] * (d[:, 1] / dt * dsin).sum(axis=1)
    coeffs = np.c_[a, b, c, e]

    # Starting point: rotate every harmonic's phase so the first one starts
    # at its semi-major axis.
    a1, b1, c1, d1 = coeffs[0]
    theta = 0.5 * np.arctan2(2 * (a1 * b1 + c1 * d1), a1 ** 2 + c1 ** 2 - b1 ** 2 - d1 ** 2)
    for i in range(harmonics):
        ct, st = np.cos((i + 1) * theta), np.sin((i + 1) * theta)
        ai, bi, ci, di = coeffs[i]
        coeffs[i] = [ai * ct + bi * st, -ai * st + bi * ct, ci * ct + di * st, -ci * st + di * ct]
    size = np.hypot(coeffs[0, 0], coeffs[0, 2])
    return coeffs / size if size > 0 else coeffs


def add_fourier_features(df: pd.DataFrame, profiles: dict, harmonics: int = EFD_HARMONICS) -> pd.DataFrame:
    """Elliptic Fourier descriptors of the drawing's outline (see
    `outline_contour`, `elliptic_fourier_descriptors`): efd_<n>_<a|b|c|d>
    for n = 1..harmonics. Shape only - the size is in the other features."""
    columns = [f"efd_{i}_{part}" for i in range(1, harmonics + 1) for part in "abcd"]

    def compute(p: Profile):
        contour = outline_contour(p)
        if contour is None or len(contour) < 2 * harmonics:
            return {}
        return dict(zip(columns, elliptic_fourier_descriptors(contour, harmonics).ravel()))
    return _apply(df, profiles, compute, columns)


# Feature groups by name (the `features` setting of HandcraftedFeatures).
FEATURE_GROUPS = {
    "position": add_position_features,
    "size": add_size_features,
    "axis_distance": add_axis_distance_features,
    "thickness": add_thickness_features,
    "fourier": add_fourier_features,
}


def extract_features(folder, groups=tuple(FEATURE_GROUPS), names=None,
                     efd_harmonics: int = EFD_HARMONICS) -> pd.DataFrame:
    """The feature table of a Vectorizer output folder: one row per image
    (index: file name; all images in its metadata, or only `names`), the
    columns of every group in `groups` (see FEATURE_GROUPS)."""
    unknown = set(groups) - set(FEATURE_GROUPS)
    if unknown:
        raise ValueError(f"Unknown feature groups {sorted(unknown)}; choose from {list(FEATURE_GROUPS)}")
    profiles = load_profiles(folder, names)
    df = pd.DataFrame(index=pd.Index(list(profiles), name="filename"))
    for group in groups:
        if group == "fourier":
            df = add_fourier_features(df, profiles, harmonics=efd_harmonics)
        else:
            df = FEATURE_GROUPS[group](df, profiles)
    return df


def feature_matrix(table: pd.DataFrame) -> pd.DataFrame:
    """The numeric columns used as the feature vector: the text `position`
    and `above_frac` (= 1 - below_frac) are left out; missing values are
    filled with the column's median (0 where a column has none)."""
    numeric = table.drop(columns=["position", "above_frac"], errors="ignore").astype(float)
    return numeric.fillna(numeric.median()).fillna(0.0)


# --------------------------------------------------------------------- #
# Pipeline step
# --------------------------------------------------------------------- #

class HandcraftedFeatures(FeatureLearningStep):
    """Feature step that measures the profile drawings instead of learning
    features (`model.type: handcrafted`). It needs the files of a Vectorizer
    run with `save_svg: true`: `input_path` is that folder (or a train/test
    subfolder of it, see `locate_sources`); in memory mode `Pipeline` points
    it at the saved preprocessed images (`reads_saved_images`).

    `run()` writes to `output_path`:
      handcrafted_features.csv   the readable table (NaN where not measurable)
      features.csv               filename + feature vector, for clustering
      feature_columns.txt        the vector's columns, in order
    """

    # Pipeline (memory mode) sets input_path to the saved preprocessed folder.
    reads_saved_images = True

    def __init__(
        self,
        input_path: str | None = None,
        features=tuple(FEATURE_GROUPS),
        efd_harmonics: int = EFD_HARMONICS,
        output_path=None,
        name=None,
        run_id="default_run",
    ):
        """
        Args:
            input_path: Vectorizer output folder (or a subfolder of it with
                the images to use).
            features: feature groups to compute, from FEATURE_GROUPS:
                position, size, axis_distance, thickness, fourier.
            efd_harmonics: number of elliptic Fourier harmonics (4 values each).
        """
        super().__init__(input_path=input_path or "", output_path=output_path, name=name, run_id=run_id)
        self.features = tuple(features)
        unknown = set(self.features) - set(FEATURE_GROUPS)
        if unknown:
            raise ValueError(f"Unknown feature groups {sorted(unknown)}; choose from {list(FEATURE_GROUPS)}")
        self.efd_harmonics = efd_harmonics
        self.hparams = {"features": ",".join(self.features), "efd_harmonics": efd_harmonics}

    def run(self, mode="extract_features", **_ignored) -> pd.DataFrame:
        """Compute the features of every image in `input_path`. `mode` and
        the training arguments (epochs, lr, ...) are accepted so the step
        can stand in for a learned model; nothing is trained."""
        if not self.input_path:
            raise ValueError(f"{self.name} needs input_path: a Vectorizer output folder with _svg/")
        names = [path.name for path in collect_image_paths(self.input_path)]
        tracking.log_params({"feature_step": self.name})
        tracking.log_params(self.hparams, prefix=f"feature/{self.name}/")

        table = extract_features(self.input_path, self.features, names, self.efd_harmonics)
        matrix = feature_matrix(table)
        output = Path(self.output_path)
        output.mkdir(parents=True, exist_ok=True)
        table.to_csv(output / "handcrafted_features.csv")
        (output / "feature_columns.txt").write_text("\n".join(matrix.columns) + "\n", encoding="utf-8")
        result = pd.DataFrame({
            "filename": matrix.index,
            "features": [" ".join(f"{v:.6g}" for v in row) for row in matrix.to_numpy()],
        })
        result.to_csv(output / "features.csv", index=False)

        measured = int(table.drop(columns=["position"], errors="ignore").notna().any(axis=1).sum())
        tracking.log_metrics({"n_images": len(table), "n_measured": measured,
                              "n_features": matrix.shape[1]}, prefix="feature/")
        tracking.log_artifact(output / "handcrafted_features.csv", artifact_path="feature")
        self._log_features(output / "features.csv")
        print(f"{self.name}: {measured}/{len(table)} images measured, {matrix.shape[1]} features "
              f"-> {output / 'features.csv'}")
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("folder", help="Vectorizer output folder (with _svg/ and extracted_metadata.csv)")
    parser.add_argument("--out", help="CSV to write (default: <folder>/handcrafted_features.csv)")
    args = parser.parse_args(argv)
    df = extract_features(args.folder)
    out = Path(args.out) if args.out else Path(args.folder) / "handcrafted_features.csv"
    df.to_csv(out)
    n_ok = int(df["rim_height_cm"].notna().sum())
    print(f"{len(df)} images, {n_ok} with features -> {out}")


if __name__ == "__main__":
    main()
