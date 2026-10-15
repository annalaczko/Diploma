"""Drawing the coordinate system back onto preprocessed images: the shared
axes lookup (`AxisDrawing`, also used by `Vectorizer(axes="draw")`) and the
raster `AxisRenderer` step."""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from pipeline.preprocessing import axes as axes_lib
from pipeline.preprocessing.dataset import ImageRecord, collect_image_paths, flatten_name
from pipeline.preprocessing.preprocessing_step import PreprocessingStep


class AxisDrawing:
    """Mixin for steps that draw the axes recorded by `BackgroundEraser(record_axes=True)`.

    The axes rows come from `axes_source` - an earlier step with
    `provides_axes` (set by `Pipeline`, e.g. a BackgroundEraser with record_axes) - or, when
    the step runs on its own, from a `metadata_csv` file (extracted_metadata.csv). The drawing step then
    provides an updated table itself (its output may be a wider frame).
    """

    uses_axes = True

    def _init_axes(self, metadata_csv=None):
        self.metadata_csv = metadata_csv
        self.axes_source = None  # set by Pipeline
        self.axes_table: dict[str, dict] = {}  # this step's output rows
        self.no_axes: list[str] = []
        self._csv_rows = None

    def _input_row(self, name: str) -> dict | None:
        """The axes row for an input image, or None if there is none."""
        if self.axes_source is not None:
            return self.axes_source.axes_table.get(name)
        if self.metadata_csv is not None:
            if self._csv_rows is None:
                self._csv_rows = axes_lib.read_metadata_csv(self.metadata_csv)
            return self._csv_rows.get(name)
        raise RuntimeError(
            f"{self.name} draws the axes but has no axes data: run it in a Pipeline after "
            "BackgroundEraser(record_axes=True), or pass metadata_csv (the extracted_metadata.csv written "
            "next to the images)"
        )

    # A drawing step's output rows replace its input's (Pipeline exports the last).
    @property
    def provides_metadata(self) -> bool:
        return self.provides_axes

    def _axes_frame(self, row: dict, width: int, height: int, margin: float):
        """(input-image points of the row, extended output window)."""
        points = axes_lib.scaled_points(row, width, height)
        frame = axes_lib.extended_frame(width, height, axes_lib.segments_of(points), margin)
        return points, frame

    def _reset_axes(self):
        self.axes_table = {}
        self.no_axes = []


class AxisRenderer(AxisDrawing, PreprocessingStep):
    """Draws the coordinate system onto raster images (e.g. Repair's output):
    the image is extended with white so the axes fit (keeping its size, so
    the content shrinks), then the axes are drawn as straight lines of
    `axis_thickness` px. Images without axes are passed through unchanged
    (names in `no_axes`).

    Shrinking a raster thins its lines; `binarize` turns every pixel darker
    than `threshold` black afterwards, so thin lines stay visible. After a
    Vectorizer, prefer `Vectorizer(axes="draw")`, which redraws everything
    from vectors at full line width.
    """

    provides_axes = True

    def __init__(
        self,
        input_path: str | None = None,
        output_path: str | None = None,
        name: str | None = None,
        delete_input_data: bool = False,
        run_id: str = "default_run",
        axis_thickness: int = 2,
        margin: float = 0.03,
        binarize: bool = True,
        threshold: int = 235,
        antialias: bool = False,
        metadata_csv: str | None = None,
        image_extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"),
    ):
        """
        Args:
            axis_thickness: width (px) of the drawn axes.
            margin: white space kept around the axis ends, as a fraction
                of the image size.
            binarize: after shrinking, make pixels darker than `threshold`
                black and the rest white.
            threshold: see `binarize`.
            antialias: draw anti-aliased axis lines.
            metadata_csv: extracted_metadata.csv to read the axes from when not run in a
                Pipeline after BackgroundEraser(record_axes=True).
            image_extensions: file suffixes treated as input images.
        """
        super().__init__(input_path=input_path, output_path=output_path, name=name,
                         delete_input_data=delete_input_data, run_id=run_id)
        self.axis_thickness = axis_thickness
        self.margin = margin
        self.binarize = binarize
        self.threshold = threshold
        self.antialias = antialias
        self.image_extensions = image_extensions
        self._n_seen = 0
        self._init_axes(metadata_csv)

    def render(self, img: Image.Image, row: dict | None) -> tuple[Image.Image, dict | None]:
        """Extend `img` to fit the row's axes and draw them. Returns the new
        image and the row in its coordinates (unchanged without axes)."""
        gray = np.asarray(img.convert("L"))
        if not axes_lib.has_axes(row):
            return Image.fromarray(gray), row
        height, width = gray.shape
        points, (x0, y0, fw, fh) = self._axes_frame(row, width, height, self.margin)
        # Paste onto a whole-pixel white canvas, then shrink back to the size.
        left, top = int(round(-x0)), int(round(-y0))
        canvas_w = max(width + left, int(round(fw)))
        canvas_h = max(height + top, int(round(fh)))
        canvas = np.full((canvas_h, canvas_w), 255, np.uint8)
        canvas[top:top + height, left:left + width] = gray
        out = cv2.resize(canvas, (width, height), interpolation=cv2.INTER_AREA)
        if self.binarize:
            out = np.where(out < self.threshold, 0, 255).astype(np.uint8)
        # Input -> canvas (shift by the padding) -> output (shrink).
        out_points = (points + (left, top)) * (width / canvas_w, height / canvas_h)
        axes_lib.draw_segments(out, axes_lib.segments_of(out_points), self.axis_thickness, self.antialias)
        # The same in the row's frame (the input image is its uniform rescale).
        to_input = width / row["frame_w"]
        moved = axes_lib.moved_row(row, out_points, (width, height), factor=to_input * width / canvas_w,
                                   shift=(left / to_input, top / to_input))
        return Image.fromarray(out), moved

    def _render_one(self, img: Image.Image, name: str) -> Image.Image:
        self._n_seen += 1
        row = self._input_row(name)
        if not axes_lib.has_axes(row):
            self.no_axes.append(name)
        out, out_row = self.render(img, row)
        if out_row is not None:
            self.axes_table[name] = out_row
        return out

    def run(self) -> Path:
        """Draw the axes on every image under `input_path`; also writes
        <output folder>/extracted_metadata.csv in the output images' coordinates."""
        input_paths = collect_image_paths(self.input_path, self.image_extensions)
        if not input_paths:
            raise FileNotFoundError(
                f"No existing images with extensions {self.image_extensions} found under {self.input_path}"
            )
        output_dir = Path(self.output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.begin_run()
        for path in tqdm(input_paths, desc="Drawing axes", unit="image"):
            name = flatten_name(path, self.input_path)
            with Image.open(path) as img:
                self._render_one(img, name).save(output_dir / name)
        axes_lib.write_metadata_csv(self.axes_table, output_dir / axes_lib.METADATA_CSV)
        self.log_summary()
        if self.delete_input_data_flag:
            self.delete_input_data()
        return output_dir

    def process_batch(self, records: list[ImageRecord]) -> list[ImageRecord]:
        return [ImageRecord(r.name, self._render_one(r.image, r.name)) for r in records]

    def begin_run(self):
        self._n_seen = 0
        self._reset_axes()

    def failure_lists(self):
        return {"no_axes": self.no_axes}
