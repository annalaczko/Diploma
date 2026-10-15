"""Vectorizer step: traces each cropped shard drawing into polylines and
redraws it as a clean line image (see `vectorization.py` for the tracing)."""

from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from pipeline.preprocessing import axes as axes_lib
from pipeline.preprocessing import vectorization
from pipeline.preprocessing.axis_renderer import AxisDrawing
from pipeline.preprocessing.dataset import ImageRecord, collect_image_paths, flatten_name
from pipeline.preprocessing.preprocessing_step import PreprocessingStep


class Vectorizer(AxisDrawing, PreprocessingStep):
    """Vectorizes cropped, normalized shard drawings (e.g. after Normalizer
    or Repair) and outputs them redrawn from the traced polylines: the same
    size, black lines of `line_thickness` px on white, without paper noise,
    specks, small gaps or skeleton spurs. Optionally also writes each
    drawing as an SVG.

    With `axes="draw"` it also draws the coordinate system recorded by
    `BackgroundEraser(record_axes=True)`: the frame is extended so the axes fit (output size
    unchanged, so the drawing shrinks), and drawing, axes and everything are
    redrawn from vectors at full line width. Images without axes are drawn
    without them (names in `no_axes`).

    Images on which nothing can be traced (blank or noise-only) are handled
    by `on_failure` and their names collected in `no_trace` (logged to
    MLflow as `no_trace.txt`, see `PreprocessingStep.log_summary`).
    """

    def __init__(
        self,
        input_path: str | None = None,
        output_path: str | None = None,
        name: str | None = None,
        delete_input_data: bool = False,
        run_id: str = "default_run",
        line_thickness: int = 2,
        antialias: bool = False,
        max_paths: int = 12,
        max_gap_frac: float = 0.05,
        min_keep_frac: float = 0.06,
        remove_axes: bool = False,
        on_failure: str = "keep",
        axes: str = "none",
        axis_thickness: int | None = None,
        axes_margin: float = 0.03,
        metadata_csv: str | None = None,
        save_svg: bool = False,
        debug_images: int = 10,
        image_extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"),
    ):
        """
        Args:
            line_thickness: width (px) of the redrawn lines.
            antialias: draw anti-aliased (gray-edged) lines instead of
                crisp black-and-white ones.
            max_paths: keep at most this many polylines (longest first).
            max_gap_frac: bridge gaps between line ends up to this fraction
                of the image size, when the gap continues both strokes.
            min_keep_frac: drop polylines shorter than this fraction of the
                longest one (hatching, annotation fragments).
            remove_axes: cut straight construction-axis remnants out of the
                drawing. Off by default: it can also cut straight shard
                walls.
            on_failure: what to do with an image nothing can be traced on:
                'keep' (default) outputs it unchanged, 'drop' leaves it out.
            axes: 'none' (default) ignores the coordinate system; 'draw'
                extends the frame and draws the axes (see the class docstring).
            axis_thickness: width (px) of the drawn axes; None = line_thickness.
            axes_margin: with axes='draw', white space kept around the axis
                ends, as a fraction of the image size.
            metadata_csv: with axes='draw', the extracted_metadata.csv to read the axes from
                when not run in a Pipeline after a BackgroundEraser with record_axes.
            save_svg: also write <name>.svg into `<output folder>/_svg`
                (in both modes).
            debug_images: how many of the first images also get a debug
                overlay (input | traced paths over it, upscaled) in a
                `<output folder>/_debug` directory. 0 disables it.
            image_extensions: file suffixes treated as input images.
        """
        super().__init__(
            input_path=input_path,
            output_path=output_path,
            name=name,
            delete_input_data=delete_input_data,
            run_id=run_id,
        )
        if on_failure not in ("keep", "drop"):
            raise ValueError("on_failure must be 'keep' or 'drop'")
        if axes not in ("none", "draw"):
            raise ValueError("axes must be 'none' or 'draw'")
        self.line_thickness = line_thickness
        self.antialias = antialias
        self.max_paths = max_paths
        self.max_gap_frac = max_gap_frac
        self.min_keep_frac = min_keep_frac
        self.remove_axes = remove_axes
        self.on_failure = on_failure
        self.axes = axes
        self.axis_thickness = axis_thickness
        self.axes_margin = axes_margin
        self.save_svg = save_svg
        self.debug_images = debug_images
        self.image_extensions = image_extensions
        self.no_trace: list[str] = []
        self._n_seen = 0
        self._debug_saved = 0  # debug overlays saved so far (counts across batches)
        self._init_axes(metadata_csv)

    # Only a drawing Vectorizer needs and produces axes rows (see Pipeline).
    @property
    def uses_axes(self) -> bool:
        return self.axes == "draw"

    @property
    def provides_axes(self) -> bool:
        return self.axes == "draw"

    # ------------------------------------------------------------------ #
    # Per-image logic, shared by both modes
    # ------------------------------------------------------------------ #

    def trace(self, img: Image.Image) -> vectorization.TraceResult:
        """Trace one image into polylines (raises RuntimeError if nothing
        can be traced)."""
        gray = np.asarray(img.convert("L"))
        return vectorization.trace_image(
            gray,
            max_paths=self.max_paths,
            max_gap_frac=self.max_gap_frac,
            min_keep_frac=self.min_keep_frac,
            remove_axes=self.remove_axes,
        )

    def process_image(self, img: Image.Image) -> Image.Image:
        """Trace one image and redraw it from the polylines (grayscale).
        Raises RuntimeError if nothing can be traced."""
        result = self.trace(img)
        return Image.fromarray(self._render(result, img.size))

    def _render(self, result: vectorization.TraceResult, size, axis_segments=None) -> np.ndarray:
        width, height = size
        canvas = vectorization.render_polylines(
            result.polylines, (height, width), thickness=self.line_thickness, antialias=self.antialias
        )
        if axis_segments:
            thickness = self.axis_thickness if self.axis_thickness is not None else self.line_thickness
            axes_lib.draw_segments(canvas, axis_segments, thickness, self.antialias)
        return canvas

    def _with_axes(self, result: vectorization.TraceResult, size, name: str):
        """With axes='draw': the traced result moved into the frame extended
        for the image's axes, the axes in that frame, and the updated row."""
        if self.axes != "draw":
            return result, None
        row = self._input_row(name)
        if not axes_lib.has_axes(row):
            self.no_axes.append(name)
            if row is not None:
                self.axes_table[name] = row  # same frame, nothing to move
            return result, None
        width, height = size
        points, frame = self._axes_frame(row, width, height, self.axes_margin)
        out_points = axes_lib.to_output(points, frame, width)
        polylines = [axes_lib.to_output(p, frame, width).astype(np.float32) for p in result.polylines]
        # The same move in the row's frame (the input image is its uniform rescale).
        to_input = width / row["frame_w"]
        self.axes_table[name] = axes_lib.moved_row(row, out_points, size, factor=to_input * width / frame[2],
                                                   shift=(-frame[0] / to_input, -frame[1] / to_input))
        return replace(result, polylines=polylines), axes_lib.segments_of(out_points)

    def _vectorize_one(self, img: Image.Image, name: str) -> Image.Image | None:
        """`process_image` plus the side outputs (SVG, debug overlay) and
        the `on_failure` handling. Returns None for a dropped image."""
        self._n_seen += 1
        try:
            result = self.trace(img)
        except RuntimeError:
            self.no_trace.append(name)
            if self.axes == "draw" and self.on_failure == "keep":
                row = self._input_row(name)
                if row is not None:
                    self.axes_table[name] = row  # image passed through unchanged
            return img.convert("L") if self.on_failure == "keep" else None

        drawn, axis_segments = self._with_axes(result, img.size, name)
        if self.save_svg:
            svg_dir = self.side_output_dir("svg")
            svg_dir.mkdir(parents=True, exist_ok=True)
            vectorization.write_svg(drawn, (img.size[1], img.size[0]), name,
                                    svg_dir / f"{Path(name).stem}.svg", axes_segments=axis_segments)
        if self._debug_saved < self.debug_images:
            debug_dir = self.side_output_dir("debug")
            debug_dir.mkdir(parents=True, exist_ok=True)
            gray = np.asarray(img.convert("L"))
            vectorization.write_debug(gray, result, debug_dir / f"{Path(name).stem}_debug.png")
            self._debug_saved += 1
        return Image.fromarray(self._render(drawn, img.size, axis_segments))

    # ------------------------------------------------------------------ #
    # Disk and in-memory modes
    # ------------------------------------------------------------------ #

    def run(self) -> Path:
        """Vectorize every image under `input_path` and write the redrawn
        images to `output_path` under the same (flattened) names."""
        input_paths = collect_image_paths(self.input_path, self.image_extensions)
        if not input_paths:
            raise FileNotFoundError(
                f"No existing images with extensions {self.image_extensions} found under {self.input_path}"
            )
        output_dir = Path(self.output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.begin_run()
        for path in tqdm(input_paths, desc="Vectorizing images", unit="image"):
            name = flatten_name(path, self.input_path)
            with Image.open(path) as img:
                out = self._vectorize_one(img, name)
            if out is not None:
                out.save(output_dir / name)
        if self.axes == "draw":
            axes_lib.write_metadata_csv(self.axes_table, output_dir / axes_lib.METADATA_CSV)
        self.log_summary()

        if self.delete_input_data_flag:
            self.delete_input_data()
        return output_dir

    def process_batch(self, records: list[ImageRecord]) -> list[ImageRecord]:
        """In-memory mode: vectorize each image, keeping its name (images
        dropped by `on_failure='drop'` are left out)."""
        output = []
        for record in records:
            out = self._vectorize_one(record.image, record.name)
            if out is not None:
                output.append(ImageRecord(record.name, out))
        return output

    # ------------------------------------------------------------------ #
    # Run bookkeeping (see PreprocessingStep)
    # ------------------------------------------------------------------ #

    def begin_run(self):
        self.no_trace = []
        self._n_seen = 0
        self._debug_saved = 0
        self._reset_axes()

    def failure_lists(self):
        if self.axes == "draw":
            return {"no_trace": self.no_trace, "no_axes": self.no_axes}
        return {"no_trace": self.no_trace}
