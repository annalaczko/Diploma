"""Repair step: turns a cleaned scan into a crisp black-and-white line
drawing.

The stages, in order (each is its own method, so it can be tried alone):

    1. enhance_contrast      push gray pixels toward black or white
    2. threshold_and_dilate  binarize (Otsu) and slightly thicken the lines
    3. to_skeleton           thin every line to 1 px          } only with
    4. repair_skeleton       join line ends that almost touch } skeletonize=True
    5. thicken               draw the result with `output_thickness` px lines

Inside the stages lines are white (255) on black (0), as OpenCV's
morphology expects; the final image is inverted back to black on white.
"""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageEnhance
from scipy.ndimage import convolve
from scipy.spatial.distance import cdist
from skimage.filters import threshold_otsu
from skimage.morphology import skeletonize
from tqdm import tqdm

from pipeline.preprocessing.preprocessing_step import PreprocessingStep
from pipeline.preprocessing.dataset import ImageRecord, collect_image_paths


class Repair(PreprocessingStep):
    """Cleans up scanned line drawings and repairs small gaps in the lines.

    Settings (all optional):
        contrast_factor: factor passed to ImageEnhance.Contrast; higher
            values push mid-tones toward pure black/white.
        dilate_kernel_size: side length of the square structuring
            element used before skeletonization, to pre-close small
            gaps (e.g. 3 for a (3,3) kernel).
        dilate_iterations: number of dilation iterations applied with
            that kernel.
        max_distance: maximum pixel distance between two skeleton
            endpoints for them to be joined during the repair step.
        line_thickness: thickness (in px) of the connecting line drawn
            between matched endpoints.
        min_component_size: connected components smaller than this
            (in px) are treated as noise and removed before repairing.
            Set to 0 to disable this noise-filtering step.
        skeletonize: if True (default), thin lines to a 1px skeleton
            and run the endpoint gap-repair step. If False, both are
            skipped entirely - the pipeline only binarizes (threshold
            + dilate) and, if `output_thickness > 1`, thickens the
            result, without any skeleton-based contour repair.
        re_skeletonize: if True, re-thin the result after connecting
            gaps, so the output stays a clean 1px skeleton instead of
            having slightly thicker joins. Only relevant when
            `skeletonize` is True.
        output_thickness: final line thickness (in px) of the output
            image. 1 keeps the raw 1px skeleton; any larger odd value
            (3, 5, ...) dilates the repaired skeleton uniformly to
            that width as the very last step.
        image_extensions: file suffixes considered as input images
            when input_path points to a directory.
        debug_images: how many of the first images also get a debug
            strip (every intermediate stage side by side, see
            `_save_debug_image`), written to a sibling
            `<output folder>/_debug` directory. 0 disables it.
    """

    def __init__(
        self,
        input_path: str | None = None,
        output_path: str | None = None,
        name: str | None = None,
        delete_input_data: bool = False,
        run_id: str = "default_run",
        contrast_factor: float = 10.0,
        dilate_kernel_size: int = 3,
        dilate_iterations: int = 3,
        max_distance: int = 15,
        line_thickness: int = 1,
        min_component_size: int = 0,
        skeletonize: bool = True,
        re_skeletonize: bool = True,
        output_thickness: int = 1,
        debug_images: int = 10,
        image_extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"),
    ):
        super().__init__(
            input_path=input_path,
            output_path=output_path,
            name=name,
            delete_input_data=delete_input_data,
            run_id=run_id,
        )
        self.contrast_factor = contrast_factor
        self.dilate_kernel_size = dilate_kernel_size
        self.dilate_iterations = dilate_iterations
        self.max_distance = max_distance
        self.line_thickness = line_thickness
        self.min_component_size = min_component_size
        self.skeletonize = skeletonize
        self.re_skeletonize = re_skeletonize
        self.output_thickness = output_thickness
        self.image_extensions = image_extensions
        self.debug_images = debug_images
        self._debug_saved = 0  # debug strips saved so far (counts across batches)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def run(self) -> Path:
        """Disk mode: repair every image under `input_path` and save it
        under the same name in `output_path`, which is returned."""
        input_paths = collect_image_paths(self.input_path, self.image_extensions)
        output_dir = Path(self.output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        if not input_paths:
            source_description = self.input_path if isinstance(self.input_path, (list, tuple)) else Path(self.input_path)
            raise FileNotFoundError(
                f"No existing images with extensions {self.image_extensions} found under {source_description}"
            )

        debug_dir = self.side_output_dir("debug")

        self._debug_saved = 0
        for image_path in tqdm(input_paths, desc="Repairing images", unit="image"):
            repaired = self._repair_one(Image.open(image_path), image_path.name, debug_dir)
            Image.fromarray(repaired).save(output_dir / image_path.name)

        if self.delete_input_data_flag:
            self.delete_input_data()

        return output_dir

    def process_batch(self, records: list[ImageRecord]) -> list[ImageRecord]:
        """In-memory mode: repair each image, keeping its name. Debug strips
        (if `debug_images > 0`) still go to `<output folder>/_debug`, for
        the first `debug_images` images across all batches."""
        debug_dir = self.side_output_dir("debug")
        return [
            ImageRecord(r.name, Image.fromarray(self._repair_one(r.image, r.name, debug_dir)))
            for r in records
        ]

    def _repair_one(self, img: Image.Image, name: str, debug_dir: Path) -> np.ndarray:
        """`process_image` on one image, also saving its debug strip while
        fewer than `debug_images` have been saved so far."""
        img = img.convert("L")
        if self._debug_saved < self.debug_images:
            repaired, stages = self.process_image(img, return_stages=True)
            self._save_debug_image(stages, debug_dir / name)
            self._debug_saved += 1
            return repaired
        return self.process_image(img)

    def process_image(self, img: Image.Image, return_stages: bool = False):
        """All stages on one grayscale PIL image (see the module docstring).
        Both modes (disk and in-memory) use this method.

        Returns the result as a uint8 array (black lines, 0, on white, 255).
        With `return_stages=True` it returns `(image, stages)`, where
        `stages` is an ordered {label: image} dict of every intermediate
        result, for the debug strip (`_save_debug_image`).
        """
        stages = {"input": np.array(img)}
        enhanced = self.enhance_contrast(img)
        stages["contrast"] = np.array(enhanced)
        dilated = self.threshold_and_dilate(enhanced)
        stages["threshold+dilate"] = 255 - dilated

        if self.skeletonize:
            skeleton = self.to_skeleton(dilated)
            stages["skeleton"] = 255 - skeleton
            processed = self.repair_skeleton(skeleton)
            stages["repair (red=added, blue=removed)"] = self._diff_overlay(skeleton, processed)
        else:
            processed = dilated
            if self.min_component_size > 0:
                processed = self._remove_small_components(processed, self.min_component_size)
                stages["small comps (blue=removed)"] = self._diff_overlay(dilated, processed)

        if self.output_thickness > 1:
            processed = self.thicken(processed, self.output_thickness)

        result = 255 - processed
        stages["output"] = result
        if return_stages:
            return result, stages
        return result

    # ------------------------------------------------------------------ #
    # Pipeline stages (each also usable standalone)
    # ------------------------------------------------------------------ #

    def enhance_contrast(self, img: Image.Image) -> Image.Image:
        """Stage 1: push the image toward pure black/white via contrast enhancement."""
        return ImageEnhance.Contrast(img).enhance(self.contrast_factor)

    def threshold_and_dilate(self, img: Image.Image) -> np.ndarray:
        """Stage 2: Otsu-binarize (dark lines on light bg -> white lines on
        black bg) and dilate to pre-close small gaps before skeletonization.
        """
        img_arr = np.array(img)
        _, binary = cv2.threshold(img_arr, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        kernel = np.ones((self.dilate_kernel_size, self.dilate_kernel_size), np.uint8)
        dilated = cv2.dilate(binary, kernel, iterations=self.dilate_iterations)
        return dilated

    def to_skeleton(self, binary_arr: np.ndarray) -> np.ndarray:
        """Stage 3: thin the white-on-black lines to a 1 px skeleton. (Otsu
        on the already binary image just reproduces its 0/255 split.)"""
        thresh = threshold_otsu(binary_arr)
        binary = binary_arr > thresh  # True where the (white) line is
        skeleton = skeletonize(binary)
        return (skeleton * 255).astype(np.uint8)

    def repair_skeleton(self, skeleton: np.ndarray) -> np.ndarray:
        """Stage 4: connect nearby endpoints to close remaining gaps.

        Steps: optional noise removal -> endpoint detection ->
        nearest-endpoint connection -> optional re-skeletonization.
        """
        binary = (skeleton > 0).astype(np.uint8) * 255

        if self.min_component_size > 0:
            binary = self._remove_small_components(binary, self.min_component_size)

        connected = self._connect_nearby_endpoints(
            binary, max_distance=self.max_distance, thickness=self.line_thickness
        )

        if self.re_skeletonize:
            thin = skeletonize(connected > 0)
            connected = (thin * 255).astype(np.uint8)

        return connected

    def thicken(self, binary: np.ndarray, thickness: int) -> np.ndarray:
        """Stage 5: thicken the lines to about `thickness` px, by dilating
        with a `thickness` x `thickness` square."""
        if thickness <= 1:
            return binary
        kernel = np.ones((thickness, thickness), np.uint8)
        return cv2.dilate(binary, kernel, iterations=1)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _diff_overlay(before: np.ndarray, after: np.ndarray) -> np.ndarray:
        """BGR image comparing two white-lines-on-black masks, drawn as
        dark lines on white: black = in both, red = only in `after`
        (added), blue = only in `before` (removed)."""
        b, a = before > 0, after > 0
        vis = np.full((*before.shape, 3), 255, np.uint8)
        vis[b & a] = (0, 0, 0)
        vis[a & ~b] = (0, 0, 255)
        vis[b & ~a] = (255, 0, 0)
        return vis

    @staticmethod
    def _save_debug_image(stages: dict, out_path: Path) -> None:
        """Write every stage from `process_image(..., return_stages=True)`
        side by side, each with its label above it."""
        panels = []
        for label, arr in stages.items():
            panel = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR) if arr.ndim == 2 else arr
            header = np.full((24, panel.shape[1], 3), 255, np.uint8)
            cv2.putText(header, label, (4, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
            panels.append(np.vstack([header, panel]))
            # thin gray separator between panels
            panels.append(np.full((panels[-1].shape[0], 4, 3), 128, np.uint8))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), np.hstack(panels[:-1]))

    @staticmethod
    def _remove_small_components(binary: np.ndarray, min_size: int) -> np.ndarray:
        """Drop connected components smaller than `min_size` pixels (noise removal)."""
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        cleaned = np.zeros_like(binary)
        for i in range(1, n_labels):  # 0 = background
            if stats[i, cv2.CC_STAT_AREA] >= min_size:
                cleaned[labels == i] = 255
        return cleaned

    @staticmethod
    def _find_skeleton_endpoints(skeleton: np.ndarray) -> np.ndarray:
        """Return (row, col) coordinates of skeleton endpoints
        (skeleton pixels with exactly one skeleton neighbour)."""
        skel = (skeleton > 0).astype(np.uint8)
        kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]])
        neighbor_count = convolve(skel, kernel, mode="constant", cval=0)
        # value 11 = the pixel itself (10) plus exactly 1 neighbour
        endpoints = np.argwhere(neighbor_count == 11)
        return endpoints

    def _connect_nearby_endpoints(
        self, skeleton: np.ndarray, max_distance: int, thickness: int = 1
    ) -> np.ndarray:
        """Draw a line from each endpoint to its nearest endpoint if that is
        at most `max_distance` px away. Each endpoint is joined at most once."""
        result = skeleton.copy()
        endpoints = self._find_skeleton_endpoints(skeleton)

        if len(endpoints) < 2:
            return result

        dist_matrix = cdist(endpoints, endpoints)
        np.fill_diagonal(dist_matrix, np.inf)

        connected: set[int] = set()
        for i in range(len(endpoints)):
            if i in connected:
                continue
            j = int(np.argmin(dist_matrix[i]))
            if dist_matrix[i, j] <= max_distance and j not in connected:
                pt1 = tuple(endpoints[i][::-1])  # (row, col) -> (x, y) for cv2
                pt2 = tuple(endpoints[j][::-1])
                cv2.line(result, pt1, pt2, 255, thickness=thickness)
                connected.add(i)
                connected.add(j)

        return result