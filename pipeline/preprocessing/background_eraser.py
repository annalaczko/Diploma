"""BackgroundEraser: the first preprocessing step. It finds the sherd
drawing on a raw scan with a YOLO model, whites out everything else and
cuts out a square around it.

Per image (`_clear_one`):
    1. YOLO gives a box (detection mode) or an outline polygon
       (segmentation mode) of the sherd.
    2. The region is grown a little (`box_margin` / `mask_margin`), so a
       slightly too tight detection doesn't cut off thin details.
    3. Everything outside it is painted white.
    4. A square is cut out around it: fitted to the sherd, or with
       `crop_size` a fixed fraction of the page width.
"""

import os
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO

from pipeline.preprocessing.axes import METADATA_CSV, axis_row, has_axes, write_metadata_csv
from pipeline.preprocessing.axis_detection import detect_axes
from pipeline.preprocessing.dataset import ImageRecord, collect_image_paths, flatten_name
from pipeline.preprocessing.preprocessing_step import PreprocessingStep


class BackgroundEraser(PreprocessingStep):
    """Removes the background around the sherd and crops a square around it
    (see the module docstring for the steps).

    It also records one metadata row per image in `axes_table` (see
    `axes.py`): the scan's size and resolution (DPI) and where the crop lies
    on the scan, so sizes in the final image can be converted back to
    centimetres. With `record_axes=True` the row also holds the drawing's
    coordinate system (the vertical rotation axis and the horizontal rim
    line, found by `axis_detection.detect_axes`), in the crop's pixel
    coordinates. The axes are not drawn into the image here; later steps can
    do that (`Vectorizer(axes="draw")`, `AxisRenderer`).
    """
    def __init__(self,
                 input_path=None,
                 output_path=None,
                 mode = 'detection',
                 model='/caa/Homes01/alaczko/Diploma/Diploma/models/yolov8n_seg_finetuned.pt',
                 run_id="default_run",
                 batch_size=256,
                 box_margin=0.15,
                 mask_margin=0.10,
                 crop_size=None,
                 debug_images=10,
                 record_axes=False,
                 num_workers=None,
                 half=False,
                 image_extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"),
                 **kwargs):
        """
        Args:
            input_path: path to the root folder containing images. Defaults to None.
            output_path: where cleaned images are written; auto-derived
                from `run_id` when omitted (see Step).
            mode: 'detection' (default) whites out everything outside the
                (padded) bounding box; 'segmentation' whites out everything
                outside the YOLO segmentation mask (grown by `mask_margin`)
                and crops around that mask.
            model: path to the YOLO segmentation weights to use.
            run_id: identifier used to namespace the default output path (see Step).
            batch_size: number of images YOLO processes per inference batch
                (passed through to `model.predict`'s `batch` argument in
                `_detect_shards`). Lower this if you hit CUDA out-of-memory
                errors - GPU memory use scales with this, not with the
                total number of input images.
            box_margin: fraction of the detection box's width/height to
                pad outward on every side before it's used for
                whiting-out (`mode='detection'` only) and cropping (see
                `_expand_box`). A raw YOLO box is often drawn tight
                against the shard's visible edge, so a small margin keeps
                a thin protruding detail (a handle, spout, ...) from
                being clipped. 0.15 (the default) grows the box by 15% of
                its own size on each side, clamped to the image bounds.
            mask_margin: how much to grow the segmentation mask before
                whiting-out and cropping (`mode='segmentation'` only), as a
                fraction of the mask's longer side: 0.10 (the default)
                makes the mask ~10% larger overall, i.e. a uniform border
                of 5% of its size all around (see `_dilate_radius`). Like
                `box_margin`, this keeps a slightly-too-tight mask from
                clipping the shard's outline.
            crop_size: side of the square crop. None (the default) fits
                it to the shard (the grown mask or padded box), so every
                shard fills its image. A number makes it that fraction of
                the scan's width instead (0.5 = half the page width),
                still centered on the shard: every crop then covers the
                same part of the page, so after resizing the shards keep
                their sizes relative to each other. A shard larger than
                the square is cut off at its edges; such images are
                listed in `clipped`.
            debug_images: how many of the first processed images also get
                a debug overlay (original image with the YOLO mask/box, the
                grown mask and the crop square drawn on it), written to
                `<output folder>/_debug` (see
                `_save_debug_image`). 0 disables it.
            record_axes: also find and record each scan's coordinate system
                (see the class docstring). Off by default, e.g. for photos
                without one, where the search would only cost time and
                mark every image as `no_axes`. Use with
                `mode='segmentation'`: in detection mode the "mask" is the
                whole box, so axis lines crossing it stay visible.
            num_workers: threads that clear (and search for axes on) the
                images of a batch in parallel; None = min(16, CPU count),
                0 or 1 = one at a time. The results don't depend on it.
            half: run YOLO in half precision (FP16, GPU only): faster, but
                masks can differ by a pixel here and there from FP32.
            image_extensions: file suffixes treated as input images.
            **kwargs: extra options, e.g. device='cuda'/'cpu' for inference.
        """

        super().__init__(input_path=input_path, output_path=output_path, run_id=run_id)
        self.mode = mode
        self.model = model
        self.batch_size = batch_size
        self.box_margin = box_margin
        self.mask_margin = mask_margin
        self.crop_size = crop_size
        self.debug_images = debug_images
        self.kwargs = kwargs
        self.image_extensions = image_extensions
        self._model = None
        self._debug_saved = 0  # in-memory mode: debug overlays saved so far, across batches
        # Flat names (see `_flatten_output_name`) of the images skipped
        # because YOLO found no shard on them, and how many images were
        # seen in total; reset by `begin_run`, reported by `log_summary`.
        self.no_shard: list[str] = []
        self._n_seen = 0
        # With a fixed `crop_size`: images whose shard didn't fit the square.
        self.clipped: list[str] = []
        self.record_axes = record_axes
        self.num_workers = min(16, os.cpu_count() or 1) if num_workers is None else num_workers
        self.half = half
        # Flat image name -> metadata row (see axes.axis_row), and with
        # record_axes the names of images without a coordinate system.
        self.axes_table: dict[str, dict] = {}
        self.no_axes: list[str] = []

    # Pipeline links later steps that draw the axes to `axes_table`.
    @property
    def provides_axes(self) -> bool:
        return self.record_axes

    # Pipeline writes the last metadata provider's rows to extracted_metadata.csv.
    provides_metadata = True

    def _get_model(self):
        """Load the YOLO model once and reuse it (in-memory mode calls
        `process_batch` many times)."""
        if self._model is None:
            self._model = YOLO(self.model)
        return self._model

    def _device(self):
        return self.kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    def _debug_dir(self) -> Path:
        return self.side_output_dir("debug")

    def process_batch(self, records: list[ImageRecord]) -> list[ImageRecord]:
        """In-memory mode: run YOLO on one batch of images and return the
        background-erased, cropped versions (grayscale scans stay
        grayscale, everything else becomes RGB), keeping their names.
        Images with no detection are dropped and logged, as in `run()`."""
        images = [self._as_array(r.image) for r in records]
        model_results = self._predict([self._bgr(img) for img in images])

        work = []
        self._n_seen += len(records)
        for record, img, model_result in zip(records, images, model_results):
            item = self._format_result(model_result, record.name)
            if item is None:
                # Not printed per image: the count is printed at the end
                # (log_summary) and the names go to MLflow as no_shard.txt.
                self.no_shard.append(record.name)
                continue
            item["dpi"] = _dpi(record.image.info)
            work.append((record.name, img, item))

        def clear(index, name, img, item):
            out, debug_info, row = self._clear_one(img, item, crop=True, box_margin=self.box_margin,
                                                    mask_margin=self.mask_margin)
            if self._debug_saved + index < self.debug_images:
                self._save_debug_image(img, *debug_info, self._debug_dir() / name)
            return out, row

        cleared = self._map(clear, [(i, *args) for i, args in enumerate(work)])
        self._debug_saved = min(self.debug_images, self._debug_saved + len(work))

        output = []
        for (name, _, item), (out, row) in zip(work, cleared):
            self._store_row(item, row)
            if out.ndim == 3:
                out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
            output.append(ImageRecord(name, Image.fromarray(out)))
        return output

    @staticmethod
    def _as_array(image: Image.Image) -> np.ndarray:
        """A PIL image as the array `_clear_one` works on: grayscale
        (H, W) for grayscale scans - a third of the work of three equal
        channels, with the same pixels - BGR (H, W, 3) for everything else."""
        if image.mode == "L":
            return np.asarray(image)
        return cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)

    @staticmethod
    def _bgr(img: np.ndarray) -> np.ndarray:
        """The 3-channel version YOLO needs."""
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img

    def _predict(self, source, **kwargs):
        if self.half:
            # ultralytics >= 8.4 replaced the deprecated `half=True` with
            # `quantize=16`; FP32 is the default, so nothing is passed then.
            kwargs["quantize"] = 16
        return self._get_model().predict(
            source,
            verbose=False,
            batch=self.batch_size,
            iou=0.3,
            device=self._device(),
            **kwargs,
        )

    def _map(self, function, args_list):
        """`[function(*args) for args in args_list]`, on `num_workers`
        threads (OpenCV and numpy release the GIL), in order."""
        if self.num_workers <= 1 or len(args_list) <= 1:
            return [function(*args) for args in args_list]
        with ThreadPoolExecutor(max_workers=self.num_workers) as pool:
            return list(pool.map(lambda args: function(*args), args_list))

    def run(self, keep_results=True):
        """Run YOLO segmentation on every input image and write cleaned
        (background-erased, cropped) versions to `self.output_path`.

        Args:
            keep_results: if True (default), returns `self.output_path`;
                if False, returns None. Useful when this step is chained
                purely for its side effect of writing files.
        """
        image_paths = collect_image_paths(self.input_path, self.image_extensions)
        if not image_paths:
            raise FileNotFoundError(
                f"No existing images with extensions {self.image_extensions} found under {self.input_path}"
            )

        self.begin_run()
        self._n_seen = len(image_paths)
        model = self._get_model()
        tmp_dir, path_map = self._make_flat_symlink_dir(image_paths)
        try:
            print("Detecting shards...")
            shards = self._detect_shards(model, tmp_dir, path_map, batch_size=self.batch_size)
            print("Clearing images...")
            self._clear_images(shards, output_dir=self.output_path, crop=True,
                               box_margin=self.box_margin, mask_margin=self.mask_margin,
                               debug_dir=self._debug_dir(), debug_images=self.debug_images)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        # The metadata (and axes) of every written image, in its crop's coordinates.
        write_metadata_csv(self.axes_table, Path(self.output_path) / METADATA_CSV)
        self.log_summary()
        return self.output_path if keep_results else None

    def begin_run(self):
        """Forget the images skipped (and axes recorded) in a previous run."""
        self._debug_saved = 0
        self.no_shard = []
        self._n_seen = 0
        self.clipped = []
        self.axes_table = {}
        self.no_axes = []

    def failure_lists(self) -> dict[str, list[str]]:
        """Image names skipped or only partly processed, by reason. Each
        entry becomes `<reason>.txt` and `<reason>_count` in MLflow."""
        lists = {"no_shard": self.no_shard}
        if self.record_axes:
            lists["no_axes"] = self.no_axes
        if self.crop_size is not None:
            lists["clipped"] = self.clipped
        return lists

    def _make_flat_symlink_dir(self, paths):
        """Symlink every image into a single flat temp directory with
        sequential zero-padded names (e.g. 000000.jpg, 000001.png, ...).

        YOLO's `model.predict` is pointed at this directory rather than at
        the original (possibly deeply nested) paths, since it walks a
        single flat folder more predictably than an arbitrary directory
        tree. The returned `path_map` lets callers translate the symlink
        paths YOLO reports back in its results to the real source paths.

        Returns:
            (tmp_dir, path_map): the temp directory path, and a dict
            mapping each symlink path to its original source path.
        """
        tmp_dir = tempfile.mkdtemp(prefix="yolo_batch_")
        path_map = {}
        for i, p in enumerate(paths):
            path = Path(p)
            ext = path.suffix
            link_path = os.path.join(tmp_dir, f"{i:06d}{ext}")
            # Absolute target: a relative one would resolve against tmp_dir,
            # leaving a dangling link that YOLO silently skips.
            os.symlink(os.path.abspath(path), link_path)
            path_map[link_path] = str(path)
        return tmp_dir, path_map

    def _detect_shards(self, model, tmp_dir, path_map, batch_size=256):
        """Run the YOLO model over the flat symlink directory and collect
        one segmentation result per successfully detected image.

        Streams predictions (`stream=True`) so results are processed one
        at a time instead of all being held in memory at once, which
        matters for large batches of high-resolution images.

        Args:
            model: a loaded `ultralytics.YOLO` model.
            tmp_dir: the flat symlink directory to run inference over.
            path_map: symlink path -> original path, from
                `_make_flat_symlink_dir`, used to report results (and
                failures) against the real source paths.
            batch_size: inference batch size passed to `model.predict`.

        Returns:
            A list of dicts, one per successfully segmented image, each
            with 'path' (original path), 'masks' (polygon points) and
            'boxes' (bounding boxes) as produced by YOLO. Images with no
            detected mask, or that raised during processing, are skipped
            and logged instead of raising.
        """
        model_results = self._predict(tmp_dir, stream=True)

        results_formatted = []
        failed_paths = []

        for idx, model_result in enumerate(
            tqdm(model_results, total=len(path_map), desc="Detecting shards", unit="image")
        ):
            symlink_path = model_result.path
            original_path = path_map.get(symlink_path, symlink_path)
            try:
                result = self._format_result(model_result, original_path)
                if result is None:
                    failed_paths.append(original_path)
                    continue
                results_formatted.append(result)
            except Exception as e:
                print(f"Error processing image {original_path} (index {idx}): {e}")
                failed_paths.append(original_path)
        self.no_shard.extend(self._flatten_output_name(p) for p in failed_paths)

        print(f"Detected {len(results_formatted)} shard(s), failed on {len(failed_paths)}")

        return results_formatted
    
    def _format_result(self, model_result, path):
        """Turn one YOLO result into the `{'path', 'masks'?, 'boxes'}` dict
        `_clear_one` expects, or None if nothing was detected."""
        result = {"path": path}
        if self.mode == "segmentation":
            if getattr(model_result, "masks", None) is None:
                return None
            result["masks"] = model_result.masks.xy
            result["boxes"] = model_result.boxes.xyxy
        elif self.mode == "detection":
            # No mask to check here - a detection with zero predicted
            # boxes is the equivalent "nothing found" case, so it needs the
            # same skip-and-log treatment as a missing mask above
            # (otherwise `_clear_one` would index into an empty boxes tensor).
            boxes = model_result.boxes.xyxy
            if boxes is None or len(boxes) == 0:
                return None
            result["boxes"] = boxes
        return result

    def _clear_images(self, image_dict, output_dir=None, crop=True, box_margin=0.15, mask_margin=0.10,
                      debug_dir=None, debug_images=0):
        """Whites out every pixel outside the detected mask for each image
        and, if `crop` is True, crops to a square canvas centered on its
        detection box (padded with white where the box runs off-image).

        Args:
            image_dict: list of dicts with 'path', 'masks' and 'boxes'
                entries, as produced by `_detect_shards`.
            output_dir: if given, each processed image is written here
                under a flattened, collision-free filename that encodes its
                ancestor directory names (see `_flatten_output_name`).
            crop: whether to crop to a square canvas around the detection
                box, in addition to whiting out the background.
            box_margin: see `__init__`'s docstring - padded outward on
                every side of the detection box (via `_expand_box`)
                before it's used for whiting-out (detection mode) and
                for cropping, so a tightly-drawn box doesn't clip a
                detail right at its edge.
            mask_margin: see `__init__`'s docstring - how much the
                segmentation mask is grown (via `_dilate_radius`) before
                whiting-out; the crop is then centered on the grown
                mask's bounding rectangle instead of the detection box.
            debug_dir: where to write debug overlays (see
                `_save_debug_image`); None disables them.
            debug_images: how many of the first images get a debug overlay.

        Returns:
            list of dicts with 'path' and the processed 'image' array.
        """
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        def clear(index, item):
            path = item['path']
            # ANYCOLOR: grayscale scans stay one channel (see `_as_array`).
            img = cv2.imread(path, cv2.IMREAD_ANYCOLOR)
            if img is None:
                print(f"Failed to read image: {path}")
                return None
            with Image.open(path) as header:  # lazy: reads only the header
                item['dpi'] = _dpi(header.info)
            out, debug_info, row = self._clear_one(img, item, crop=crop, box_margin=box_margin,
                                                    mask_margin=mask_margin)
            if debug_dir is not None and index < debug_images:
                self._save_debug_image(img, *debug_info, Path(debug_dir) / self._flatten_output_name(path))
            if output_dir:
                cv2.imwrite(os.path.join(output_dir, self._flatten_output_name(path)), out)
            return out, row

        results = []
        cleared = self._map(clear, list(enumerate(tqdm(image_dict, desc="Clearing images"))))
        for item, done in zip(image_dict, cleared):
            if done is None:
                continue
            out, row = done
            self._store_row(item, row)
            results.append({'path': item['path'], 'image': out})
        return results

    def _clear_one(self, img, item, crop=True, box_margin=0.15, mask_margin=0.10):
        """Whiten everything outside the detected shard in one BGR (or
        grayscale) image and (if `crop`) cut a white-padded square around
        it. Shared by the disk (`_clear_images`) and in-memory
        (`process_batch`) modes; safe to run on several threads at once
        (it changes nothing on `self`).

        Only the part of the scan the square covers is touched: the mask is
        rasterized, grown and applied there (plus a border of the growth
        radius), never on the whole A4 sheet, which is most of the work
        otherwise. The result is the same as doing it on the whole sheet.

        Args:
            img: the original BGR (H, W, 3) or grayscale (H, W) image.
            item: one `_format_result` dict ('boxes', and 'masks' in
                segmentation mode; 'dpi', the scan's resolution, if known).
            crop, box_margin, mask_margin: see `_clear_images`.

        Returns:
            (out, debug_info, row): the processed image (same channels as
            `img`), the `(box, polygon, mask, mask_offset, square)` tuple
            `_save_debug_image` draws, and the metadata row (see
            `axes.axis_row`; with the axes if `record_axes`).
        """
        h, w = img.shape[:2]

        # .cpu() first since YOLO's boxes are a CUDA tensor when
        # running on GPU. Computed once and expanded here so both the
        # detection-mode polygon below and the crop further down use
        # the exact same padded box.
        box = np.asarray(item['boxes'][0].cpu(), dtype=np.float32).reshape(-1)  # (x1, y1, x2, y2)
        x1, y1, x2, y2 = self._expand_box(box, w, h, box_margin)

        segmented = 'masks' in item and len(item['masks']) > 0
        if segmented:
            polygon = np.asarray(item['masks'][0], dtype=np.float32)  # (N, 2) point list
        else:
            # Detection mode has no segmentation mask, only a
            # bounding box - turn its (padded) corners into an
            # explicit 4-corner rectangle, a valid (N, 2) polygon for
            # cv2.fillPoly below, same shape as the segmentation-mask
            # branch produces.
            polygon = np.array(
                [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32
            )
        points = polygon.astype(np.int32)

        radius = 0
        if segmented and len(points):
            # Bounding rectangle of the rasterized polygon (clipped to the
            # image, as fillPoly clips it), grown by the dilation radius:
            # the grown mask's extent, without rasterizing the whole sheet.
            px, py, pw, ph = cv2.boundingRect(points)
            px1, py1 = max(px, 0), max(py, 0)
            px2, py2 = min(px + pw, w), min(py + ph, h)
            if px2 > px1 and py2 > py1:
                radius = self._dilate_radius(px2 - px1, py2 - py1, mask_margin)
                # Crop around the grown mask itself rather than the YOLO
                # box, so the canvas fits exactly what was kept. Falls
                # back to the padded box if the polygon lies off-image.
                x1, y1 = max(px1 - radius, 0), max(py1 - radius, 0)
                x2, y2 = min(px2 + radius, w), min(py2 + radius, h)

        square_coords = None
        if crop:
            # Build a square canvas centered on the (already padded)
            # detection box - or, in segmentation mode, the grown
            # mask's bounding rectangle - big enough to fit its longer side, and
            # white-pad wherever the square runs off the edge of the
            # original image.
            box_w = x2 - x1
            box_h = y2 - y1
            side = max(box_w, box_h)
            if self.crop_size is not None:
                # Fixed fraction of the page width: the same scale for every image.
                fixed = self.crop_size * w
                item['clipped'] = fixed < side
                side = fixed

            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            half = side / 2.0

            # Single integer half-size shared by both axes so the
            # square canvas always has equal width and height
            half_i = int(round(half))
            side_i = 2 * half_i

            cx_i = int(round(cx))
            cy_i = int(round(cy))

            sq_x1_i = cx_i - half_i
            sq_x2_i = cx_i + half_i
            sq_y1_i = cy_i - half_i
            sq_y2_i = cy_i + half_i

            # Valid region that actually overlaps the image
            src_x1 = max(sq_x1_i, 0)
            src_y1 = max(sq_y1_i, 0)
            src_x2 = min(sq_x2_i, w)
            src_y2 = min(sq_y2_i, h)
            square_coords = (sq_x1_i, sq_y1_i, sq_x2_i, sq_y2_i)
        else:
            src_x1, src_y1, src_x2, src_y2 = 0, 0, w, h

        # Rasterize and grow the mask only around the region that is kept,
        # with a border of `radius` so the dilation sees every mask pixel
        # that reaches into the region.
        wx1, wy1 = max(src_x1 - radius - 1, 0), max(src_y1 - radius - 1, 0)
        wx2, wy2 = min(max(src_x2, wx1) + radius + 1, w), min(max(src_y2, wy1) + radius + 1, h)
        mask = np.zeros((wy2 - wy1, wx2 - wx1), dtype=np.uint8)
        cv2.fillPoly(mask, [points], 255, offset=(-wx1, -wy1))
        if radius >= 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
            mask = cv2.dilate(mask, kernel)

        region = np.full((max(src_y2 - src_y1, 0), max(src_x2 - src_x1, 0)) + img.shape[2:], 255, img.dtype)
        if src_x2 > src_x1 and src_y2 > src_y1:
            keep = mask[src_y1 - wy1:src_y2 - wy1, src_x1 - wx1:src_x2 - wx1].astype(bool)
            if img.ndim == 3:
                keep = keep[..., None]
            region = np.where(keep, img[src_y1:src_y2, src_x1:src_x2], region)

        if crop:
            # Where that region lands inside the (white) square canvas
            dst_x1 = src_x1 - sq_x1_i
            dst_y1 = src_y1 - sq_y1_i
            out = np.full((side_i, side_i) + img.shape[2:], 255, dtype=img.dtype)
            if region.size:
                out[dst_y1:dst_y1 + region.shape[0], dst_x1:dst_x1 + region.shape[1]] = region
        else:
            out = region

        row = self._metadata_row(img, item, square_coords, out)
        return out, (box, polygon, mask, (wx1, wy1), square_coords), row

    def _metadata_row(self, img, item, crop_box, out) -> dict:
        """The image's metadata row (see `axes.axis_row`) in the crop's
        (`crop_box`) coordinates; with `record_axes`, the coordinate system
        is searched on the original scan."""
        name = self._flatten_output_name(item.get("path", ""))
        detection = None
        if self.record_axes:
            gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            detection = detect_axes(gray)
        return axis_row(name, detection, crop_box, (out.shape[1], out.shape[0]),
                        scan_size=(img.shape[1], img.shape[0]), dpi=item.get("dpi"))

    def _store_row(self, item, row) -> None:
        """Record one image's metadata row in `axes_table` (and, with
        `record_axes`, `no_axes`). Called in input order, after the
        parallel `_clear_one`s."""
        name = self._flatten_output_name(item.get("path", ""))
        self.axes_table[name] = row
        if item.get('clipped'):
            self.clipped.append(name)
        if self.record_axes and not has_axes(row):
            self.no_axes.append(name)  # counted at the end (log_summary), names in MLflow

    @staticmethod
    def _save_debug_image(img, box, polygon, mask, mask_offset, square, out_path):
        """Write the original image with what the eraser did drawn on it:
            red   - the raw YOLO detection box,
            yellow - the raw YOLO mask polygon (in detection mode this is
                     just the padded box),
            green - the outline of the final (grown) mask; everything
                    outside it was whited out,
            blue  - the square crop window (may run off-image, where the
                    output is white-padded).

        Args:
            img: the original BGR image (not modified).
            box: raw YOLO box `(x1, y1, x2, y2)`.
            polygon: (N, 2) polygon the mask was rasterized from.
            mask: uint8 final mask of the region around the crop.
            mask_offset: `(x, y)` of the mask's top-left corner in `img`.
            square: crop square `(x1, y1, x2, y2)`, or None if not cropping.
            out_path: file to write.
        """
        vis = img.copy() if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        t = max(2, round(max(img.shape[:2]) / 300))  # line width scaled to image size
        bx1, by1, bx2, by2 = (int(round(v)) for v in box)
        cv2.rectangle(vis, (bx1, by1), (bx2, by2), (0, 0, 255), t)
        cv2.polylines(vis, [polygon.astype(np.int32)], True, (0, 255, 255), t)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE, offset=mask_offset)
        cv2.drawContours(vis, contours, -1, (0, 255, 0), t)
        if square is not None:
            cv2.rectangle(vis, square[:2], square[2:], (255, 0, 0), t)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), vis)

    @staticmethod
    def _dilate_radius(width, height, margin) -> int:
        """Radius of the disk-shaped dilation that grows a mask with this
        bounding rectangle so it is about `margin` larger along its longer
        side (`margin / 2` of that side added all around); 0 = no growth.

        Dilation rather than scaling the polygon about its center: it adds
        the same border everywhere, so a concave or elongated shard's thin
        edges get the same safety margin as its bulk.
        """
        if margin <= 0:
            return 0
        radius = int(round(max(width, height) * margin / 2))
        return radius if radius >= 1 else 0

    @staticmethod
    def _expand_box(box, image_width, image_height, margin):
        """Grow a `(x1, y1, x2, y2)` box by `margin` times its own
        width/height on every side (0.15: 15% left and 15% right), clamped
        to the image. Returns a float32 array."""
        x1, y1, x2, y2 = box
        pad_x = (x2 - x1) * margin
        pad_y = (y2 - y1) * margin
        return np.array([
            max(x1 - pad_x, 0.0),
            max(y1 - pad_y, 0.0),
            min(x2 + pad_x, image_width),
            min(y2 + pad_y, image_height),
        ], dtype=np.float32)

    def _flatten_output_name(self, path) -> str:
        """The output file name of a source image: its folders below
        `input_path` joined with "__" (see `dataset.flatten_name`), so the
        many `001.jpg`s of different subfolders don't overwrite each other
        in the flat output folder."""
        return flatten_name(path, self.input_path)


def _dpi(info: dict) -> float | None:
    """Horizontal resolution (dots per inch) from a PIL image's `info`, or
    None if the file doesn't store one."""
    dpi = info.get("dpi")
    if not dpi:
        return None
    try:
        # PNG stores pixels per metre, so 150 dpi reads back as 150.0124.
        value = round(float(dpi[0]), 1)
    except (TypeError, ValueError, IndexError):
        return None
    return value if value > 0 else None
