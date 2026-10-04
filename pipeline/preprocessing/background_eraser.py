"""Concrete preprocessing steps. Add new steps here or in new files,
each inheriting from PreprocessingStep."""
from pathlib import Path

import shutil
import tempfile

import cv2
import numpy as np
import os
import torch
from ultralytics import YOLO
from tqdm import tqdm

from pipeline.preprocessing.preprocessing_step import PreprocessingStep
from pipeline.preprocessing.dataset import collect_image_paths

class BackgroundEraser(PreprocessingStep):
    """Runs a YOLO segmentation model on each image, whites out everything
    outside the detected shard mask, and (optionally) crops to a square
    canvas centered on the detection box.
    """
    def __init__(self,
                 input_path=None,
                 mode = 'detection',
                 model='/caa/Homes01/alaczko/Diploma/Diploma/models/yolov8n_seg_finetuned.pt',
                 run_id="default_run",
                 batch_size=256,
                 box_margin=0.15,
                 mask_margin=0.10,
                 debug_images=10,
                 image_extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"),
                 **kwargs):
        """
        Args:
            input_path: path to the root folder containing images. Defaults to None.
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
                of 5% of its size all around (see `_dilate_mask`). Like
                `box_margin`, this keeps a slightly-too-tight mask from
                clipping the shard's outline.
            debug_images: how many of the first processed images also get
                a debug overlay (original image with the YOLO mask/box, the
                grown mask and the crop square drawn on it), written to a
                sibling `<output folder>_debug` directory (see
                `_save_debug_image`). 0 disables it.
            image_extensions: file suffixes treated as input images.
            **kwargs: extra options, e.g. device='cuda'/'cpu' for inference.
        """

        super().__init__(input_path=input_path, run_id=run_id)
        self.mode = mode
        self.model = model
        self.batch_size = batch_size
        self.box_margin = box_margin
        self.mask_margin = mask_margin
        self.debug_images = debug_images
        self.kwargs = kwargs
        self.image_extensions = image_extensions

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

        model = YOLO(self.model)
        tmp_dir, path_map = self._make_flat_symlink_dir(image_paths)
        try:
            print("Detecting shards...")
            shards = self._detect_shards(model, tmp_dir, path_map, batch_size=self.batch_size)
            print("Clearing images...")
            # Sibling folder, not a subfolder: later steps collect their
            # input images recursively, so debug overlays inside the output
            # folder would be picked up as training data.
            output = Path(self.output_path)
            debug_dir = output.parent / f"{output.name}_debug"
            self._clear_images(shards, output_dir=self.output_path, crop=True,
                               box_margin=self.box_margin, mask_margin=self.mask_margin,
                               debug_dir=debug_dir, debug_images=self.debug_images)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        return self.output_path if keep_results else None

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
            os.symlink(str(path), link_path)
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
        device = self.kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")

        try:
            model_results = model.predict(
                tmp_dir,
                verbose=False,
                batch=batch_size,
                iou=0.3,
                stream=True,
                device=device,
            )
        except Exception as e:
            print(f"Error when calling model.predict: {e}")
            raise

        results_formatted = []
        failed_paths = []

        for idx, model_result in enumerate(
            tqdm(model_results, total=len(path_map), desc="Detecting shards", unit="image")
        ):
            symlink_path = model_result.path
            original_path = path_map.get(symlink_path, symlink_path)
            try:
                result = {"path": original_path}
                if self.mode == "segmentation":
                    if getattr(model_result, "masks", None) is None:
                        failed_paths.append(original_path)
                        continue
                    result["masks"] = model_result.masks.xy
                    result["boxes"] = model_result.boxes.xyxy
                elif self.mode == "detection":
                    # No mask to check here - a detection with zero
                    # predicted boxes is the equivalent "nothing found"
                    # case, so it needs the same skip-and-log treatment
                    # as a missing mask above (otherwise `_clear_images`
                    # would later index into an empty boxes tensor).
                    boxes = model_result.boxes.xyxy
                    if boxes is None or len(boxes) == 0:
                        failed_paths.append(original_path)
                        continue
                    result["boxes"] = boxes
                results_formatted.append(result)
            except Exception as e:
                print(f"Error processing image {original_path} (index {idx}): {e}")
                failed_paths.append(original_path)

        print(f"Detected {len(results_formatted)} shard(s), failed on {len(failed_paths)}")
        if failed_paths:
            for p in failed_paths:
                print(f"  - {p}")

        return results_formatted
    
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
                segmentation mask is grown (via `_dilate_mask`) before
                whiting-out; the crop is then centered on the grown
                mask's bounding rectangle instead of the detection box.
            debug_dir: where to write debug overlays (see
                `_save_debug_image`); None disables them.
            debug_images: how many of the first images get a debug overlay.

        Returns:
            list of dicts with 'path' and the processed 'image' array.
        """
        results = []
        debug_saved = 0
        for item in tqdm(image_dict, desc="Clearing images"):
            path = item['path']

            img = cv2.imread(path)
            if img is None:
                print(f"Failed to read image: {path}")
                continue

            h, w = img.shape[:2]

            # .cpu() first since YOLO's boxes are a CUDA tensor when
            # running on GPU. Computed once and expanded here so both the
            # detection-mode polygon below and the crop further down use
            # the exact same padded box.
            box = np.asarray(item['boxes'][0].cpu(), dtype=np.float32).reshape(-1)  # (x1, y1, x2, y2)
            x1, y1, x2, y2 = self._expand_box(box, w, h, box_margin)

            if 'masks' not in item or not item['masks']:
                # Detection mode has no segmentation mask, only a
                # bounding box - turn its (padded) corners into an
                # explicit 4-corner rectangle, a valid (N, 2) polygon for
                # cv2.fillPoly below, same shape as the segmentation-mask
                # branch produces.
                polygon = np.array(
                    [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32
                )
            else:
                polygon = np.asarray(item['masks'][0], dtype=np.float32)  # (N, 2) point list

            # Rasterize a binary mask from the polygon
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [polygon.astype(np.int32)], 255)

            if 'masks' in item and item['masks']:
                mask = self._dilate_mask(mask, mask_margin)
                # Crop around the grown mask itself rather than the YOLO
                # box, so the canvas fits exactly what was kept. Falls
                # back to the padded box if the polygon was degenerate.
                ys, xs = np.nonzero(mask)
                if len(xs):
                    x1, y1, x2, y2 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1

            debug_mask = mask
            mask = mask.astype(bool)

            white_bg = np.full_like(img, 255)
            out = np.where(mask[..., None], img, white_bg)

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

                # Where that region lands inside the (white) square canvas
                dst_x1 = src_x1 - sq_x1_i
                dst_y1 = src_y1 - sq_y1_i
                dst_x2 = dst_x1 + (src_x2 - src_x1)
                dst_y2 = dst_y1 + (src_y2 - src_y1)

                square = np.full((side_i, side_i, img.shape[2]), 255, dtype=out.dtype)
                if src_x2 > src_x1 and src_y2 > src_y1:
                    square[dst_y1:dst_y2, dst_x1:dst_x2] = out[src_y1:src_y2, src_x1:src_x2]

                out = square
                square_coords = (sq_x1_i, sq_y1_i, sq_x2_i, sq_y2_i)

            if debug_dir is not None and debug_saved < debug_images:
                self._save_debug_image(
                    img, box, polygon, debug_mask, square_coords,
                    Path(debug_dir) / self._flatten_output_name(path),
                )
                debug_saved += 1

            results.append({'path': path, 'image': out})

            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                out_path = os.path.join(output_dir, self._flatten_output_name(path))
                cv2.imwrite(out_path, out)

        return results

    @staticmethod
    def _save_debug_image(img, box, polygon, mask, square, out_path):
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
            mask: (H, W) uint8 final mask.
            square: crop square `(x1, y1, x2, y2)`, or None if not cropping.
            out_path: file to write.
        """
        vis = img.copy()
        t = max(2, round(max(img.shape[:2]) / 300))  # line width scaled to image size
        bx1, by1, bx2, by2 = (int(round(v)) for v in box)
        cv2.rectangle(vis, (bx1, by1), (bx2, by2), (0, 0, 255), t)
        cv2.polylines(vis, [polygon.astype(np.int32)], True, (0, 255, 255), t)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, (0, 255, 0), t)
        if square is not None:
            cv2.rectangle(vis, square[:2], square[2:], (255, 0, 0), t)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), vis)

    @staticmethod
    def _dilate_mask(mask, margin):
        """Grow a binary uint8 mask by a uniform border, so the result is
        about `margin` larger than the original along its longer side
        (`margin / 2` of that side added all around).

        Uses morphological dilation with a disk-shaped kernel rather than
        scaling the polygon about its center: dilation adds the same
        border everywhere, so a concave or elongated shard's thin edges
        get the same safety margin as its bulk.

        Args:
            mask: (H, W) uint8 mask, nonzero inside the shard.
            margin: growth as a fraction of the mask's longer side; <= 0
                returns the mask unchanged.

        Returns:
            The dilated (H, W) uint8 mask (clipped to the image by
            construction).
        """
        ys, xs = np.nonzero(mask)
        if margin <= 0 or not len(xs):
            return mask
        size = max(xs.max() - xs.min(), ys.max() - ys.min()) + 1
        radius = int(round(size * margin / 2))
        if radius < 1:
            return mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
        return cv2.dilate(mask, kernel)

    @staticmethod
    def _expand_box(box, image_width, image_height, margin):
        """Pad a `(x1, y1, x2, y2)` box outward by `margin` (a fraction of
        its own width/height) on every side, then clamp to the image
        bounds so the result never runs off-image.

        E.g. `margin=0.15` grows the box by 15% of its width on the left
        and again on the right (30% wider overall), and the same on
        top/bottom - a small safety buffer around a YOLO detection, which
        is often drawn tight against the shard's visible edge and can
        otherwise clip a thin protruding detail (a handle, spout, ...).

        Args:
            box: `(x1, y1, x2, y2)` in pixel coordinates.
            image_width, image_height: clamp bounds - the padded box
                never extends past `[0, image_width]` x `[0, image_height]`.
            margin: fraction of the box's width/height to pad outward on
                each side.

        Returns:
            The padded, clamped `(x1, y1, x2, y2)` as a float32 ndarray.
        """
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
        """Build a unique output filename that encodes `path`'s ancestor
        directory names (relative to `self.input_path`), instead of just
        its bare basename.

        `output_dir` is a single flat directory, but `self.input_path` is
        often a tree of subfolders (e.g. one per survey/album) that reuse
        the same simple, sequential filenames (`001.jpg`, `002.jpg`, ...)
        independently of each other. Saving under the bare basename would
        make images from different subfolders silently overwrite one
        another whenever their names collide - keeping the parent folder
        names in the output filename instead makes every name unique.

        Args:
            path: the original (nested) source image path.

        Returns:
            A flat filename such as `"2012_Survey Gladiatorenschule__006.jpg"`
            for `<input_path>/2012_Survey Gladiatorenschule/006.jpg`. Falls
            back to just the basename if `path` isn't actually under
            `self.input_path` (e.g. `input_path` was a list of paths
            rather than one common root directory).
        """
        path = Path(path)
        try:
            relative = path.resolve().relative_to(Path(self.input_path).resolve())
            ancestor_dirs = relative.parts[:-1]
        except (TypeError, ValueError):
            ancestor_dirs = ()

        return "__".join([*ancestor_dirs, path.name])