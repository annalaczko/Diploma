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
                 mode = 'segmentation',
                 model='/caa/Homes01/alaczko/Diploma/Diploma/models/yolov8n_seg_finetuned.pt',
                 run_id="default_run",
                 image_extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"),
                 **kwargs):
        """
        Args:
            input_path: path to the root folder containing images. Defaults to None.
            mode: detection mode; only 'segmentation' is implemented. Defaults to 'segmentation'.
            model: path to the YOLO segmentation weights to use.
            run_id: identifier used to namespace the default output path (see Step).
            image_extensions: file suffixes treated as input images.
            **kwargs: extra options, e.g. device='cuda'/'cpu' for inference.
        """

        super().__init__(input_path=input_path, run_id=run_id)
        self.mode = mode
        self.model = model
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
            shards = self._detect_shards(model, tmp_dir, path_map)
            print("Clearing images...")
            self._clear_images(shards, output_dir=self.output_path, crop=True)
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
                    raise NotImplementedError("Detection mode not implemented yet.")
                results_formatted.append(result)
            except Exception as e:
                print(f"Error processing image {original_path} (index {idx}): {e}")
                failed_paths.append(original_path)

        print(f"Detected {len(results_formatted)} shard(s), failed on {len(failed_paths)}")
        if failed_paths:
            for p in failed_paths:
                print(f"  - {p}")

        return results_formatted
    
    def _clear_images(self, image_dict, output_dir=None, crop=True):
        """Whites out every pixel outside the detected mask for each image
        and, if `crop` is True, crops to a square canvas centered on its
        detection box (padded with white where the box runs off-image).

        Args:
            image_dict: list of dicts with 'path', 'masks' and 'boxes'
                entries, as produced by `_detect_shards`.
            output_dir: if given, each processed image is written here
                under its original filename.
            crop: whether to crop to a square canvas around the detection
                box, in addition to whiting out the background.

        Returns:
            list of dicts with 'path' and the processed 'image' array.
        """
        results = []
        for item in tqdm(image_dict, desc="Clearing images"):
            path = item['path']
            polygon = np.asarray(item['masks'][0], dtype=np.float32)  # (N, 2) point list

            img = cv2.imread(path)
            if img is None:
                print(f"Failed to read image: {path}")
                continue

            h, w = img.shape[:2]

            # Rasterize a binary mask from the polygon
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [polygon.astype(np.int32)], 255)
            mask = mask.astype(bool)

            white_bg = np.full_like(img, 255)
            out = np.where(mask[..., None], img, white_bg)

            if crop:
                # Build a square canvas centered on the detection box, big
                # enough to fit its longer side, and white-pad wherever the
                # square runs off the edge of the original image.
                box = np.asarray(item['boxes'][0].cpu(), dtype=np.float32).reshape(-1)  # (x1, y1, x2, y2)
                x1, y1, x2, y2 = box

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

            results.append({'path': path, 'image': out})

            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                out_path = os.path.join(output_dir, os.path.basename(path))
                cv2.imwrite(out_path, out)

        return results