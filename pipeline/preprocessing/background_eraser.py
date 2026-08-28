"""Concrete preprocessing steps. Add new steps here or in new files,
each inheriting from PreprocessingStep."""
from importlib.resources import path

import shutil
import tempfile
from xml.parsers.expat import model

import cv2
import numpy as np
import os
import time
import torch
from pyparsing import results
from tornado import gen
from ultralytics import YOLO
from tqdm import tqdm

from pipeline.preprocessing.preprocessing_step import PreprocessingStep

class BackgroundEraser(PreprocessingStep):
    """ Erases the background of images

    Args:
        PreprocessingStep (_type_): _description_
    """
    def __init__(self, input_path=None, mode = 'segmentation', model='/caa/Homes01/alaczko/Diploma/Diploma/models/yolov8n_seg_finetuned.pt', **kwargs):
        """_summary_

        Args:
            input_path (_type_, optional): _description_. Defaults to None.
            mode (str, optional): _description_. Defaults to 'segmentation'.
            model (str, optional): _description_. Defaults to '/caa/Homes01/alaczko/Diploma/Diploma/models/yolov8n_seg_finetuned.pt'.
            save_path (str, optional): _description_. Defaults to '/caa/Homes01/alaczko/Diploma/Diploma/data/processed/cleared_images'.
        """

        super().__init__(input_path = input_path)
        self.mode = mode
        self.model = model
        self.kwargs = kwargs

    def run(self, keep_results=True):
        model = YOLO(self.model)
        tmp_dir, path_map = self._make_flat_symlink_dir(self.input_path)
        try:
            print("_detectshards")
            shards = self._detect_shards(model, tmp_dir, path_map)
            print("clearing images")
            self._clear_images(shards, output_dir=self.output_path, crop=True)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        return self.output_path if keep_results else None

    def _make_flat_symlink_dir(self, paths):
        tmp_dir = tempfile.mkdtemp(prefix="yolo_batch_")
        path_map = {}
        for i, p in enumerate(paths):
            ext = os.path.splitext(p)[1]
            link_path = os.path.join(tmp_dir, f"{i:06d}{ext}")
            os.symlink(p, link_path)
            path_map[link_path] = p
        return tmp_dir, path_map

    def _detect_shards(self, model, tmp_dir, path_map, batch_size=256):
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
        """_summary_

        Args:
            image_dict (_type_): _description_
            output_dir (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        results = []
        for item in tqdm(image_dict, desc="Clearing images"):
            path = item['path']
            polygon = np.asarray(item['masks'][0], dtype=np.float32)  # (N, 2) pontlista

            img = cv2.imread(path)
            if img is None:
                print(f"Nem sikerült beolvasni: {path}")
                continue

            h, w = img.shape[:2]

            # Raszteres bináris maszk generálása a poligonból
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [polygon.astype(np.int32)], 255)
            mask = mask.astype(bool)

            white_bg = np.full_like(img, 255)
            out = np.where(mask[..., None], img, white_bg)

            if crop:
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
    
    def _clear_images(self, image_dict, output_dir=None, crop=True):
        """_summary_

        Args:
            image_dict (_type_): _description_
            output_dir (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        results = []
        for item in tqdm(image_dict, desc="Clearing images"):
            path = item['path']
            polygon = np.asarray(item['masks'][0], dtype=np.float32)  # (N, 2) pontlista

            img = cv2.imread(path)
            if img is None:
                print(f"Nem sikerült beolvasni: {path}")
                continue

            h, w = img.shape[:2]

            # Raszteres bináris maszk generálása a poligonból
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [polygon.astype(np.int32)], 255)
            mask = mask.astype(bool)

            white_bg = np.full_like(img, 255)
            out = np.where(mask[..., None], img, white_bg)

            if crop:
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