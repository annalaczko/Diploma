"""Concrete preprocessing steps. Add new steps here or in new files,
each inheriting from PreprocessingStep."""
from importlib.resources import path

from os import path

import cv2
import numpy as np
import os
from pyparsing import results
from ultralytics import YOLO
from tqdm import tqdm

from pipeline.preprocessing.base import PreprocessingStep


class ImageEnhancement(PreprocessingStep):
    """_summary_

    Args:
        PreprocessingStep (_type_): _description_
    """
    def __init__(self, method: str = "denoise", **kwargs):
        super().__init__()
        self.method = method
        self.kwargs = kwargs

    # def run(self, data: Any) -> Any:
    #     # TODO: implement enhancement (e.g. denoising, contrast normalization)
    #     return data


class BackgroundEraser(PreprocessingStep):
    """ Erases the background of images

    Args:
        PreprocessingStep (_type_): _description_
    """
    def __init__(self, image_paths=None, mode = 'segmentation', model='/caa/Homes01/alaczko/Diploma/Diploma/models/yolov8n_seg_finetuned.pt', save_path="/caa/Homes01/alaczko/Diploma/Diploma/data/processed/cleared_images", **kwargs):
        """_summary_

        Args:
            image_paths (_type_, optional): _description_. Defaults to None.
            mode (str, optional): _description_. Defaults to 'segmentation'.
            model (str, optional): _description_. Defaults to '/caa/Homes01/alaczko/Diploma/Diploma/models/yolov8n_seg_finetuned.pt'.
            save_path (str, optional): _description_. Defaults to '/caa/Homes01/alaczko/Diploma/Diploma/data/processed/cleared_images'.
        """

        super().__init__()
        self.image_paths = image_paths
        self.save_path = save_path
        self.mode = mode
        self.model = model
        self.kwargs = kwargs

    def run(self, keep_results=False):
        """ Only public method, which is called by the pipeline. It runs the background erasing process.

        Returns:
            _type_: _description_
        """
        model = YOLO(self.model)
      
        shards=self._detect_shards(self.image_paths, model)
        #shards_no_duplicates=self._erase_duplicates(shards)
        self._clear_images(shards, output_dir=self.save_path)

        return 0
    
    
    """Concrete preprocessing steps. Add new steps here or in new files,
each inheriting from PreprocessingStep."""
from importlib.resources import path

from os import path

import cv2
import numpy as np
import os
from pyparsing import results
from ultralytics import YOLO
from tqdm import tqdm

from pipeline.preprocessing.base import PreprocessingStep


class ImageEnhancement(PreprocessingStep):
    """_summary_

    Args:
        PreprocessingStep (_type_): _description_
    """
    def __init__(self, method: str = "denoise", **kwargs):
        super().__init__()
        self.method = method
        self.kwargs = kwargs

    # def run(self, data: Any) -> Any:
    #     # TODO: implement enhancement (e.g. denoising, contrast normalization)
    #     return data


class BackgroundEraser(PreprocessingStep):
    """ Erases the background of images

    Args:
        PreprocessingStep (_type_): _description_
    """
    def __init__(self, image_paths=None, mode = 'segmentation', model='/caa/Homes01/alaczko/Diploma/Diploma/models/yolov8n_seg_finetuned.pt', save_path="/caa/Homes01/alaczko/Diploma/Diploma/data/processed/cleared_images", **kwargs):
        """_summary_

        Args:
            image_paths (_type_, optional): _description_. Defaults to None.
            mode (str, optional): _description_. Defaults to 'segmentation'.
            model (str, optional): _description_. Defaults to '/caa/Homes01/alaczko/Diploma/Diploma/models/yolov8n_seg_finetuned.pt'.
            save_path (str, optional): _description_. Defaults to '/caa/Homes01/alaczko/Diploma/Diploma/data/processed/cleared_images'.
        """

        super().__init__()
        self.image_paths = image_paths
        self.save_path = save_path
        self.mode = mode
        self.model = model
        self.kwargs = kwargs

    def run(self, keep_results=False):
        """ Only public method, which is called by the pipeline. It runs the background erasing process.

        Returns:
            _type_: _description_
        """
        model = YOLO(self.model)
      
        shards=self._detect_shards(self.image_paths, model)
        #shards_no_duplicates=self._erase_duplicates(shards)
        self._clear_images(shards, output_dir=self.save_path)

        return 0
    
    
    def _detect_shards(self, image_paths, model):
        """ detects the shards in the images using the provided model and returns the results in a formatted way.

        Args:
            image_paths (_type_): image paths to be processed
            model (_type_): model to be used for detection

        Raises:
            NotImplementedError: _description_

        Returns:
            _type_: _description_
        """

        model_results = model.predict(image_paths, verbose=True, batch=32, iou=0.3)
        results_formatted = []
        failed_paths = []

        for image_path, model_result in tqdm(
            zip(image_paths, model_results),
            total=len(image_paths),
            desc="Detecting shards"
        ):
            result = {}
            result['path'] = image_path

            if self.mode == 'segmentation':
                if model_result.masks is None:
                    failed_paths.append(image_path)
                    continue  # skip this image, no mask detected
                result['masks'] = model_result.masks.xy
                result['boxes'] = model_result.boxes.xyxy
            elif self.mode == 'detection':
                raise NotImplementedError("Detection mode not implemented yet.")

            results_formatted.append(result)

        if failed_paths:
            print(f"Warning: no mask detected for {len(failed_paths)} image(s):")
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