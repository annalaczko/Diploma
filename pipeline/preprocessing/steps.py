"""Concrete preprocessing steps. Add new steps here or in new files,
each inheriting from PreprocessingStep."""
import cv2
import numpy as np
import os
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
            elif self.mode == 'detection':
                raise NotImplementedError("Detection mode not implemented yet.")

            results_formatted.append(result)

        if failed_paths:
            print(f"Warning: no mask detected for {len(failed_paths)} image(s):")
            for p in failed_paths:
                print(f"  - {p}")

        return results_formatted
    
    def _clear_images(self, image_dict, output_dir=None):
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

            results.append({'path': path, 'image': out})

            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                out_path = os.path.join(output_dir, os.path.basename(path))
                cv2.imwrite(out_path, out)

        return results