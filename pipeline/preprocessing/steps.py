"""Concrete preprocessing steps. Add new steps here or in new files,
each inheriting from PreprocessingStep."""
import cv2
import numpy as np
import os
from ultralytics import YOLO

from pipeline.preprocessing.base import PreprocessingStep


class ImageEnhancement(PreprocessingStep):
    def __init__(self, method: str = "denoise", **kwargs):
        super().__init__()
        self.method = method
        self.kwargs = kwargs

    # def run(self, data: Any) -> Any:
    #     # TODO: implement enhancement (e.g. denoising, contrast normalization)
    #     return data


class BackgroundEraser(PreprocessingStep):
    def __init__(self, image_paths=None, mode = 'segmentation', model='/caa/Homes01/alaczko/Diploma/Diploma/models/yolov8n_seg_finetuned.pt', **kwargs):
        super().__init__()
        self.image_paths = image_paths
        print(image_paths)
        self.mode = mode
        self.model = model
        self.kwargs = kwargs

    def run(self):
        model = YOLO(self.model)
      
        shards=self._detect_shards(self.image_paths, model)
        #shards_no_duplicates=self._erase_duplicates(shards)
        self._clear_images(shards, output_dir="../../data/processed/cleared_images")

        return 0
    
    def _detect_shards(self,image_paths, model):
        model_results = model.predict(image_paths,verbose=True, batch=32, iou=0.3)
        results_formatted = []
        for model_result in model_results:
            result={}
            result['path'] = model_result.path
            if self.mode=='segmentation':
                result['masks']=model_result.masks.xy
            elif self.mode=='detection':
                raise NotImplementedError("Detection mode not implemented yet.")
            
            results_formatted.append(result)

        return results_formatted
    
    def _clear_images(self, data, output_dir=None):
        """
        data: list of dict, kulcsok: 'path' (kép elérési útja), 'masks' (lista maszkokból)
        Az első maszkon kívül minden fehér lesz.
        """

        print(data)
        results = []
        for item in data:
            path = item['path']
            mask = np.asarray(item['masks'][0])  # csak az első maszk

            img = cv2.imread(path)
            if img is None:
                print(f"Nem sikerült beolvasni: {path}")
                continue

            # bináris maszkká alakítás, ha nem az
            if mask.dtype != bool:
                mask = mask > 0

            # ha a maszk mérete eltér a képtől, igazítsuk
            if mask.shape[:2] != img.shape[:2]:
                mask = cv2.resize(
                    mask.astype(np.uint8),
                    (img.shape[1], img.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                ).astype(bool)

            white_bg = np.full_like(img, 255)
            out = np.where(mask[..., None], img, white_bg)

            results.append({'path': path, 'image': out})

            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                out_path = os.path.join(output_dir, os.path.basename(path))
                cv2.imwrite(out_path, out)

        return results