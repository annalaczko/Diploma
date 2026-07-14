"""Concrete preprocessing steps. Add new steps here or in new files,
each inheriting from PreprocessingStep."""

from typing import Any

from pipeline.preprocessing.base import PreprocessingStep


class ImageEnhancement(PreprocessingStep):
    def __init__(self, method: str = "denoise", **kwargs):
        super().__init__()
        self.method = method
        self.kwargs = kwargs

    def run(self, data: Any) -> Any:
        # TODO: implement enhancement (e.g. denoising, contrast normalization)
        return data


class ObjectDetection(PreprocessingStep):
    def __init__(self, model_name: str = "yolo", conf_threshold: float = 0.5):
        super().__init__()
        self.model_name = model_name
        self.conf_threshold = conf_threshold

    def run(self, data: Any) -> Any:
        # TODO: implement detection, return cropped sherd drawings
        return data
