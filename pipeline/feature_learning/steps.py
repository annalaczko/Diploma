"""Concrete feature learning steps."""

from typing import Any

import numpy as np

from pipeline.feature_learning.base import FeatureLearningStep


class PretrainedFeatureLearning(FeatureLearningStep):
    def __init__(self, backbone: str = "resnet50", normalize: bool = True):
        super().__init__()
        self.backbone = backbone
        self.normalize = normalize

    def run(self, data: Any) -> np.ndarray:
        # TODO: extract features from pretrained backbone
        features = np.zeros((len(data), 512))
        return self.l2_normalize(features) if self.normalize else features


class VAEFeatureLearning(FeatureLearningStep):
    def __init__(self, latent_dim: int = 128, epochs: int = 100, normalize: bool = True):
        super().__init__()
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.normalize = normalize

    def run(self, data: Any) -> np.ndarray:
        # TODO: train VAE, return latent vectors
        features = np.zeros((len(data), self.latent_dim))
        return self.l2_normalize(features) if self.normalize else features
