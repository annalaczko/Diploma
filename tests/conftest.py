"""Shared pytest fixtures."""

import sys
import types
from unittest.mock import MagicMock

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Mock heavy optional dependencies that are only available on the GPU server
# (cv2, ultralytics). This must happen before any pipeline module is imported.
# ---------------------------------------------------------------------------
def _mock_module(name: str) -> MagicMock:
    # No spec — lets patch() freely create attributes like cv2.imread, cv2.imwrite.
    mod = MagicMock()
    sys.modules[name] = mod
    return mod

for _dep in ("cv2", "ultralytics", "ultralytics.models"):
    if _dep not in sys.modules:
        _mock_module(_dep)

# Expose ultralytics.YOLO as a mock class so `from ultralytics import YOLO` works.
sys.modules["ultralytics"].YOLO = MagicMock()

from pipeline.clustering.steps import HierarchicalClustering, KMeansClustering
from pipeline.feature_learning.steps import PretrainedFeatureLearning, VAEFeatureLearning


@pytest.fixture
def sample_features():
    """10 db 512-dimenziós feature vektor."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((10, 512)).astype(np.float32)


@pytest.fixture
def sample_features_small():
    """5 db 4-dimenziós feature vektor – gyors unit tesztekhez."""
    rng = np.random.default_rng(0)
    return rng.standard_normal((5, 4)).astype(np.float32)


@pytest.fixture
def kmeans_step():
    return KMeansClustering(n_clusters=3, random_state=0)


@pytest.fixture
def hierarchical_step():
    return HierarchicalClustering(n_clusters=3, linkage="ward")


@pytest.fixture
def pretrained_fl():
    return PretrainedFeatureLearning(backbone="resnet50", normalize=True)


@pytest.fixture
def vae_fl():
    return VAEFeatureLearning(latent_dim=64, epochs=1, normalize=True)
