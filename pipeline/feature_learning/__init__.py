"""Feature learning steps: turn preprocessed images into fixed-length
feature vectors. Includes a VAE (unsupervised) and a classifier head that
can be trained on top of a VAE's encoder."""

from pipeline.feature_learning.featurelearning_step import FeatureLearningStep
from pipeline.feature_learning.vae import VAE
from pipeline.feature_learning.vae_classifier import VAEClassifier

__all__ = ["FeatureLearningStep", "VAE", "VAEClassifier"]
