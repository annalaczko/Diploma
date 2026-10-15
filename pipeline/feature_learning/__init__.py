"""Feature learning steps: turn preprocessed images into fixed-length
feature vectors. Includes a VAE (unsupervised, reconstruction-based) and
two unsupervised self-supervised baselines - SimCLR (contrastive) and
BYOL (no negative pairs) - plus a classifier head that can be trained on
top of a VAE's encoder."""

from pipeline.feature_learning.featurelearning_step import FeatureLearningStep
from pipeline.feature_learning.early_stopping import EarlyStopping
from pipeline.feature_learning.vae import VAE
from pipeline.feature_learning.vae_classifier import VAEClassifier
from pipeline.feature_learning.simclr import SimCLR
from pipeline.feature_learning.byol import BYOL
from pipeline.feature_learning.handcrafted_features import HandcraftedFeatures

__all__ = ["FeatureLearningStep", "EarlyStopping", "VAE", "VAEClassifier", "SimCLR", "BYOL", "HandcraftedFeatures"]
