"""Tests for feature learning steps."""

import numpy as np
import pytest

from pipeline.feature_learning.base import FeatureLearningStep
from pipeline.feature_learning.steps import PretrainedFeatureLearning, VAEFeatureLearning


# --- l2_normalize static method ---

class TestL2Normalize:
    def test_output_has_unit_norm(self):
        vecs = np.array([[3.0, 4.0], [1.0, 0.0], [0.0, 2.0]])
        out = FeatureLearningStep.l2_normalize(vecs)
        norms = np.linalg.norm(out, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    def test_zero_vector_does_not_raise(self):
        vecs = np.array([[0.0, 0.0], [1.0, 0.0]])
        out = FeatureLearningStep.l2_normalize(vecs)
        # zero vector stays zero (no NaN/inf)
        assert not np.any(np.isnan(out))
        assert not np.any(np.isinf(out))

    def test_already_normalized_unchanged(self):
        vecs = np.array([[1.0, 0.0], [0.0, 1.0]])
        out = FeatureLearningStep.l2_normalize(vecs)
        np.testing.assert_allclose(out, vecs, atol=1e-6)

    def test_output_shape_preserved(self, sample_features_small):
        out = FeatureLearningStep.l2_normalize(sample_features_small)
        assert out.shape == sample_features_small.shape


# --- PretrainedFeatureLearning ---

class TestPretrainedFeatureLearning:
    def test_default_params(self):
        fl = PretrainedFeatureLearning()
        assert fl.backbone == "resnet50"
        assert fl.normalize is True

    def test_run_output_shape(self, pretrained_fl):
        data = list(range(10))
        out = pretrained_fl.run(data)
        assert out.shape == (10, 512)

    @pytest.mark.xfail(
        reason="Stub returns zero vectors; real backbone implementation will produce unit norms."
    )
    def test_run_normalized_output_has_unit_norm(self, pretrained_fl, sample_features):
        data = list(range(len(sample_features)))
        out = pretrained_fl.run(data)
        norms = np.linalg.norm(out, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    def test_run_unnormalized_when_disabled(self):
        fl = PretrainedFeatureLearning(normalize=False)
        out = fl.run(list(range(5)))
        # stub returns zeros — norms are 0, not 1
        norms = np.linalg.norm(out, axis=1)
        np.testing.assert_allclose(norms, 0.0, atol=1e-6)

    def test_callable_alias(self, pretrained_fl):
        data = list(range(4))
        assert np.array_equal(pretrained_fl(data), pretrained_fl.run(data))


# --- VAEFeatureLearning ---

class TestVAEFeatureLearning:
    def test_default_params(self):
        fl = VAEFeatureLearning()
        assert fl.latent_dim == 128
        assert fl.epochs == 100
        assert fl.normalize is True

    def test_run_output_shape(self, vae_fl):
        data = list(range(7))
        out = vae_fl.run(data)
        assert out.shape == (7, 64)

    @pytest.mark.xfail(
        reason="Stub returns zero vectors; real VAE implementation will produce unit norms."
    )
    def test_run_normalized_output_has_unit_norm(self, vae_fl):
        data = list(range(6))
        out = vae_fl.run(data)
        norms = np.linalg.norm(out, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    def test_latent_dim_respected(self):
        fl = VAEFeatureLearning(latent_dim=32, normalize=False)
        out = fl.run(list(range(3)))
        assert out.shape[1] == 32

    def test_callable_alias(self, vae_fl):
        data = list(range(4))
        assert np.array_equal(vae_fl(data), vae_fl.run(data))
