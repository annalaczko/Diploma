"""Shape tests of the VAE (Encoder / Decoder / VAE): the spatial sizes
(256 -> 32 in the encoder, 32 -> 256 in the decoder) and the channel counts
come out as the Conv2d / ConvTranspose2d formulas predict.

    pytest tests/test_feature_learning.py -v
"""
import numpy as np
import pytest
import torch
from PIL import Image

from pipeline.feature_learning.vae import VAE, Encoder, Decoder

BATCH_SIZE = 4
IMG_SIZE = 256
LATENT_DIM = 128


@pytest.fixture
def sample_batch():
    """A batch of random grayscale 256x256 images in [0, 1] (what BCE expects)."""
    return torch.rand(BATCH_SIZE, 1, IMG_SIZE, IMG_SIZE)


@pytest.fixture
def encoder():
    return Encoder()


@pytest.fixture
def decoder():
    return Decoder()


@pytest.fixture
def vae():
    return VAE()


# ---------------------------------------------------------------------------
# ENCODER: (N, 1, 256, 256) -> mu, sigma, both (N, 128)
# ---------------------------------------------------------------------------
class TestEncoder:
    def test_output_shape(self, encoder, sample_batch):
        mu, sigma = encoder(sample_batch)
        assert mu.shape == (BATCH_SIZE, LATENT_DIM)
        assert sigma.shape == (BATCH_SIZE, LATENT_DIM)

    def test_intermediate_spatial_downsampling(self, encoder, sample_batch):
        """
        The spatial size halves per block:
        256 (conv0, same) -> 128 (block1) -> 64 (block2) -> 32 (block3).
        That gives flatten_size = 128 * 32 * 32 = 131072 in the code.
        """
        activations = {}

        def make_hook(name):
            def hook(module, inp, out):
                activations[name] = tuple(out.shape)
            return hook

        encoder.conv0.register_forward_hook(make_hook("conv0"))
        encoder.conv1_1.register_forward_hook(make_hook("conv1_1"))
        encoder.conv2_1.register_forward_hook(make_hook("conv2_1"))
        encoder.conv3_1.register_forward_hook(make_hook("conv3_1"))

        encoder(sample_batch)

        assert activations["conv0"][-2:] == (256, 256)
        assert activations["conv1_1"][-2:] == (128, 128)
        assert activations["conv2_1"][-2:] == (64, 64)
        assert activations["conv3_1"][-2:] == (32, 32)
        assert activations["conv3_1"][1] == 128  # channels: 2**(f+3)


# ---------------------------------------------------------------------------
# DECODER: (N, 128, 32, 32) -> (N, 1, 256, 256)
# ---------------------------------------------------------------------------
class TestDecoder:
    def test_output_shape_from_reshaped_latent(self, decoder):
        """
        decoder.forward() expects the already reshaped (N,128,32,32) tensor:
        the fc + view step happens in VAE.forward(), not here.
        """
        z_spatial = torch.randn(BATCH_SIZE, LATENT_DIM, 32, 32)
        out = decoder(z_spatial)
        assert out.shape == (BATCH_SIZE, 1, IMG_SIZE, IMG_SIZE)

    def test_fc_layer_reshapes_latent_correctly(self, decoder):
        """decoder.fc: a 128-vector becomes exactly 128*32*32 values, viewable as (N,128,32,32)."""
        z = torch.randn(BATCH_SIZE, LATENT_DIM)
        z_flat = decoder.fc(z)
        assert z_flat.shape == (BATCH_SIZE, 128 * 32 * 32)

        z_spatial = z_flat.view(-1, 128, 32, 32)
        assert z_spatial.shape == (BATCH_SIZE, 128, 32, 32)

    def test_intermediate_spatial_upsampling(self, decoder):
        """
        The spatial size doubles per block:
        32 -> 64 (block1) -> 128 (block2) -> 256 (block3).
        The stride-1 layers (conv0, conv*_0, conv4) keep the size ("same").
        """
        activations = {}

        def make_hook(name):
            def hook(module, inp, out):
                activations[name] = tuple(out.shape)
            return hook

        decoder.conv0.register_forward_hook(make_hook("conv0"))
        decoder.conv1_1.register_forward_hook(make_hook("conv1_1"))
        decoder.conv2_1.register_forward_hook(make_hook("conv2_1"))
        decoder.conv3_1.register_forward_hook(make_hook("conv3_1"))
        decoder.conv4.register_forward_hook(make_hook("conv4"))

        z_spatial = torch.randn(BATCH_SIZE, LATENT_DIM, 32, 32)
        decoder(z_spatial)

        assert activations["conv0"][-2:] == (32, 32)      # same
        assert activations["conv1_1"][-2:] == (64, 64)     # 2x
        assert activations["conv2_1"][-2:] == (128, 128)   # 2x
        assert activations["conv3_1"][-2:] == (256, 256)   # 2x
        assert activations["conv4"][-2:] == (256, 256)     # same
        assert activations["conv4"][1] == 1                # grayscale kimenet


# ---------------------------------------------------------------------------
# VAE: the whole forward pass (encoder -> reparameterization -> decoder)
# ---------------------------------------------------------------------------
class TestVAEEndToEnd:
    def test_reconstruction_shape_matches_input(self, vae, sample_batch):
        """The reconstruction has exactly the input's shape (the BCE loss needs that)."""
        recon, mu, sigma = vae(sample_batch)
        assert recon.shape == sample_batch.shape
        assert mu.shape == (BATCH_SIZE, LATENT_DIM)
        assert sigma.shape == (BATCH_SIZE, LATENT_DIM)

    @pytest.mark.parametrize("batch_size", [1, 2, 8])
    def test_variable_batch_sizes(self, vae, batch_size):
        """The model works with any batch size."""
        x = torch.rand(batch_size, 1, IMG_SIZE, IMG_SIZE)
        recon, mu, sigma = vae(x)
        assert recon.shape == (batch_size, 1, IMG_SIZE, IMG_SIZE)
        assert mu.shape == (batch_size, LATENT_DIM)
        assert sigma.shape == (batch_size, LATENT_DIM)

    def test_output_values_in_valid_range(self, vae, sample_batch):
        """The loss is BCE, so the decoder output must lie in [0, 1] (the
        sigmoid at the end of Decoder.forward)."""
        recon, _, _ = vae(sample_batch)
        assert recon.min() >= 0.0 and recon.max() <= 1.0, (
            "The decoder output is not in [0, 1]: the sigmoid after conv4, "
            "needed by the BCE loss, is missing."
        )


class TestFeatureExport:
    def test_extract_features_saves_filename_and_vector_csv(self, tmp_path):
        image_dir = tmp_path / "images"
        image_dir.mkdir()

        for i in range(2):
            img = Image.new("L", (256, 256), color=255 * (i + 1) % 256)
            img.save(image_dir / f"sample_{i}.png")

        vae = VAE(input_path=str(image_dir), output_path=str(tmp_path / "out"))
        out = vae.extract_features(batch_size=1)

        assert list(out["filename"]) == ["sample_0.png", "sample_1.png"]
        assert isinstance(out["features"][0], np.ndarray)
        assert out["features"][0].shape == (LATENT_DIM,)

        csv_lines = (tmp_path / "out" / "features.csv").read_text(encoding="utf-8").splitlines()
        assert csv_lines[0].startswith("filename,")
        assert "sample_0.png" in csv_lines[1]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])