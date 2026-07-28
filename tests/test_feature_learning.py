"""
Shape teszt suite a VAE feature learning stephez (Encoder / Decoder / VAE).

Cél: bebiztositani, hogy a spatial meretek (256 -> 32 az encoderben,
32 -> 256 a decoderben) es a csatornaszamok pontosan ugy alakulnak,
ahogy a ConvTranspose2d / Conv2d formulak alapjan varjuk.

Futtatas:
    pytest test_feature_learning.py -v

FONTOS: igazitsd az alabbi import sort a sajat projekt struktrurajahoz
(pl. ha a VAE osztaly masik fajlban van, mint amit korabban kuldtel).
"""
import pytest
import torch

from pipeline.feature_learning.steps import VAE, Encoder, Decoder

BATCH_SIZE = 4
IMG_SIZE = 256
LATENT_DIM = 128


@pytest.fixture
def sample_batch():
    """Egy batch veletlen grayscale 256x256 kep [0,1] tartomanyban (BCE loss elvarasa)."""
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
# ENCODER: (N, 1, 256, 256) -> mu, sigma mindketto (N, 128)
# ---------------------------------------------------------------------------
class TestEncoder:
    def test_output_shape(self, encoder, sample_batch):
        mu, sigma = encoder(sample_batch)
        assert mu.shape == (BATCH_SIZE, LATENT_DIM)
        assert sigma.shape == (BATCH_SIZE, LATENT_DIM)

    def test_intermediate_spatial_downsampling(self, encoder, sample_batch):
        """
        A spatial meretnek blokkonkent feleznie kell:
        256 (conv0, same) -> 128 (block1) -> 64 (block2) -> 32 (block3).
        Ez adja ki a flatten_size = 128 * 32 * 32 = 131072 erteket a kodban.
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
        assert activations["conv3_1"][1] == 128  # csatorna: 2**(f+3)


# ---------------------------------------------------------------------------
# DECODER: (N, 128, 32, 32) -> (N, 1, 256, 256)
# ---------------------------------------------------------------------------
class TestDecoder:
    def test_output_shape_from_reshaped_latent(self, decoder):
        """
        decoder.forward() mar a reshape-elt (N,128,32,32) tenzort varja --
        az fc + view lepes a VAE.forward()-ban tortenik, nem itt.
        """
        z_spatial = torch.randn(BATCH_SIZE, LATENT_DIM, 32, 32)
        out = decoder(z_spatial)
        assert out.shape == (BATCH_SIZE, 1, IMG_SIZE, IMG_SIZE)

    def test_fc_layer_reshapes_latent_correctly(self, decoder):
        """decoder.fc: 128-vektorbol pontosan 128*32*32 elem, ami (N,128,32,32)-re view-olhato."""
        z = torch.randn(BATCH_SIZE, LATENT_DIM)
        z_flat = decoder.fc(z)
        assert z_flat.shape == (BATCH_SIZE, 128 * 32 * 32)

        z_spatial = z_flat.view(-1, 128, 32, 32)
        assert z_spatial.shape == (BATCH_SIZE, 128, 32, 32)

    def test_intermediate_spatial_upsampling(self, decoder):
        """
        A spatial meretnek blokkonkent duplazodnia kell:
        32 -> 64 (block1) -> 128 (block2) -> 256 (block3).
        A stride=1 retegek (conv0, conv*_0, conv4) "same" konvolucioként
        nem valtoztatjak a meretet -- ezt is ellenorizzuk.
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
# VAE: teljes end-to-end kor (encoder -> reparam trick -> decoder)
# ---------------------------------------------------------------------------
class TestVAEEndToEnd:
    def test_reconstruction_shape_matches_input(self, vae, sample_batch):
        """A legfontosabb teszt: recon merete pontosan egyezzen a bemenettel (BCE loss miatt kritikus)."""
        recon, mu, sigma = vae(sample_batch)
        assert recon.shape == sample_batch.shape
        assert mu.shape == (BATCH_SIZE, LATENT_DIM)
        assert sigma.shape == (BATCH_SIZE, LATENT_DIM)

    @pytest.mark.parametrize("batch_size", [1, 2, 8])
    def test_variable_batch_sizes(self, vae, batch_size):
        """Batch-meret fuggetlenseg: a modellnek barmilyen batch_size-ra mukodnie kell."""
        x = torch.rand(batch_size, 1, IMG_SIZE, IMG_SIZE)
        recon, mu, sigma = vae(x)
        assert recon.shape == (batch_size, 1, IMG_SIZE, IMG_SIZE)
        assert mu.shape == (batch_size, LATENT_DIM)
        assert sigma.shape == (batch_size, LATENT_DIM)

    def test_output_values_in_valid_range(self, vae, sample_batch):
        """
        Mivel a loss BCE (binary_cross_entropy), a decoder kimenetenek
        [0,1] tartomanyba kellene esnie -- de a jelenlegi Decoder.forward()
        vegen NINCS Sigmoid a conv4 utan! Ez a teszt varhatoan elbukik,
        amig ez hianyzik -- ez a celja: figyelmeztetni ra.
        """
        recon, _, _ = vae(sample_batch)
        assert recon.min() >= 0.0 and recon.max() <= 1.0, (
            "A decoder kimenete nincs [0,1] tartomanyba szoritva -- "
            "hianyzik egy Sigmoid a conv4 utan, ami a BCE losshoz kell."
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])