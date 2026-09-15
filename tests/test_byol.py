import os

import numpy as np
import torch
from PIL import Image

from pipeline.feature_learning.byol import BYOL

BATCH_SIZE = 4
IMG_SIZE = 256
LATENT_DIM = 32
HIDDEN_DIM = 64
PROJECTION_DIM = 16


def _random_images(dir_path, count=4, size=IMG_SIZE):
    for i in range(count):
        arr = (np.random.rand(size, size) * 255).astype("uint8")
        Image.fromarray(arr, mode="L").save(os.path.join(dir_path, f"img{i}.png"))


def _make_model(input_path, **overrides):
    kwargs = dict(
        input_path=str(input_path),
        k=LATENT_DIM,
        hidden_dim=HIDDEN_DIM,
        projection_dim=PROJECTION_DIM,
    )
    kwargs.update(overrides)
    return BYOL(**kwargs)


def test_target_network_starts_as_an_exact_copy_of_online(tmp_path):
    model = _make_model(tmp_path)

    for online_param, target_param in zip(
        model.online_encoder.parameters(), model.target_encoder.parameters()
    ):
        assert torch.equal(online_param, target_param)
    for online_param, target_param in zip(
        model.online_projector.parameters(), model.target_projector.parameters()
    ):
        assert torch.equal(online_param, target_param)


def test_target_network_parameters_are_frozen(tmp_path):
    model = _make_model(tmp_path)

    assert all(not p.requires_grad for p in model.target_encoder.parameters())
    assert all(not p.requires_grad for p in model.target_projector.parameters())
    assert all(p.requires_grad for p in model.online_encoder.parameters())


def test_forward_returns_a_finite_scalar_loss(tmp_path):
    model = _make_model(tmp_path)
    view1 = torch.rand(BATCH_SIZE, 1, IMG_SIZE, IMG_SIZE)
    view2 = torch.rand(BATCH_SIZE, 1, IMG_SIZE, IMG_SIZE)

    loss = model(view1, view2)

    assert loss.dim() == 0
    assert torch.isfinite(loss)
    # BYOL's per-sample loss is 2 - 2*cosine_similarity, summed over two
    # directions, so it's bounded in [0, 8].
    assert 0.0 <= loss.item() <= 8.0


def test_byol_loss_is_zero_for_identical_normalized_vectors():
    v = torch.tensor([[3.0, 4.0], [1.0, 0.0]])  # norms 5 and 1
    loss = BYOL._byol_loss(v, v)
    assert torch.allclose(loss, torch.zeros(2), atol=1e-6)


def test_byol_loss_is_maximal_for_opposite_vectors():
    pred = torch.tensor([[1.0, 0.0]])
    target = torch.tensor([[-1.0, 0.0]])
    loss = BYOL._byol_loss(pred, target)
    assert torch.allclose(loss, torch.tensor([4.0]), atol=1e-6)


def test_update_target_network_moves_target_toward_online(tmp_path):
    model = _make_model(tmp_path, tau=0.5)

    with torch.no_grad():
        for param in model.online_encoder.parameters():
            param.add_(1.0)  # push online away from target

    target_before = next(model.online_encoder.parameters()) is not next(
        model.target_encoder.parameters()
    )
    assert target_before  # sanity: they're distinct tensors, not aliases

    target_param_before = next(model.target_encoder.parameters()).clone()
    model._update_target_network()
    target_param_after = next(model.target_encoder.parameters())
    online_param = next(model.online_encoder.parameters())

    # With tau=0.5, the target should land exactly halfway between where
    # it was and the online network's current value.
    expected = 0.5 * target_param_before + 0.5 * online_param
    assert torch.allclose(target_param_after, expected, atol=1e-5)


def test_run_rejects_invalid_mode(tmp_path):
    model = _make_model(tmp_path)
    try:
        model.run(mode="not_a_real_mode")
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for an invalid mode")


def test_extract_features_returns_filename_and_feature_columns(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    _random_images(source_dir, count=3)

    output_dir = tmp_path / "out"
    model = _make_model(source_dir, output_path=str(output_dir))

    result = model.extract_features(batch_size=2)

    assert list(result.columns) == ["filename", "features"]
    assert len(result) == 3
    assert all(len(vec) == LATENT_DIM for vec in result["features"])
    assert (output_dir / "features.csv").exists()


def test_train_model_reduces_loss_over_a_few_epochs(tmp_path):
    # Reimplements train_model's loop directly (rather than calling it)
    # so per-epoch losses can be collected without parsing print output.
    from torchvision import transforms

    from pipeline.feature_learning.simclr import SimCLRDataset
    from pipeline.preprocessing.augmentation import build_augmentation_transform
    from pipeline.preprocessing.dataset import collect_image_paths

    source_dir = tmp_path / "source"
    source_dir.mkdir()
    _random_images(source_dir, count=8)

    torch.manual_seed(0)
    model = _make_model(source_dir, tau=0.9)

    view_transform = transforms.Compose([build_augmentation_transform(), transforms.ToTensor()])
    dataset = SimCLRDataset(collect_image_paths(str(source_dir)), view_transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, drop_last=True)
    optimizer = torch.optim.Adam(
        list(model.online_encoder.parameters())
        + list(model.online_projector.parameters())
        + list(model.online_predictor.parameters()),
        lr=1e-3,
    )

    epoch_losses = []
    model.train()
    for _ in range(5):
        total_loss = 0.0
        for view1, view2 in loader:
            optimizer.zero_grad()
            loss = model(view1, view2)
            loss.backward()
            optimizer.step()
            model._update_target_network()
            total_loss += loss.item()
        epoch_losses.append(total_loss / len(loader))

    # Should trend down over 5 epochs, even on tiny random data - allow
    # some noise by comparing the first vs. last epoch rather than
    # requiring strict monotonicity.
    assert epoch_losses[-1] < epoch_losses[0]
