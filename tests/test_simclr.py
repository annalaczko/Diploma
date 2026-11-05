import os

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from pipeline.feature_learning.simclr import SimCLR
from pipeline.preprocessing.augmentation import Augmentor

BATCH_SIZE = 4
IMG_SIZE = 256
LATENT_DIM = 32
PROJECTION_DIM = 16


def _random_images(dir_path, count=4, size=IMG_SIZE):
    for i in range(count):
        arr = (np.random.rand(size, size) * 255).astype("uint8")
        Image.fromarray(arr, mode="L").save(os.path.join(dir_path, f"img{i}.png"))


def test_forward_returns_embedding_and_normalized_projection(tmp_path):
    model = SimCLR(input_path=str(tmp_path), k=LATENT_DIM, projection_dim=PROJECTION_DIM)
    inputs = torch.rand(BATCH_SIZE, 1, IMG_SIZE, IMG_SIZE)

    h, z = model(inputs)

    assert h.shape == (BATCH_SIZE, LATENT_DIM)
    assert z.shape == (BATCH_SIZE, PROJECTION_DIM)
    # z is L2-normalized inside forward(), so every row's norm should be 1.
    norms = z.norm(dim=1)
    assert torch.allclose(norms, torch.ones(BATCH_SIZE), atol=1e-5)


def test_nt_xent_loss_prefers_well_separated_positive_pairs():
    # Two "images", each with two identical views -> a trivially perfect
    # positive alignment with a clearly separated negative.
    aligned = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ])
    z1_aligned, z2_aligned = aligned[:2], aligned[2:]

    # Same four vectors, but shuffled so positives no longer match up -
    # the loss should be worse (higher) than the well-aligned case.
    misaligned = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 1.0],
        [1.0, 0.0],
    ])
    z1_misaligned, z2_misaligned = misaligned[:2], misaligned[2:]

    loss_aligned = SimCLR._nt_xent_loss(z1_aligned, z2_aligned, temperature=0.5)
    loss_misaligned = SimCLR._nt_xent_loss(z1_misaligned, z2_misaligned, temperature=0.5)

    assert loss_aligned.item() < loss_misaligned.item()


def test_run_rejects_invalid_mode(tmp_path):
    model = SimCLR(input_path=str(tmp_path))
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
    model = SimCLR(
        input_path=str(source_dir),
        output_path=str(output_dir),
        k=LATENT_DIM,
        projection_dim=PROJECTION_DIM,
    )

    result = model.extract_features(batch_size=2)

    assert list(result.columns) == ["filename", "features"]
    assert len(result) == 3
    assert all(len(vec) == LATENT_DIM for vec in result["features"])
    assert (output_dir / "features.csv").exists()


def test_train_model_accepts_an_existing_augmentors_transform(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    _random_images(source_dir, count=4)

    # Built only for its already-tuned .transform - never call .run() on it.
    augmentor = Augmentor(
        input_path=str(source_dir),
        output_path=str(tmp_path / "augmentor_out"),
        crop_scale=(0.7, 1.0),
    )
    view_transform = transforms.Compose([augmentor.transform, transforms.ToTensor()])

    model = SimCLR(
        input_path=str(source_dir),
        output_path=str(tmp_path / "out"),
        k=LATENT_DIM,
        projection_dim=PROJECTION_DIM,
    )

    result = model.train_model(
        str(source_dir), epochs=1, batch_size=2, view_transform=view_transform
    )

    assert list(result.columns) == ["filename", "features"]
    assert len(result) == 4
