"""Tests for the in-memory (batch) pipeline mode: every preprocessing
step's `process_batch` must match its disk-mode `run()`, and `Pipeline`
must give the feature-learning step the same images either way."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from pipeline.feature_learning.vae import VAE
from pipeline.pipeline import Pipeline
from pipeline.preprocessing import (
    Augmentor,
    BackgroundEraser,
    ImageDataset,
    ImageRecord,
    InMemoryImages,
    Normalizer,
    Repair,
    Vectorizer,
    collect_image_paths,
    iter_image_batches,
)


def _line_drawings(folder, count=4, size=(300, 200)):
    """Non-square RGB images with a few dark lines on white."""
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for i in range(count):
        arr = np.full((size[1], size[0], 3), 255, np.uint8)
        for _ in range(3):
            y = rng.integers(10, size[1] - 10)
            arr[y:y + 3, 20:size[0] - 20] = 0
        Image.fromarray(arr).save(folder / f"img_{i}.png")
    return folder


def _records(folder):
    return [ImageRecord(p.name, Image.open(p).copy()) for p in sorted(folder.glob("*.png"))]


def _disk_images(folder):
    return {p.name: np.asarray(Image.open(p)) for p in sorted(folder.glob("*.png"))}


def test_in_memory_images_requires_one_size():
    records = [
        ImageRecord("a.png", Image.new("L", (8, 8))),
        ImageRecord("b.png", Image.new("L", (9, 9))),
    ]
    with pytest.raises(ValueError, match="different sizes"):
        InMemoryImages.from_records(records)


def test_iter_image_batches_applies_exif_orientation(tmp_path):
    # Many scans are stored rotated with an EXIF orientation tag; disk mode's
    # cv2.imread applies it, so the in-memory loader must too.
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 degrees clockwise
    (tmp_path / "raw").mkdir()
    Image.new("L", (30, 10)).save(tmp_path / "raw" / "scan.jpg", exif=exif)

    [[record]] = list(iter_image_batches(str(tmp_path / "raw")))

    assert record.name == "scan.jpg"
    assert record.image.size == (10, 30)


def test_image_dataset_applies_transform_to_in_memory_images():
    images = InMemoryImages(["a.png", "b.png"], np.zeros((2, 16, 16), np.uint8))
    dataset = ImageDataset(images, target_size=None, transform=lambda img: img.point(lambda _: 255))

    assert dataset.names == ["a.png", "b.png"]
    item = dataset[0]
    assert item.shape == (1, 16, 16)
    assert torch.all(item == 1.0)


@pytest.mark.parametrize(
    "make_step",
    [
        lambda out: Normalizer(output_path=out, target_size=64, background_color=(255, 255, 255)),
        lambda out: Repair(output_path=out, skeletonize=False, output_thickness=2, debug_images=0),
        lambda out: Vectorizer(output_path=out, debug_images=0),
    ],
    ids=["Normalizer", "Repair", "Vectorizer"],
)
def test_process_batch_matches_disk_run(tmp_path, make_step):
    source = _line_drawings(tmp_path / "raw")
    step = make_step(str(tmp_path / "out"))
    step.input_path = str(source)
    step.run()
    on_disk = _disk_images(tmp_path / "out")

    in_memory = {r.name: np.asarray(r.image) for r in step.process_batch(_records(source))}

    assert in_memory.keys() == on_disk.keys()
    for name, array in on_disk.items():
        np.testing.assert_array_equal(in_memory[name], array)


def test_background_eraser_process_batch_with_mocked_yolo(tmp_path, monkeypatch):
    eraser = BackgroundEraser(mode="detection", box_margin=0.0, debug_images=0, run_id="test")
    found = SimpleNamespace(boxes=SimpleNamespace(xyxy=torch.tensor([[10.0, 20.0, 50.0, 40.0]])))
    nothing = SimpleNamespace(boxes=SimpleNamespace(xyxy=torch.zeros((0, 4))))
    model = SimpleNamespace(predict=lambda images, **kwargs: [found, nothing][:len(images)])
    monkeypatch.setattr(eraser, "_get_model", lambda: model)

    records = [
        ImageRecord("hit.png", Image.new("RGB", (100, 80), (0, 0, 0))),
        ImageRecord("miss.png", Image.new("RGB", (100, 80), (0, 0, 0))),
    ]
    out = eraser.process_batch(records)

    assert [r.name for r in out] == ["hit.png"]  # no detection -> dropped
    assert out[0].image.size == (40, 40)  # square crop around the 40x20 box
    assert out[0].image.mode == "RGB"


def _small_vae(tmp_path):
    torch.manual_seed(0)
    vae = VAE(output_path=str(tmp_path / "features"))
    vae.eval()
    return vae


def _steps(tmp_path):
    return [
        Normalizer(output_path=str(tmp_path / "normalized"), background_color=(255, 255, 255)),
        Repair(output_path=str(tmp_path / "repaired"), skeletonize=False, debug_images=0),
    ]


def test_pipeline_memory_and_disk_give_same_features(tmp_path):
    source = _line_drawings(tmp_path / "raw")
    vae = _small_vae(tmp_path)
    pipe = Pipeline(_steps(tmp_path), feature_learning_step=vae)

    disk = pipe.run(str(source), mode="disk")
    memory = pipe.run(str(source), mode="memory", preprocess_batch_size=3)

    assert len(memory["images"]) == 4
    assert list(memory["features"]["filename"]) == list(disk["features"]["filename"])
    np.testing.assert_allclose(
        np.stack(memory["features"]["features"]),
        np.stack(disk["features"]["features"]),
        rtol=1e-4, atol=1e-4,
    )


def test_pipeline_memory_mode_augments_on_the_fly(tmp_path):
    source = _line_drawings(tmp_path / "raw")
    augmentor = Augmentor(output_path=str(tmp_path / "augmented"), num_augmentations=2)
    vae = _small_vae(tmp_path)
    pipe = Pipeline([*_steps(tmp_path), augmentor], feature_learning_step=vae)

    result = pipe.run(str(source), mode="memory")

    # No augmented copies were made, the transform goes to training instead.
    assert len(result["images"]) == 4
    assert not list((tmp_path / "augmented").glob("*"))
    assert vae.train_transform is augmentor.transform

    disk = pipe.run(str(source), mode="disk")
    assert len(disk["features"]) == 4 * 3  # originals + 2 copies each
    assert vae.images is None


@pytest.mark.parametrize("save", ["last", "all"])
def test_memory_mode_saves_preprocessed_images_like_disk_mode(tmp_path, save):
    source = _line_drawings(tmp_path / "raw")
    disk_steps = [
        Normalizer(output_path=str(tmp_path / "disk_normalized"), background_color=(255, 255, 255)),
        Repair(output_path=str(tmp_path / "disk_repaired"), skeletonize=False, debug_images=0),
    ]
    Pipeline(disk_steps).run(str(source), mode="disk")

    result = Pipeline(_steps(tmp_path)).run(str(source), mode="memory", save_preprocessed=save)

    assert result["output_path"] == tmp_path / "repaired"
    assert _disk_images(tmp_path / "repaired").keys() == set(result["images"].names)
    for name, array in _disk_images(tmp_path / "disk_repaired").items():
        np.testing.assert_array_equal(_disk_images(tmp_path / "repaired")[name], array)
    # 'last' writes only the final step's folder, 'all' every step's.
    normalized = list((tmp_path / "normalized").glob("*.png")) if (tmp_path / "normalized").exists() else []
    assert len(normalized) == (4 if save == "all" else 0)


def test_memory_mode_saves_nothing_by_default(tmp_path):
    source = _line_drawings(tmp_path / "raw")

    result = Pipeline(_steps(tmp_path)).run(str(source), mode="memory")

    assert "output_path" not in result
    assert not list((tmp_path / "repaired").glob("*.png"))


def test_prefetch_keeps_order_and_raises_errors():
    from pipeline.preprocessing import prefetch

    assert list(prefetch(iter(range(50)), depth=3)) == list(range(50))

    def failing():
        yield 1
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        list(prefetch(failing()))


def test_collect_image_paths_skips_side_output_folders(tmp_path):
    for rel in ["a.png", "sub/b.png", "_debug/c.png", "sub/_svg/d.png"]:
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        Image.new("L", (4, 4)).save(tmp_path / rel)

    names = [p.relative_to(tmp_path).as_posix() for p in collect_image_paths(tmp_path)]

    assert names == ["a.png", "sub/b.png"]


def test_background_eraser_crop_size_is_fraction_of_page_width(monkeypatch):
    eraser = BackgroundEraser(mode="detection", box_margin=0.0, crop_size=0.5, debug_images=0, run_id="test")
    small = SimpleNamespace(boxes=SimpleNamespace(xyxy=torch.tensor([[10.0, 20.0, 50.0, 40.0]])))
    large = SimpleNamespace(boxes=SimpleNamespace(xyxy=torch.tensor([[0.0, 0.0, 90.0, 70.0]])))
    model = SimpleNamespace(predict=lambda images, **kwargs: [small, large][:len(images)])
    monkeypatch.setattr(eraser, "_get_model", lambda: model)

    records = [
        ImageRecord("small.png", Image.new("RGB", (100, 80), (0, 0, 0))),
        ImageRecord("large.png", Image.new("RGB", (100, 80), (0, 0, 0))),
    ]
    out = eraser.process_batch(records)

    assert [r.image.size for r in out] == [(50, 50), (50, 50)]  # half of the 100 px page width
    assert eraser.clipped == ["large.png"]  # the 90x70 box doesn't fit
    assert eraser.failure_lists()["clipped"] == ["large.png"]
