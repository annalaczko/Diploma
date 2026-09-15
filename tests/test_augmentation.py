from PIL import Image

from pipeline.preprocessing.augmentation import Augmentor


def test_augmentor_generates_requested_number_of_copies_per_image(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()

    Image.new("RGB", (40, 40), color=(10, 20, 30)).save(source_dir / "a.png")
    Image.new("RGB", (40, 40), color=(40, 50, 60)).save(source_dir / "b.png")

    output_dir = tmp_path / "out"
    augmentor = Augmentor(
        input_path=str(source_dir),
        output_path=str(output_dir),
        num_augmentations=3,
    )

    result = augmentor.run()

    assert result == output_dir
    # original + 3 augmented copies, for each of the 2 source images
    assert sorted(p.name for p in output_dir.iterdir()) == [
        "a.png", "a_aug0.png", "a_aug1.png", "a_aug2.png",
        "b.png", "b_aug0.png", "b_aug1.png", "b_aug2.png",
    ]


def test_augmentor_keep_original_false_omits_unaugmented_copy(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    Image.new("L", (30, 30), 255).save(source_dir / "sample.png")

    output_dir = tmp_path / "out"
    augmentor = Augmentor(
        input_path=str(source_dir),
        output_path=str(output_dir),
        num_augmentations=2,
        keep_original=False,
    )

    augmentor.run()

    output_names = {p.name for p in output_dir.iterdir()}
    assert output_names == {"sample_aug0.png", "sample_aug1.png"}


def test_augmentor_resizes_augmented_copies_to_output_size_and_preserves_mode(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    Image.new("L", (64, 48), 200).save(source_dir / "sample.png")

    output_dir = tmp_path / "out"
    augmentor = Augmentor(
        input_path=str(source_dir),
        output_path=str(output_dir),
        num_augmentations=1,
        output_size=32,
    )

    augmentor.run()

    # The random-resized-crop always produces a square output_size x
    # output_size image, regardless of the source image's own size/shape.
    augmented = Image.open(output_dir / "sample_aug0.png")
    assert augmented.size == (32, 32)
    assert augmented.mode == "L"

    # The kept, unaugmented original is left completely untouched.
    original = Image.open(output_dir / "sample.png")
    assert original.size == (64, 48)


def test_augmentor_zero_augmentations_with_keep_original_just_copies(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    Image.new("RGB", (20, 20), (1, 2, 3)).save(source_dir / "sample.png")

    output_dir = tmp_path / "out"
    augmentor = Augmentor(
        input_path=str(source_dir),
        output_path=str(output_dir),
        num_augmentations=0,
        keep_original=True,
    )

    augmentor.run()

    assert [p.name for p in output_dir.iterdir()] == ["sample.png"]


def test_augmentor_rejects_negative_num_augmentations(tmp_path):
    try:
        Augmentor(input_path=str(tmp_path), num_augmentations=-1)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for negative num_augmentations")


def test_augmentor_raises_when_no_images_found(tmp_path):
    source_dir = tmp_path / "empty"
    source_dir.mkdir()

    augmentor = Augmentor(input_path=str(source_dir), output_path=str(tmp_path / "out"))

    try:
        augmentor.run()
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("Expected FileNotFoundError when no images are found")


def test_augmentor_is_reproducible_with_seed(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    Image.new("L", (50, 50), 128).save(source_dir / "sample.png")

    out_a = tmp_path / "out_a"
    Augmentor(
        input_path=str(source_dir), output_path=str(out_a),
        num_augmentations=1, keep_original=False, seed=42,
    ).run()

    out_b = tmp_path / "out_b"
    Augmentor(
        input_path=str(source_dir), output_path=str(out_b),
        num_augmentations=1, keep_original=False, seed=42,
    ).run()

    img_a = Image.open(out_a / "sample_aug0.png")
    img_b = Image.open(out_b / "sample_aug0.png")
    assert img_a.tobytes() == img_b.tobytes()
