import numpy as np
import pytest
from PIL import Image

from pipeline.preprocessing.background_eraser import BackgroundEraser
from pipeline.preprocessing.normalizer import Normalizer
from pipeline.preprocessing.dataset import collect_image_paths


def test_background_eraser_expand_box_pads_by_the_given_fraction_on_each_side():
    box = np.array([100.0, 100.0, 200.0, 300.0])  # 100 wide, 200 tall

    expanded = BackgroundEraser._expand_box(box, image_width=1000, image_height=1000, margin=0.15)

    assert expanded == pytest.approx([85.0, 70.0, 215.0, 330.0])


def test_background_eraser_expand_box_clamps_to_image_bounds():
    box = np.array([5.0, 5.0, 50.0, 50.0])

    expanded = BackgroundEraser._expand_box(box, image_width=40, image_height=40, margin=0.5)

    assert expanded[0] == 0.0
    assert expanded[1] == 0.0
    assert expanded[2] == 40.0
    assert expanded[3] == 40.0


def test_background_eraser_collects_images_recursively_from_directory(tmp_path):
    source_dir = tmp_path / "source"
    nested_dir = source_dir / "nested"
    nested_dir.mkdir(parents=True)

    img_a = Image.new("RGB", (40, 40), color=(10, 20, 30))
    img_b = Image.new("RGB", (50, 50), color=(40, 50, 60))
    img_a.save(source_dir / "a.png")
    img_b.save(nested_dir / "b.png")

    eraser = BackgroundEraser(input_path=str(source_dir), output_path=str(tmp_path / "out"))
    paths = collect_image_paths(eraser.input_path, eraser.image_extensions)

    assert {p.name for p in paths} == {"a.png", "b.png"}
    assert all(p.exists() for p in paths)


def test_background_eraser_flattened_names_avoid_collisions_across_subfolders(tmp_path):
    source_dir = tmp_path / "source"
    survey_a = source_dir / "2012_Survey"
    survey_b = source_dir / "2016_Survey"
    survey_a.mkdir(parents=True)
    survey_b.mkdir(parents=True)

    # Different surveys reusing the same simple, sequential filename - the
    # exact real-world situation that used to make one silently overwrite
    # the other in a flat output directory.
    path_a = survey_a / "006.jpg"
    path_b = survey_b / "006.jpg"
    path_a.touch()
    path_b.touch()

    eraser = BackgroundEraser(input_path=str(source_dir), output_path=str(tmp_path / "out"))

    name_a = eraser._flatten_output_name(path_a)
    name_b = eraser._flatten_output_name(path_b)

    assert name_a != name_b
    assert name_a == "2012_Survey__006.jpg"
    assert name_b == "2016_Survey__006.jpg"


def test_background_eraser_flattened_name_includes_every_ancestor_folder(tmp_path):
    source_dir = tmp_path / "source"
    nested = source_dir / "00_Selection" / "2012_Survey" / "subfolder" / "006.jpg"
    nested.parent.mkdir(parents=True)
    nested.touch()

    eraser = BackgroundEraser(input_path=str(source_dir), output_path=str(tmp_path / "out"))

    assert eraser._flatten_output_name(nested) == "00_Selection__2012_Survey__subfolder__006.jpg"


def test_background_eraser_flattened_name_falls_back_to_basename_outside_input_path(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    outside_dir = tmp_path / "elsewhere"
    outside_dir.mkdir()
    stray_path = outside_dir / "006.jpg"
    stray_path.touch()

    eraser = BackgroundEraser(input_path=str(source_dir), output_path=str(tmp_path / "out"))

    assert eraser._flatten_output_name(stray_path) == "006.jpg"


def test_normalizer_pads_and_resizes_images(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()

    img = Image.new("RGB", (120, 80), color=(255, 0, 0))
    img.save(source_dir / "sample.png")

    output_dir = tmp_path / "out"
    normalizer = Normalizer(input_path=str(source_dir), output_path=str(output_dir))

    result = normalizer.run()

    processed = output_dir / "sample.png"
    assert result == output_dir
    assert processed.exists()

    resized = Image.open(processed)
    assert resized.size == (256, 256)
    assert resized.mode == "RGB"


def test_normalizer_accepts_list_of_paths(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()

    img_a = Image.new("RGB", (50, 50), color=(10, 20, 30))
    img_b = Image.new("RGB", (200, 100), color=(40, 50, 60))
    img_a.save(source_dir / "a.png")
    img_b.save(source_dir / "b.png")

    output_dir = tmp_path / "out"
    normalizer = Normalizer(input_path=[str(source_dir / "a.png"), str(source_dir / "b.png")], output_path=str(output_dir))

    result = normalizer.run()

    assert result == output_dir
    assert (output_dir / "a.png").exists()
    assert (output_dir / "b.png").exists()
