from pathlib import Path

from PIL import Image

from pipeline.preprocessing.normalizer import Normalizer


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
