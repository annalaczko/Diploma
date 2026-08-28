from pathlib import Path

from PIL import Image

from pipeline.preprocessing.repair import Repair


def test_collect_image_paths_accepts_list_of_paths(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()

    img_a = Image.new("L", (10, 10), 255)
    img_b = Image.new("L", (12, 12), 0)
    img_a.save(image_dir / "a.png")
    img_b.save(image_dir / "b.jpg")
    (image_dir / "notes.txt").write_text("ignore me", encoding="utf-8")

    repair = Repair(input_path=[str(image_dir), str(image_dir / "a.png")], output_path=str(tmp_path / "out"))

    paths = repair._collect_image_paths(repair.input_path)

    assert {p.name for p in paths} == {"a.png", "b.jpg"}
    assert all(p.exists() for p in paths)


def test_run_processes_existing_input_paths_only(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()

    input_file = source_dir / "sample.png"
    Image.new("L", (20, 20), 255).save(input_file)

    output_dir = tmp_path / "out"
    repair = Repair(input_path=[str(source_dir), str(tmp_path / "missing.png")], output_path=str(output_dir))

    result = repair.run()

    assert result.exists()
    assert (result / "sample.png").exists()
