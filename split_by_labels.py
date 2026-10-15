"""Split a folder of preprocessed images into train/ and test/ by a labels
file: an image goes to test/ if its file name without the extension
(e.g. "a__007" for "a__007.jpg") is in the labels file's ID column, every
other image to train/. The IDs are used as they are.

    python split_by_labels.py data/processed/<run id>
    python split_by_labels.py data/processed/<run id> --labels data/other_labels.csv --id-column ID
    python split_by_labels.py data/processed/<run id> --copy --out data/processed/legion/no_axes

By default the images are moved into <folder>/train and <folder>/test, so the
folder ends up as the layout run_experiment.py expects
(data.preprocessed_path: <folder>/train). Only images directly in the folder
are split, so running it again does not touch train/ and test/.
"""

import argparse
import shutil
import unicodedata
from pathlib import Path

import pandas as pd

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def _key(name: str) -> str:
    # Accented names (e.g. "Künette") can be stored in two Unicode forms.
    return unicodedata.normalize("NFC", name.strip())


def split_by_labels(folder, labels="data/ocr_data.csv", out=None, copy=False, id_column="ID"):
    folder = Path(folder)
    out = Path(out) if out else folder
    table = pd.read_csv(labels)
    if id_column not in table:
        raise SystemExit(f"{labels} has no column {id_column!r} (columns: {list(table.columns)})")
    test_ids = {_key(str(i)) for i in table[id_column].dropna()}

    images = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise SystemExit(f"No images directly in {folder}")
    train_dir, test_dir = out / "train", out / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    transfer = shutil.copy2 if copy else shutil.move
    n_test = 0
    for image in images:
        is_test = _key(image.stem) in test_ids  # stem: the file name without its extension
        transfer(str(image), str((test_dir if is_test else train_dir) / image.name))
        n_test += is_test

    found = {_key(p.stem) for p in images}
    missing = sorted(test_ids - found)
    print(f"{'Copied' if copy else 'Moved'} {len(images)} images from {folder}")
    print(f"  train: {len(images) - n_test} -> {train_dir}")
    print(f"  test:  {n_test} -> {test_dir}")
    if missing:
        print(f"  {len(missing)} labelled IDs have no image in {folder} (e.g. {missing[0]})")


def main():
    parser = argparse.ArgumentParser(description="Split images into train/test by the OCR labels")
    parser.add_argument("folder", help="folder with the images (only files directly in it)")
    parser.add_argument("--labels", default="data/ocr_data.csv", help="labels CSV (default: %(default)s)")
    parser.add_argument("--id-column", default="ID",
                        help="column with the image names without extension (default: %(default)s)")
    parser.add_argument("--out", default=None, help="where train/ and test/ are created (default: the folder itself)")
    parser.add_argument("--copy", action="store_true", help="copy instead of move, keeping the originals")
    args = parser.parse_args()
    split_by_labels(args.folder, args.labels, args.out, args.copy, args.id_column)


if __name__ == "__main__":
    main()
