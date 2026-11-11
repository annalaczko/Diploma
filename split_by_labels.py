"""Split the images under a folder into test/ and train/ subfolders of that
same folder by the labels file: an image whose ID (file name without
extension, with its subfolders joined by "__", see `flatten_name`) is in the
labels file goes to test/, every other image to train/.

    python split_by_labels.py <folder> [--labels /data/alaczko/ocr_data.csv] [--copy] [--dry-run]

The subfolders below <folder> are kept (<folder>/a/b/001.jpg ->
<folder>/test/a/b/001.jpg), so running the pipeline on <folder>/test gives
the same flat names (and IDs) as running it on <folder>. Works for raw scans
(nested folders) and preprocessed images (flat "a__b__001.png" names) alike.
By default the files are hard links (no extra disk space, the originals stay);
--copy copies them instead.
"""

import argparse
import os
import shutil
from pathlib import Path

import pandas as pd

from pipeline.preprocessing.dataset import collect_image_paths, flatten_name

SPLITS = ("train", "test")


def split_by_labels(folder: str, labels_path: str = "/data/alaczko/ocr_data.csv", copy: bool = False,
                    dry_run: bool = False) -> dict[str, list[Path]]:
    """Put every image under `folder` into `folder/test` (its ID is in the
    `ID` column of `labels_path`) or `folder/train` (it isn't).

    Images already inside `folder/train` or `folder/test`, and side outputs
    in "_" folders (`_debug`, `_svg`), are left out. Files that already
    exist at the destination are skipped, so the script can be run again.

    Returns:
        {"train": [...], "test": [...]}: the source image paths of each split.
    """
    folder = Path(folder)
    ids = set(pd.read_csv(labels_path)["ID"].astype(str))

    images = [p for p in collect_image_paths(folder)
              if p.relative_to(folder).parts[0] not in SPLITS]
    print(f"Found {len(images)} images under {folder}, {len(ids)} labelled IDs in {labels_path}")

    splits = {split: [] for split in SPLITS}
    found_ids = set()
    for path in images:
        image_id = Path(flatten_name(path, folder)).stem
        if image_id in ids:
            found_ids.add(image_id)
            splits["test"].append(path)
        else:
            splits["train"].append(path)

    for split, paths in splits.items():
        print(f"{split.capitalize():5}: {len(paths)} -> {folder / split}")
    missing = sorted(ids - found_ids)
    if missing:
        print(f"{len(missing)} labelled IDs have no image under {folder}, e.g. {missing[:3]}")
    if dry_run:
        return splits

    for split, paths in splits.items():
        for path in paths:
            target = folder / split / path.relative_to(folder)
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if copy:
                shutil.copy2(path, target)
            else:
                os.link(path, target)
    return splits


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", help="folder searched (recursively) for images; train/ and test/ are created in it")
    parser.add_argument("--labels", default="/data/alaczko/ocr_data.csv",
                        help="CSV with an ID column (file name without extension); these images go to test/")
    parser.add_argument("--copy", action="store_true", help="copy the files instead of hard-linking them")
    parser.add_argument("--dry-run", action="store_true", help="only print the counts, write nothing")
    args = parser.parse_args()

    split_by_labels(args.folder, args.labels, args.copy, args.dry_run)
