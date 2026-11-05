"""Randomly split the images under a folder into train/, val/ and test/
subfolders of another folder (copies, the originals stay).

    python split_train_val.py <source folder> <target folder> [--val_ratio 0.2] [--test_ratio 0.03]
"""

import argparse
import random
import shutil
from pathlib import Path


def split_train_val_test(source_dir: str, final_dir: str, val_ratio: float = 0.2, test_ratio: float = 0.03, seed: int = 42, suffix: str = ".jpg"):
    """Shuffle the files with the given extension(s) under `source_dir`
    (searched recursively, case-insensitive) with `seed`, and copy them into
    `final_dir/train`, `final_dir/val` and `final_dir/test`.

    `suffix` is a list of extensions such as [".jpg", ".jpeg"] (the command
    line passes a list; a plain string would be read letter by letter).
    """
    source_dir = Path(source_dir)
    final_dir = Path(final_dir)

    files = [p for p in source_dir.rglob("*") if p.is_file() and p.suffix.lower() in [s.lower() for s in suffix]]
    print(f"Found {len(files)} {suffix} files under {source_dir}")

    random.seed(seed)
    random.shuffle(files)

    n_val = int(len(files) * val_ratio)
    n_test = int(len(files) * test_ratio)
    splits = {
        "train": files[n_val + n_test:],
        "val": files[:n_val],
        "test": files[n_val:n_val + n_test],
    }

    for split, split_files in splits.items():
        split_dir = final_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for f in split_files:
            shutil.copy2(f, split_dir / f.name)

    print(f"Train: {len(splits['train'])} -> {final_dir / 'train'}")
    print(f"Val:   {len(splits['val'])} -> {final_dir / 'val'}")
    print(f"Test:  {len(splits['test'])} -> {final_dir / 'test'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source_dir", help="folder searched (recursively) for the files")
    parser.add_argument("final_dir", help="folder the train/val/test subfolders are created in")
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--test_ratio", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--suffix", nargs="+", default=[".jpg"],
                        help="file extension(s) to take, e.g. .jpg .jpeg")
    args = parser.parse_args()

    split_train_val_test(args.source_dir, args.final_dir, args.val_ratio, args.test_ratio, args.seed, args.suffix)
