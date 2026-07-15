import argparse
import random
import shutil
from pathlib import Path

def split_train_val_test(source_dir: str, final_dir: str, val_ratio: float = 0.2, test_ratio: float = 0.03, seed: int = 42, suffix: str = ".jpg"):
    source_dir = Path(source_dir)
    final_dir = Path(final_dir)

    # all jpgs recursively (jpg and jpeg, case-insensitive)
    files = [p for p in source_dir.rglob("*") if p.is_file() and p.suffix.lower() in [s.lower() for s in suffix]]
    print(f"Found {len(files)} {suffix} files under {source_dir}")

    random.seed(seed)
    random.shuffle(files)

    n_val = int(len(files) * val_ratio)
    n_test = int(len(files) * test_ratio)

    val_files = files[:n_val]
    test_files = files[n_val:n_val + n_test]
    train_files = files[n_val + n_test:]

    # interim mappa a source_dir mellé (azaz a szülő mappában)
    train_dir = final_dir / "train"
    val_dir = final_dir / "val"
    test_dir = final_dir / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    for f in train_files:
        shutil.copy2(f, train_dir / f.name)
    for f in val_files:
        shutil.copy2(f, val_dir / f.name)
    for f in test_files:
        shutil.copy2(f, test_dir / f.name)

    print(f"Train: {len(train_files)} -> {train_dir}")
    print(f"Val:   {len(val_files)} -> {val_dir}")
    print(f"Test:  {len(test_files)} -> {test_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source_dir", help="A mappa, amely alatt a fileokat keresi")
    parser.add_argument("final_dir", help="A mappa, amelybe a fileokat helyezi")
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--test_ratio", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--suffix", nargs="+", default=[".jpg"], 
                     help="A kiterjesztés(ek), amiket keresünk (pl. .jpg .jpeg)")
    args = parser.parse_args()

    split_train_val_test(args.source_dir, args.final_dir, args.val_ratio, args.test_ratio, args.seed, args.suffix)