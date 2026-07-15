import argparse
from pathlib import Path
import shutil

def mirror_split(images_split_dir: str, labels_source_dir: str, labels_final_dir: str, label_suffix: str = ".txt"):
    images_split_dir = Path(images_split_dir)
    labels_source_dir = Path(labels_source_dir)
    labels_final_dir = Path(labels_final_dir)

    # label fájlok gyors kereséséhez: stem -> path lookup
    label_lookup = {p.stem: p for p in labels_source_dir.rglob(f"*{label_suffix}") if p.is_file()}
    print(f"Found {len(label_lookup)} label files under {labels_source_dir}")

    for split in ["train", "val", "test"]:
        img_dir = images_split_dir / split
        if not img_dir.exists():
            continue

        out_dir = labels_final_dir / split
        out_dir.mkdir(parents=True, exist_ok=True)

        matched, missing = 0, []
        for img_path in img_dir.iterdir():
            if not img_path.is_file():
                continue
            stem = img_path.stem
            label_path = label_lookup.get(stem)
            if label_path is None:
                missing.append(stem)
                continue
            shutil.copy2(label_path, out_dir / label_path.name)
            matched += 1

        print(f"{split}: {matched} labels matched, {len(missing)} missing")
        if missing:
            print(f"  Hiányzó labelek (első 10): {missing[:10]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("images_split_dir", help="A mappa, ami már tartalmazza a train/val/test almappákat képekkel")
    parser.add_argument("labels_source_dir", help="A mappa, ahol az összes label fájl van (rekurzívan keresi)")
    parser.add_argument("labels_final_dir", help="Ide kerül a train/val/test label struktúra")
    parser.add_argument("--label_suffix", type=str, default=".txt")
    args = parser.parse_args()

    mirror_split(args.images_split_dir, args.labels_source_dir, args.labels_final_dir, args.label_suffix)