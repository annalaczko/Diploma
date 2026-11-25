"""Which labels ended up in which cluster, and the images sorted into one folder per cluster,
for looking at a run's clusters by eye:

    python cluster_folders.py data/pipeline/<run id> [<run id> ...]
    python cluster_folders.py data/pipeline/<run id> --images /data/alaczko/centuria/images   # the original scans

For each run it writes, next to its clusters.csv (ClusteringStep/HierarchicalClustering/):
  crosstab.csv        clusters x labels (image counts), with each cluster's size, most frequent
                      label and purity (share of that label); also printed
  by_cluster/cluster_<k>/<label>__<file>   a copy of every image of cluster k; the label first,
                      so a folder sorted by name groups the labels
  by_cluster/cluster_<k>.jpg               contact sheet of the cluster (label under each image)

The labels come from the run's data.labels (ID = file name without extension, TypeRev).
The images from --images, or else the folder the run's features were extracted from: for
the Centuria runs the data.preprocessed_path of configs/centuria/models/CENT-<model>.yaml.
"""

import argparse
import math
import shutil
from pathlib import Path

import pandas as pd
import yaml
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parent
CENTURIA_MODELS = REPO / "configs/centuria/models"
THUMB = 160      # contact sheet: thumbnail side (px)
COLUMNS = 10     # contact sheet: thumbnails per row


def resolve(path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO / path


def image_folder(run_dir: Path, config: dict) -> Path:
    """The folder the run's features were extracted from."""
    name = run_dir.name.split("__")[0]
    centuria = CENTURIA_MODELS / f"{name.replace('CENT-top5-', 'CENT-', 1)}.yaml"
    if centuria.is_file():
        return resolve(yaml.safe_load(centuria.read_text())["data"]["preprocessed_path"])
    features = config["data"].get("features")
    if features:  # an earlier run's features: that run's config says where they come from
        source = resolve(features).parents[2] / "config.yaml"
        if not source.is_file():
            raise SystemExit(f"{run_dir}: cannot tell where the images are, pass --images")
        config = yaml.safe_load(source.read_text())
    data = config["data"]
    return resolve(data.get("extract_path") or data["preprocessed_path"])


def contact_sheet(images: list[tuple[Path, str]], target: Path) -> None:
    rows = math.ceil(len(images) / COLUMNS)
    sheet = Image.new("RGB", (COLUMNS * THUMB, rows * (THUMB + 14)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (path, label) in enumerate(images):
        x, y = index % COLUMNS * THUMB, index // COLUMNS * (THUMB + 14)
        with Image.open(path) as image:
            image = image.convert("RGB")
            image.thumbnail((THUMB - 4, THUMB - 4))
            sheet.paste(image, (x + 2, y + 2))
        draw.text((x + 4, y + THUMB), label[:24], fill="black")
    sheet.save(target, quality=90)


def report(run_dir: Path, images_override: Path | None) -> None:
    config = yaml.safe_load((run_dir / "config.yaml").read_text())
    clustering_dir = run_dir / "ClusteringStep/HierarchicalClustering"
    clusters = pd.read_csv(clustering_dir / "clusters.csv")
    labels_path = config["data"].get("labels")
    if labels_path:
        truth = pd.read_csv(resolve(labels_path), dtype={"ID": str})
        label_of = dict(zip(truth["ID"].str.strip(), truth["TypeRev"]))
    else:
        label_of = {}
    stems = clusters["filename"].astype(str).str.rsplit(".", n=1).str[0].str.replace(r"_aug\d+$", "", regex=True)
    clusters["label"] = stems.map(label_of).fillna("unlabelled")

    table = pd.crosstab(clusters["cluster"], clusters["label"])
    table.insert(0, "size", table.sum(axis=1))
    counts = table.drop(columns="size")
    table["top_label"] = counts.idxmax(axis=1)
    table["purity"] = (counts.max(axis=1) / table["size"]).round(3)
    table.to_csv(clustering_dir / "crosstab.csv")
    print(f"\n=== {run_dir.name}")
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(table.to_string())

    images = images_override or image_folder(run_dir, config)
    out = clustering_dir / "by_cluster"
    if out.exists():
        shutil.rmtree(out)
    missing = 0
    for cluster, group in clusters.groupby("cluster"):
        folder = out / f"cluster_{cluster}"
        folder.mkdir(parents=True)
        sheet = []
        for row in group.sort_values(["label", "filename"]).itertuples():
            source = images / row.filename
            if not source.is_file():  # e.g. --images with the scans: same stem, other extension
                matches = list(images.glob(f"{Path(row.filename).stem}.*"))
                if not matches:
                    missing += 1
                    continue
                source = matches[0]
            shutil.copy2(source, folder / f"{row.label}__{source.name}")
            sheet.append((source, row.label))
        if sheet:
            contact_sheet(sheet, out / f"cluster_{cluster}.jpg")
    print(f"  images from {images}" + (f" ({missing} not found)" if missing else ""))
    print(f"  {clustering_dir / 'crosstab.csv'}\n  {out}/")


def main() -> None:
    parser = argparse.ArgumentParser(description="Labels per cluster, and the images in one folder per cluster")
    parser.add_argument("runs", nargs="+", help="run folders, data/pipeline/<run id>")
    parser.add_argument("--images", type=Path, default=None,
                        help="folder with the images (default: where the run's features come from)")
    args = parser.parse_args()
    for run in args.runs:
        report(resolve(run), args.images)


if __name__ == "__main__":
    main()
