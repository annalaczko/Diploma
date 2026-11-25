"""Evaluate the existing models on the Centuria test data in one go:

    python run_centuria.py --gpus 0,1,2            # everything
    python run_centuria.py --only 'CENT-DINO-P-N*'  # some of the models
    python run_centuria.py --dry-run               # only check the configs
    python run_centuria.py --configs configs/centuria/top5 --skip-prep   # top5 (configs/centuria/make_top5.py)

1. Data. /data/alaczko/centuria/centuria.csv has the labels as ID (file name
   with extension) and Class; the evaluation reads ID without extension and
   TypeRev, so they are written to centuria_labels.csv next to it. The scans
   that have a Class are hard-linked into images_labelled/ (the 34 without
   one are left out, so the 9 clusters are made from labelled images only).
2. Preprocessing. Every configs/centuria/prep/<variant>.yaml (the
   preprocessings the models were trained on) is run once, and its images
   are copied to /data/alaczko/processed/centuria/<variant>/. A variant
   whose folder already exists is not run again (--redo-prep to force).
3. Models. Every configs/centuria/models/*.yaml (an existing model.pt, or a
   baseline that trains nothing) is run with run_experiment.py, one at a time
   per GPU in --gpus, each as its own process. MLflow experiment:
   centuria-eval. Logs: data/pipeline/centuria__<time>/logs/<config>.log.
4. Summary. The summary.csv of every sweep run in step 3, together in
   data/pipeline/centuria__<time>/summary.csv, best ARI first.
"""

import argparse
import fnmatch
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent
CENTURIA = Path("/data/alaczko/centuria")
LABELS_IN = CENTURIA / "centuria.csv"
LABELS_OUT = CENTURIA / "centuria_labels.csv"
LABELLED_IMAGES = CENTURIA / "images_labelled"
PROCESSED = Path("/data/alaczko/processed/centuria")
PREP_CONFIGS = REPO / "configs/centuria/prep"
MODEL_CONFIGS = REPO / "configs/centuria/models"
PIPELINE_OUT = REPO / "data/pipeline"


def prepare_data() -> None:
    """centuria_labels.csv (ID without extension, TypeRev) and images_labelled/."""
    table = pd.read_csv(LABELS_IN)
    table = table[table["Class"].notna()]
    labels = pd.DataFrame({"ID": table["ID"].str.rsplit(".", n=1).str[0], "TypeRev": table["Class"]})
    labels.to_csv(LABELS_OUT, index=False)
    LABELLED_IMAGES.mkdir(exist_ok=True)
    for name in table["ID"]:
        target = LABELLED_IMAGES / name
        if not target.exists():
            try:
                os.link(CENTURIA / "images" / name, target)
            except OSError:  # another file system: copy instead
                shutil.copy2(CENTURIA / "images" / name, target)
    print(f"Labels: {LABELS_OUT} ({len(labels)} images, {labels['TypeRev'].nunique()} classes)")


def preprocess(redo: bool) -> None:
    from pipeline.experiment import run_experiment

    for config in sorted(PREP_CONFIGS.glob("*.yaml")):
        target = PROCESSED / config.stem
        if target.exists() and not redo:
            print(f"Preprocessing {config.stem}: {target} exists, skipped")
            continue
        print(f"\n=== Preprocessing {config.stem}")
        result = run_experiment(str(config))
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(result["preprocessed_path"], target)
        print(f"  copied to {target}")


def run_models(configs: list[Path], gpus: list[str], log_dir: Path) -> dict:
    """Each config as `python run_experiment.py <config>`, one per GPU at a time."""
    todo = queue.Queue()
    for config in configs:
        todo.put(config)
    status, lock = {}, threading.Lock()

    def worker(gpu: str) -> None:
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu}
        while True:
            try:
                config = todo.get_nowait()
            except queue.Empty:
                return
            log = log_dir / f"{config.stem}.log"
            start = time.time()
            with lock:
                print(f"[GPU {gpu}] start {config.stem}", flush=True)
            with log.open("w") as out:
                code = subprocess.run([sys.executable, str(REPO / "run_experiment.py"), str(config)],
                                      cwd=REPO, env=env, stdout=out, stderr=subprocess.STDOUT).returncode
            with lock:
                status[config.stem] = "ok" if code == 0 else f"failed (exit {code}, see {log})"
                print(f"[GPU {gpu}] {status[config.stem]:<6} {config.stem} ({(time.time() - start) / 60:.1f} min, "
                      f"{len(status)}/{len(configs)} done)", flush=True)

    threads = [threading.Thread(target=worker, args=(gpu,)) for gpu in gpus]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return status


def summarize(started: float, out_dir: Path) -> None:
    """The summary.csv of every CENT- sweep that started after `started`."""
    tables = []
    for summary in PIPELINE_OUT.glob("sweep-CENT-*/summary.csv"):
        if summary.parent.stat().st_mtime < started:
            continue
        table = pd.read_csv(summary)
        table.insert(0, "config", summary.parent.name.split("__")[0].removeprefix("sweep-"))
        tables.append(table)
    if not tables:
        print("No sweep summaries found")
        return
    table = pd.concat(tables, ignore_index=True)
    if "ari" in table:
        table = table.sort_values("ari", ascending=False)
    table.to_csv(out_dir / "summary.csv", index=False)
    columns = [c for c in ["config", "clustering.method", "clustering.metric", "acc", "ari", "nmi", "silhouette"]
               if c in table]
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(table[columns].head(25).to_string(index=False))
    print(f"\nAll {len(table)} runs: {out_dir / 'summary.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the existing models on the Centuria data")
    parser.add_argument("--gpus", default="0", help="GPUs to run models on, one at a time each, e.g. 0,1,2")
    parser.add_argument("--configs", default=str(MODEL_CONFIGS),
                        help="folder of the configs to run, e.g. configs/centuria/top5 (see configs/centuria/make_top5.py)")
    parser.add_argument("--only", default="*", help="only the model configs matching this pattern, e.g. 'CENT-NTX-*'")
    parser.add_argument("--skip-prep", action="store_true", help="do not run the preprocessing")
    parser.add_argument("--redo-prep", action="store_true", help="run the preprocessing even if its folders exist")
    parser.add_argument("--dry-run", action="store_true", help="only check every config (run_experiment.py --dry-run)")
    args = parser.parse_args()

    configs = [c for c in sorted(Path(args.configs).resolve().glob("*.yaml")) if fnmatch.fnmatch(c.stem, args.only)]
    if args.dry_run:
        for config in ([] if args.skip_prep else sorted(PREP_CONFIGS.glob("*.yaml"))) + configs:
            code = subprocess.run([sys.executable, str(REPO / "run_experiment.py"), str(config), "--dry-run"],
                                  cwd=REPO, capture_output=True, text=True)
            print(f"{'ok' if code.returncode == 0 else 'FAILED':<6} {config.relative_to(REPO)}")
            if code.returncode:
                print(code.stderr[-2000:])
        return

    started = time.time()
    out_dir = PIPELINE_OUT / f"centuria__{datetime.now():%Y%m%d-%H%M%S}"
    (out_dir / "logs").mkdir(parents=True)

    prepare_data()
    if not args.skip_prep:
        preprocess(args.redo_prep)

    # Create the MLflow experiment once, so the parallel runs do not race to create it.
    from pipeline import tracking
    tracking.setup("centuria-eval")

    print(f"\n=== {len(configs)} model configs on GPUs {args.gpus}")
    status = run_models(configs, args.gpus.split(","), out_dir / "logs")
    failed = {name: why for name, why in status.items() if why != "ok"}
    print(f"\n{len(status) - len(failed)}/{len(status)} configs ok")
    for name, why in failed.items():
        print(f"  {name}: {why}")
    summarize(started, out_dir)


if __name__ == "__main__":
    main()
