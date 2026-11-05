# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Diploma-thesis research code: an image-analysis pipeline for pottery-sherd photographs and line drawings (the ROCOPOT dataset). It takes raw photos to learned feature vectors, then to clusters of pottery types, with an optional supervised classifier and evaluation metrics. Most experimentation happens in `notebooks/sandbox.ipynb`. The `pipeline/` package is the reusable library behind it.

Comments and docstrings are in English and written for readers new to the code; `pipeline/__init__.py` gives a suggested reading order.

## Commands

The Python 3.10 virtualenv lives in `venv/` (gitignored). Dependencies are in `requirements.txt`.

```bash
source venv/bin/activate
pip install -r requirements.txt

python run_experiment.py configs/<name>.yaml [--set key.sub=value] [--dry-run]   # config-driven run (configs/default.yaml)

pytest                                   # all tests (pytest.ini: testpaths=tests, pythonpath=., -v)
pytest tests/test_simclr.py              # one file
pytest tests/test_byol.py::TestName::test_name   # one test
```

There is no linter, formatter or build step configured.

## Architecture

There are four stages, and each one is a swappable `Step` subclass (`pipeline/steps/step.py`):

```
preprocessing  ->  feature_learning  ->  clustering  ->  evaluation (optional)
BackgroundEraser -> Normalizer -> Repair [-> Vectorizer | AxisRenderer] [-> Augmentor]  |  VAE / SimCLR / BYOL / VAEClassifier  |  HierarchicalClustering  |  EvaluationWith(No)GroundTruth
```

Each package has a detailed README (`pipeline/README.md` plus one per subpackage). Read the relevant one before changing a stage. Below are the cross-cutting points that you only see by reading several files:

- **Two execution modes, one per-image code path.** `Pipeline.run(source, mode="disk" | "memory", ...)` (`pipeline/pipeline.py`, example in `main.py`) runs the same configured steps either way:
  - In disk mode, each preprocessing step's `run()` writes an output folder, and that folder becomes the next step's `input_path`.
  - In memory mode, batches of `ImageRecord`s go through each step's `process_batch()`, and only the final images are kept, as an `InMemoryImages` array. Feature learners receive them via `set_images()`.
  - `Augmentor` writes copies to disk in disk mode. In memory mode it runs as an on-the-fly training transform instead.
  - Any new preprocessing logic must go in the shared per-image method (`Normalizer.normalize`, `Repair.process_image`, `BackgroundEraser._clear_one`) so the two modes can't drift apart. `tests/test_pipeline.py` checks that they match.
- **Steps can also be called one by one**, which is how most of the notebook works: `step.run()`, then pass the returned path as the next step's `input_path`.
- **`run()` signatures differ between stages.** Preprocessing, feature-learning and clustering steps read `self.input_path` (or in-memory images) and take no `data` argument. `VAE`, `SimCLR` and `BYOL` use `run(mode="train_model" | "extract_features", ...)`, where the default only extracts features.
- **Feature learning hands off to clustering through `features.csv`.** The file has `filename` and `features` columns (the VAE adds `sigma`), where `features` is a stringified vector. `HierarchicalClustering` reads the CSV in `__init__` and parses it back (`parse_feature_string`), which is why `Pipeline` takes a `clustering_factory` (and `evaluation_factory`) rather than a step instance.
- **Every output goes into the run's folder.** When `output_path` is omitted, `Step.__init__` creates `<repo>/data/pipeline/<run_id>/<ParentStageClass>/<ConcreteClass>/` straight away (anchored to the repo, not the CWD). Pass the same `run_id` to every step in an experiment so its outputs stay grouped together. `run_experiment` relies on this: images, `extracted_metadata.csv`, `model.pt`/`model.yaml`, `features.csv`, `clusters.csv` and `EvaluationStep/metrics.json` all land in their step's folder under `data/pipeline/<run id>/`; nothing is written to `data/processed` or `models/` (the user copies results there by hand). Side outputs go to `_debug`/`_svg` subfolders (`PreprocessingStep.side_output_dir`), which `collect_image_paths` skips.
- **Shared image loading.** All path discovery and tensor loading goes through `pipeline/preprocessing/dataset.py` (`collect_image_paths`, `ImageDataset`). Feature learning imports it from there too, so don't add new copies of this logic.
- **Feature learners share one encoder design.** `VAE`, `SimCLR` and `BYOL` use the same 4-block conv encoder, controlled by `f/n/s/k`, so their results can be compared on the training objective alone. `SimCLREncoder` adds BatchNorm, which is needed to prevent collapse, and BYOL reuses `SimCLREncoder`. SimCLR and BYOL get their two augmented views from `TwoViewDataset` (`pipeline/preprocessing/dataset.py`). The `Encoder` flatten size in `vae.py` is hardcoded for 256×256 input with the default `f/n/s`. Recompute it if any of those change.
- **Training loops share `EarlyStopping`** (`early_stopping.py`), a moving-average plateau detector that restores the best weights at the end. The unsupervised models watch training loss (`mode="min"`). `VAEClassifier` watches validation accuracy (`mode="max"`).
- **VAE reconstruction loss is selectable**: `VAE(recon_loss="bce" | ...)` picks from `reconstruction_losses.py`, which includes Dice and soft-IoU for thin-outline images. The KL weight follows a cyclical beta-annealing schedule.
- `pipeline/evaluation/latent_visualization.py` provides latent-space plots: t-SNE thumbnail maps and decoded transitions between clusters.

## Gotchas

- In memory mode, images are named the way `BackgroundEraser` names them on disk (`<subfolder>__<file>`). Label lookups (`VAEClassifier`, by file stem) depend on this.
- The default YOLO model in `BackgroundEraser` is an absolute path under `/caa/Homes01/alaczko/...`. The weights are in `models/` (`*.pt` files are gitignored).
- `data/`, `models/`, `external/` (e.g. DocEnTr), `DeepStructurePriors/`, `PyPotteryInk/`, `giapetto/` and `giappetto/` are gitignored side projects, third-party code or data. They are not part of the `pipeline` package.
- `split_train_val.py` and `mirror_split.py` at the repo root are standalone CLI helpers. They split images into train/val/test folders and copy the matching YOLO label files to the same splits.
