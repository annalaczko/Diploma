# `pipeline` — Pottery-Sherd Image Analysis Pipeline

This package implements an image-analysis pipeline for pottery sherd photographs
(the ROCOPOT dataset). It goes from raw photographs to a clustering of pottery
types, with an optional supervised classifier and evaluation metrics along the
way.

## Architecture

The pipeline has four stages, each defined as an interchangeable `Step`
subclass:

```
preprocessing  ->  feature_learning  ->  clustering  ->  evaluation (optional)
```

| Stage | Package | Purpose |
|---|---|---|
| Preprocessing | [`pipeline/preprocessing`](preprocessing/README.md) | Clean up raw photos: remove background, repair broken line drawings, normalize size. |
| Feature learning | [`pipeline/feature_learning`](feature_learning/README.md) | Turn preprocessed images into fixed-length feature vectors (unsupervised VAE, SimCLR, or BYOL) or predict pottery type directly (supervised classifier). |
| Clustering | [`pipeline/clustering`](clustering/README.md) | Group feature vectors into clusters, e.g. hierarchical clustering. |
| Evaluation | [`pipeline/evaluation`](evaluation/README.md) | Score cluster/classifier output against ground truth (or unsupervised, via silhouette score). |

Every stage's concrete classes inherit from a common `Step` base class:
see [`pipeline/steps`](steps/README.md).

`pipeline.pipeline.Pipeline` is a thin orchestrator that chains a configured
set of steps into one runnable experiment. See [Two ways to run the
pipeline](#two-ways-to-run-the-pipeline) below; `main.py` at the repo root is
a runnable example.

You can still call every step on its own (`step.run()`), as in the
step-by-step example below. This is handy while a stage's output still needs
inspecting/tuning before moving on to the next one.

## Two ways to run the pipeline

The same configured steps can run in two modes:

| | `mode="disk"` | `mode="memory"` |
|---|---|---|
| How images move | every step's `run()` writes its output folder, which becomes the next step's `input_path` | images are loaded `preprocess_batch_size` at a time and passed through every step's `process_batch()` |
| Intermediate folders | yes (one per step) | none (only debug images and `features.csv`) |
| Kept in RAM | nothing between steps | only the final, normalized images, as one `(N, 256, 256)` uint8 array (`InMemoryImages`, ~64 KB/image) |
| `Augmentor` | writes `keep_original` + `num_augmentations` copies per image | not run as a step: its `transform` is applied on the fly to *training* images, a fresh random view every epoch |

```python
from pipeline.pipeline import Pipeline
from pipeline.preprocessing import BackgroundEraser, Normalizer, Repair, Augmentor
from pipeline.feature_learning import VAE
from pipeline.clustering import HierarchicalClustering
from pipeline.evaluation import EvaluationWithGroundTruth

vae = VAE(k=128, run_id=run_id).to(device)
pipe = Pipeline(
    preprocessing_steps=[            # no input_path needed: run() wires them up
        BackgroundEraser(mode="segmentation", model=model_path, mask_margin=0.05),
        Normalizer(background_color=(255, 255, 255)),
        Repair(contrast_factor=5, skeletonize=False, output_thickness=2),
        Augmentor(num_augmentations=2),
    ],
    feature_learning_step=vae,
    # optional; a factory because clustering reads features.csv in __init__
    clustering_factory=lambda csv: HierarchicalClustering(input_path=csv, cluster_num=55),
    # optional, only used together with clustering_factory; built from the predicted labels
    evaluation_factory=lambda y_pred: EvaluationWithGroundTruth(y_true, y_pred),
)

# In memory, without intermediate folders:
result = pipe.run(source_folder, mode="memory", preprocess_batch_size=64,
                  feature_mode="train_model", epochs=500, lr=1e-4, batch_size=128)
# result = {"images": InMemoryImages, "features": DataFrame, "labels": ..., "metrics": ...}

# The same through folders on disk:
result = pipe.run(source_folder, mode="disk", feature_mode="train_model", epochs=500)
# result = {"output_path": <last preprocessing folder>, "features": ..., "labels": ...}
```

- `save_preprocessed` (memory mode only) also writes preprocessed images to
  disk, into each step's `output_path`, the same folders disk mode uses:
  `None` (default) writes nothing, `"last"` only the final images (what
  feature learning gets; `result["output_path"]` points at them), `"all"`
  every step's output. An `Augmentor` is not a step in memory mode, so its
  random views are never saved. Disk mode always writes every step.
- Per-image metadata: `BackgroundEraser` records each image's scan size,
  DPI and crop, and with `record_axes=True` the coordinate system (instead of
  drawing it). `Pipeline` links steps that draw the axes
  (`Vectorizer(axes="draw")`, `AxisRenderer`) to it, and after preprocessing
  writes `extracted_metadata.csv` (in the final images' pixel coordinates,
  with the scale/offset back to the scan and px/cm) next to the saved
  images, returns it as `result["metadata"]` and logs it to MLflow. See
  [`preprocessing/README.md`](preprocessing/README.md).
- `feature_mode` is passed as the feature step's `mode`
  (`"train_model"` / `"extract_features"`). Every other keyword argument
  (`epochs`, `lr`, `batch_size`, ...) goes to the feature step's `run()`.
  A step whose `run()` has no `mode` (`VAEClassifier`) ignores it.
- To reuse already preprocessed images with another model (e.g. extract
  features with a trained VAE, then train a `VAEClassifier`), skip the
  pipeline and hand the images over directly:
  `classifier.set_images(result["images"])`. To write them out for a look,
  use `result["images"].save(folder)`. To inspect a few batches, use
  `next(pipe.iter_batches(source_folder))`.
- Memory mode never JPEG-recompresses between steps, so its images can differ
  very slightly from disk mode's when the intermediate files are `.jpg`.
- In memory mode, images are named the way `BackgroundEraser` names them on
  disk (`<subfolder>__<file>`), so `features["filename"]` and the
  `VAEClassifier`'s label lookup (by stem) work the same in both modes.
- Augmentation differs between the modes. In memory mode an epoch is the
  original dataset size; in disk mode it is `1 + num_augmentations` times as
  large. Epoch counts and early-stopping patience therefore aren't directly
  comparable between the two.
- `SimCLR`/`BYOL` already augment their own two views, so in memory mode
  they ignore the `Augmentor` (and print a warning).

## Config-driven runs (`run_experiment.py`)

A whole run can be described by a short YAML file and started from the
command line. `configs/default.yaml` holds every setting with comments; an
experiment config lists only what it changes (nested sections are merged,
lists replaced):

```yaml
# configs/vae_k16_beta32.yaml
model:
  params: {k: 16, beta_max: 32}
```

```bash
python run_experiment.py configs/vae_k16_beta32.yaml
python run_experiment.py configs/vae_k16_beta32.yaml --set clustering.cluster_num=30 --set training.epochs=200
python run_experiment.py configs/vae_k16_beta32.yaml --dry-run    # resolved config + run id only
```

- `data.source: preprocessed` (default) trains on ready images in
  `data.preprocessed_path` (default `data/processed/legion/no_axes/train`);
  `raw` runs `preprocessing.steps` on `data.raw_path` first; the result is
  in the last step's folder (see below), and can be copied by hand to
  e.g. `data/processed/<name>` to be reused as `preprocessed_path`.
- `stop_after` ends a run early: `preprocessing` (with `data.source: raw`)
  only produces the preprocessed images (run name `prep-<steps>__raw`,
  e.g. `configs/only_prep/vector_axes.yaml`); `features` trains/saves the model and
  extracts features, without clustering and evaluation. `null` = whole run.
- `model.type`: `vae`, `simclr`, `byol`, `handcrafted` (measured from
  the Vectorizer's SVGs) or `dinov3` (pretrained DINOv3, the
  sandbox.ipynb method) - handcrafted and a frozen dinov3 train nothing and
  take no checkpoint; dinov3 with `model.params.finetune: true` trains like
  the learned models; see
  [`feature_learning/README.md`](feature_learning/README.md)); `model.params` go to its
  constructor. `model.checkpoint` loads a saved state_dict: with
  `training.train: true` it is fine-tuned (saved as a new file), with
  `false` only features are extracted.
- `data.extract_path` (ready images, e.g. `.../test`) is where features
  come from, and so what is clustered and evaluated; the model still
  trains on `preprocessed_path` (or the raw images). The training images'
  features are kept as `features_train.csv`. Without training
  (`training.train: false`, handcrafted, frozen dinov3) only `extract_path` is read. The
  run name's data part reads `legion-no_axes-train-to-test`.
- `data.features` (a `features.csv` of an earlier run) skips preprocessing
  and the model: the run only clusters and evaluates those features. Its
  name takes the model and data part from that run's folder.
- `clustering.method: auto` picks the linkage by cophenetic correlation;
  `standardize` z-scores the features first, as in the notebook.
- Evaluation: silhouette on every image, ACC/ARI/NMI on the images with a
  label in `data.labels` (`ID` = file name without extension, `TypeRev`).

Each run gets an id generated from its config:
`<model>-<params>_<training>__<clustering>__<data>__<date-time>`, e.g.
`vae-k16-beta32_e100__ward-euclidean-c55__legion-no_axes-train__20261012-153000`
(`name:` in the config replaces the generated part). The same id is used for:

| Where | What |
|---|---|
| MLflow run name (experiment `experiment:`) | all settings as `cfg/...` params, metrics, `config.yaml`; tags `model_path`, `started_from`, `data`, `output_dir`, `config_file`, `git_commit`, `git_dirty` |
| `data/pipeline/<run id>/` | everything the run writes, each step in its own folder (below) |

```
data/pipeline/<run id>/
  config.yaml                                  resolved config
  PreprocessingStep/<Step>/                    images (memory mode: only the last step's),
                                               extracted_metadata.csv, _debug/, _svg/
  FeatureLearningStep/<Model>/                 model.pt (state_dict), model.yaml (config +
                                               MLflow run id), features.csv (+ features_train.csv
                                               with data.extract_path)
  ClusteringStep/HierarchicalClustering/       clusters.csv, feature_importance.csv
  EvaluationStep/                              metrics.json
```

Model files are never overwritten (the run id has the date and time) and
are not stored in MLflow; the run records their path. Relative paths in configs are relative to the repo
root, and float values like `1e-4` are read as numbers. The logic lives in
`pipeline/experiment.py` (`load_config`, `run_name`, `run_experiment`, `run_sweep`).

### Sweeps

A `sweep` maps settings (dotted or nested keys) to lists of values; the run
becomes one ordinary run per combination:

```yaml
sweep:
  model.params.k: [16, 32]
  training.lr: [1.0e-3, 1.0e-4]
  clustering.cluster_num: [20, 30, 55]     # 2 x 2 x 3 = 12 runs, 4 models trained
```

- The model is trained once per combination of the non-clustering
  settings. The other combinations of that group only cluster its
  `features.csv` again (`data.features`), so they take seconds. If the
  training run fails, its group is skipped; other failures are reported
  and the sweep goes on.
- Run names get the swept settings the generated name does not show,
  e.g. `...__legion-no_axes-train__lr0.001`. Every run has its own folder
  and MLflow run, tagged `sweep: <sweep id>`.
- `data/pipeline/sweep-<config name>__<date-time>/` holds `sweep.yaml` (the
  config) and `summary.csv` (swept settings, run id, status, metrics per
  run; rewritten after every run). `--dry-run` lists the runs.
- Combinations that cannot run are left out and listed: `centroid`/`ward`
  with a metric other than `euclidean`, and `method: auto` with a second
  metric (auto ignores it, so it would repeat a run). The same clustering
  settings in a single run are an error (an unknown method or metric too,
  which `HierarchicalClustering` would quietly replace).
- A list-valued setting is swept with a list of lists (`[[0, 0], [5, 5]]`).
  `--set` cannot set a swept setting, and settings the run never gets to
  (e.g. clustering with `stop_after: features`) cannot be swept.

## Experiment tracking (MLflow)

`pipeline/tracking.py` logs every stage to MLflow. Pass `experiment_name`
(and optionally `run_name`) to `Pipeline.run` to record the whole run:

```python
result = pipe.run(source_folder, mode="memory", feature_mode="train_model", epochs=500,
                  experiment_name="vae_k128", run_name="mask0.02_axes")
```

What ends up in the run:

| Stage | Params | Metrics | Artifacts |
|---|---|---|---|
| Pipeline | `input_path`, `mode`, `feature_mode`, step chain, `feature/train/*` | `preprocessing/n_images` (memory mode) | |
| `BackgroundEraser` | `preprocessing/<step>/*` (its settings) | `n_images`, `no_shard_count`, `no_axes_count` (with `record_axes`) | `preprocessing/<step>/no_shard.txt`, `no_axes.txt`: names of the images without a shard / coordinate system, one per line |
| Feature learning | `feature/<step>/*` (architecture, e.g. `k`, `recon_loss`), `feature/train/*` (all training arguments) | per epoch: `feature/loss`, `smoothed_loss`, `lr` (+ VAE: `elbo`, `recon`, `kld`, `beta`; `VAEClassifier`: accuracies); `best_epoch`, `stopped_epoch` | `feature/features.csv` (`VAEClassifier`: `training_history.csv`) |
| Clustering | `clustering/cluster_num`, `method`, `metric` | `n_clusters`, cluster size min/max/mean, `singleton_clusters` (`compare_all`: `best_cophenetic`) | `clustering/clusters.csv` (filename -> cluster), `compare_all.csv`, `clustering/feature_importance.csv` (per dimension: eta2 = share of its variance explained by the clusters, ANOVA F; the top `clustering.feature_importance` are also printed at the end of the run) |
| Evaluation | | `eval/acc`, `ari`, `nmi` or `eval/silhouette` | `eval/per_class_accuracy.csv` |

Image names use the flat `<subfolder>__<file>` form in both modes.

Steps called one by one (as in the notebook) log the same way inside a run:

```python
from pipeline import tracking

with tracking.start_run("vae_k128", run_name="notebook"):
    vae.run(mode="train_model", epochs=100)
    labels = HierarchicalClustering(input_path=csv, cluster_num=55).run()
    metrics = EvaluationWithGroundTruth(y_true, labels).run()
```

Without an active run (and without mlflow installed) the steps log nothing.
Runs are stored in the repo, independent of the working directory:
`mlflow.db` (SQLite) and `mlartifacts/` (both gitignored). Set
`MLFLOW_TRACKING_URI` to use another store or a tracking server. Browse
the runs from the repo root with:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Typical run (end to end)

This mirrors the actual workflow used in `notebooks/sandbox.ipynb`:

```python
from pipeline.preprocessing import BackgroundEraser, Normalizer, Repair
from pipeline.feature_learning import VAE, VAEClassifier
from pipeline.clustering import HierarchicalClustering
from pipeline.evaluation import EvaluationWithGroundTruth

run_id = "run_2026-09-04"

# 1. Preprocessing: crop to the shard, normalize size, repair line drawings.
erased = BackgroundEraser(input_path="data/raw/some_survey", run_id=run_id).run()
cropped = Normalizer(input_path=erased, background_color=(255, 255, 255), run_id=run_id).run()
repaired = Repair(input_path=cropped, max_distance=20, output_thickness=3, run_id=run_id).run()

# 2a. Feature learning (unsupervised): train a VAE, extract latent features.
vae = VAE(input_path=repaired, f=4, n=3, s=1, k=128, run_id=run_id)
vae.to(device)
vae.run(mode="train_model", epochs=10)
features = vae.extract_features(batch_size=128)

# 2b. ...or supervised classification on top of the VAE's encoder.
classifier = VAEClassifier(vae=vae, labels_path="data/rocopot_results.csv", input_path=repaired)
classifier.to(device)
history = classifier.run(epochs=100)

# 3. Clustering on the extracted features.
clustering_step = HierarchicalClustering(input_path="features.csv", cluster_num=55)
labels = clustering_step.run()

# 4. Evaluation against ground truth.
eval_step = EvaluationWithGroundTruth(y_true=true_labels, y_pred=labels)
eval_step.run()
```

## Where output goes

Every `Step` (see [`pipeline/steps`](steps/README.md)) auto-derives an output
directory if you don't pass `output_path` explicitly:

```
<repo>/data/pipeline/<run_id>/<StageBaseClass>/<ConcreteClassName>/
```

e.g. `data/pipeline/run_2026-09-04/PreprocessingStep/Normalizer/` or
`data/pipeline/run_2026-09-04/FeatureLearningStep/VAE/features.csv`. This
doesn't depend on where you run from. Passing
the same `run_id` across a chain of steps keeps one experiment's outputs
grouped together and separate from other runs.

## Directory layout

```
pipeline/
├── pipeline.py              # Pipeline orchestrator: disk or in-memory mode
├── tracking.py              # optional MLflow logging used by every stage
├── experiment.py            # config-driven runs (run_experiment.py, configs/)
├── steps/                   # Step: shared abstract base for every stage
├── preprocessing/           # BackgroundEraser, Repair, Normalizer, Augmentor + shared image-loading helpers
├── feature_learning/        # VAE, SimCLR, BYOL, VAEClassifier
├── clustering/               # HierarchicalClustering
└── evaluation/                # EvaluationWithGroundTruth, EvaluationWithNoGroundTruth
```
