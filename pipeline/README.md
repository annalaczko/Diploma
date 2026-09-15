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

`pipeline.pipeline.Pipeline` is a thin orchestrator that chains a chosen set
of steps together into one runnable experiment - it owns no logic of its own
beyond calling each step's `run()` in order and threading its output into the
next step's input:

```python
from pipeline.pipeline import Pipeline

pipeline = Pipeline(
    preprocessing_steps=[...],   # list of PreprocessingStep instances
    feature_learning_step=...,   # one FeatureLearningStep instance
    clustering_step=...,         # one ClusteringStep instance
    evaluation_step=...,         # optional EvaluationStep instance
)
result = pipeline.run(data)
# result = {"features": ..., "labels": ..., "metrics": ...}
```

In practice, most experiments in this project call the concrete steps
directly (see below) rather than going through `Pipeline`, since each stage
usually needs its output inspected/tuned before moving on to the next one.
**Note:** `main.py` at the repo root is an outdated stub — it references
classes (`PretrainedFeatureLearning`, `ImageEnhancement`, `ObjectDetection`)
that no longer exist in this codebase. Use the example below, or
`notebooks/sandbox.ipynb`, as the up-to-date usage reference instead.

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
../data/pipeline/<run_id>/<StageBaseClass>/<ConcreteClassName>/
```

e.g. `../data/pipeline/run_2026-09-04/PreprocessingStep/Normalizer/`. Passing
the same `run_id` across a chain of steps keeps one experiment's outputs
grouped together and separate from other runs.

## Directory layout

```
pipeline/
├── pipeline.py              # Pipeline orchestrator class
├── steps/                   # Step: shared abstract base for every stage
├── preprocessing/           # BackgroundEraser, Repair, Normalizer, Augmentor + shared image-loading helpers
├── feature_learning/        # VAE, SimCLR, BYOL, VAEClassifier
├── clustering/               # HierarchicalClustering
└── evaluation/                # EvaluationWithGroundTruth, EvaluationWithNoGroundTruth
```
