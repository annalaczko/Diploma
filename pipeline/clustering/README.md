# `pipeline.clustering`

Groups learned feature vectors into clusters.

```
pipeline/clustering/
├── clustering_step.py         # ClusteringStep: abstract base
└── hierarchical_clustering.py  # HierarchicalClustering
```

## `ClusteringStep`

```python
from pipeline.clustering import ClusteringStep
```

Common contract for clustering methods applied to fixed feature
representations. Unlike the base `Step.run(data)`, concrete clustering steps
take **no `run()` argument** — they read their input (typically a CSV of
precomputed feature vectors) from config passed to `__init__` instead, and
`run()` returns cluster labels, shape `(n_samples,)`. Accepts `run_id` like
every other step, so its auto-derived `output_path` (when one isn't given
explicitly) is namespaced under the same pipeline run as the preprocessing/
feature-learning steps that produced its input.

## `HierarchicalClustering`

Hierarchical clustering over precomputed feature vectors, built on
`scipy.cluster.hierarchy`. Supports the same linkage methods and distance
metrics used by Parisotto, and can sweep all of their combinations to pick
the best one by cophenetic correlation.

```python
HierarchicalClustering(
    input_path,      # CSV path with a 'features' column
    cluster_num=30,   # number of flat clusters to extract
    method=None,      # scipy linkage method; falls back to 'average'
    metric=None,      # scipy distance metric; falls back to 'euclidean'
    output_path=None,  # where results are saved; auto-derived (namespaced by run_id) if omitted
    run_id="default_run",
)
```

`input_path` must be a CSV whose `features` column holds each row's feature
vector as a string (e.g. `"[0.1, 0.2, ...]"` — however `pandas`/`numpy`
happened to write it out when the CSV was saved by a feature-learning step).
`parse_feature_string` (called automatically in `__init__`) undoes this:
strips brackets, normalizes comma/whitespace separators, and parses the
numbers back into a dense `(n_samples, n_features)` float matrix, raising if
rows don't all share the same length.

Available `methods`: `single`, `complete`, `average`, `weighted`,
`centroid`, `ward`. Available `metrics`: `euclidean`, `cityblock`,
`chebyshev`, `cosine`.

### `run()`

Clusters `self.vectors` into `self.cluster_num` flat clusters using
`self.best_method` / `self.best_metric` (either passed in at construction,
or whatever `compare_all()` last picked as the winner), via
`scipy.cluster.hierarchy.fclusterdata`. Returns an `(n_samples,)` array of
cluster ids.

### `compare_all()`

Tries every `(method, metric)` combination and ranks them by **cophenetic
correlation** — how well a linkage's dendrogram distances preserve the
original pairwise distances; higher means the clustering better reflects
the true data geometry. As a side effect, updates `self.best_method` /
`self.best_metric` to the winning combination, so a subsequent `run()` call
uses it. Returns a list of dicts, one per combination, with keys `method`,
`metric` and `cophenetic coefficient score` — useful for sweeping and
plotting to pick a configuration before committing to a final `run()`.
