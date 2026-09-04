# `pipeline.evaluation`

Scores clustering/classification results, either against ground-truth
labels (ARI, NMI, accuracy) or unsupervised (silhouette score).

```
pipeline/evaluation/
└── evaluation_step.py   # EvaluationStep, EvaluationWithGroundTruth, EvaluationWithNoGroundTruth
```

## `EvaluationStep`

```python
from pipeline.evaluation import EvaluationStep
```

Common contract for evaluation methods (ARI, NMI, F-measure, precision@k,
nearest-neighbour retrieval, silhouette). `run()` takes predicted labels
(and ground truth / features as needed, passed at construction — see the
two concrete subclasses below) and computes a dict/set of metric values.

## `EvaluationWithNoGroundTruth`

Unsupervised evaluation: scores cluster labels using only the features
themselves, via the **silhouette score** — useful when the dataset isn't
fully labelled.

```python
EvaluationWithNoGroundTruth(features, y_pred)   # features: (n_samples, n_features); y_pred: (n_samples,)
step = EvaluationWithNoGroundTruth(features, y_pred)
step.run()                # computes and prints step.silhouette
```

## `EvaluationWithGroundTruth`

Supervised evaluation: compares predicted cluster/class labels against
known true labels.

```python
EvaluationWithGroundTruth(y_true, y_pred)   # both (n_samples,)
step = EvaluationWithGroundTruth(y_true=true_labels, y_pred=cluster_labels)
step.run()   # computes step.ari, step.nmi, step.acc and prints them
```

Computes:

- **ARI** (`adjusted_rand_score`) and **NMI**
  (`normalized_mutual_info_score`) from `sklearn.metrics`.
- **`clustering_accuracy(y_true, y_pred)`** — best-case accuracy under the
  *optimal* cluster-id ↔ true-label matching. Predicted cluster ids are
  arbitrary integers with no inherent correspondence to the true class ids
  (cluster `3` doesn't necessarily mean the same pottery type across two
  different clustering runs), so raw label agreement would be meaningless.
  Instead:

  1. Build a `(true classes × predicted clusters)` contingency matrix.
  2. Run the **Hungarian algorithm**
     (`scipy.optimize.linear_sum_assignment`) to find the one-to-one
     matching between clusters and true classes that maximizes the number
     of correctly "matched" samples.
  3. Return the fraction of samples matched under that optimal pairing.

  This is the standard unsupervised clustering accuracy metric used in the
  clustering literature, and `y_true`/`y_pred` don't need to share the same
  label space (they're each independently `factorize`d into dense integer
  codes before the matching step) — so it also works to compare, e.g.,
  string pottery-type labels against integer cluster ids directly.
