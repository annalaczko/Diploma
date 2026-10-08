# `pipeline.evaluation`

Scores clustering/classification results, either against ground-truth
labels (ARI, NMI, accuracy) or unsupervised (silhouette score), and draws
visual checks of a learned latent space.

```
pipeline/evaluation/
├── evaluation_step.py         # EvaluationStep, EvaluationWithGroundTruth, EvaluationWithNoGroundTruth
└── latent_visualization.py    # t-SNE thumbnail maps and cluster-to-cluster paths (plain functions, not Steps)
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
step.run()   # computes step.ari, step.nmi, step.acc, step.per_class and prints the first three
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

- **`per_class_accuracy(y_true, y_pred) -> pd.DataFrame`** (stored as
  `step.per_class` by `run()`) — the same accuracy broken down by true
  class, under the *same* optimal matching, so the overall ACC is exactly
  these per-class correct counts summed and divided by the total. One row
  per true class, sorted by class size, with columns `class`, `n`,
  `matched_cluster`, `correct`, `accuracy`, `top_cluster` and
  `top_cluster_share`. When there are fewer clusters than classes, some
  classes get no cluster at all (`accuracy` 0, `matched_cluster` `None`).
  `top_cluster` / `top_cluster_share` still show where most of such a
  class's samples went, independent of the matching.

Both methods share `_best_matching`, which builds the contingency matrix
and runs the Hungarian matching. It raises `ValueError` if `y_true` and
`y_pred` differ in length or are empty.

## `latent_visualization.py` — visual checks of a latent space

Plain plotting functions (not `Step`s), used from notebooks to see what a
trained model's feature space looks like. Each plotting function returns a
matplotlib `Axes` or `Figure`.

Getting the inputs:

- `encode_folder(vae, folder, device, batch_size=128, thumb_size=64)` →
  `(filenames, mu, thumbs)`: encodes every image under `folder` with a
  trained `VAE` (eval mode) and returns the latent means plus small
  `[0, 1]` grayscale thumbnails.
- `load_thumbnails(folder, filenames, thumb_size=64)`: thumbnails for
  features computed elsewhere (e.g. a `features.csv` from a model without
  a decoder, such as DINOv3), in `filenames` order.
- `embed_2d(features, perplexity=30, random_state=0)`: 2D t-SNE embedding.
  Local neighbourhoods are kept, but distances between far-apart groups are
  not meaningful.
- `decode(vae, z, device)`: decodes latent vectors back to images.

Plots:

- `thumbnail_map(coords, thumbs, groups, title, ax=None, zoom=0.45, max_groups=8)`:
  draws every thumbnail at its 2D position, with its ink tinted by group
  (cluster id or true class). The `max_groups` largest groups get a color
  from a fixed palette (`GROUP_COLORS`) and are labelled at their median
  position. The rest are drawn gray as "other".
- `cluster_transitions(vae, mu, clusters, thumbs, device=None, steps=8, title=None)`:
  for each consecutive pair of clusters, decodes `steps` points on the
  straight line between their latent *medoids* (top row) and shows the
  real image nearest to each point (bottom row; bold titles mark images
  from a third cluster). Medoids are used instead of centroids because a
  centroid averages very different shapes and tends to land in an empty
  region of the latent space, where the decoder only produces broken
  fragments. With `vae=None` (features without a decoder), only the
  nearest-real-image row is drawn.
- `neighbour_paths(features, clusters, thumbs, steps=8, n_neighbors=10, title=None)`:
  needs no decoder. For each consecutive pair of clusters, it follows the
  shortest path between their medoids through a symmetric
  k-nearest-neighbour graph of the real images, so every step is a real
  sherd close to the previous one. Raises `ValueError` if two clusters are
  disconnected in the graph (raise `n_neighbors`).

Clusters are visited in `cluster_order(centroids)` order: a greedy
nearest-neighbour path through the cluster centroids, starting from the one
farthest from the overall mean, so consecutive clusters are latent-space
neighbours.

```python
from pipeline.evaluation.latent_visualization import (
    encode_folder, embed_2d, thumbnail_map, cluster_transitions,
)

names, mu, thumbs = encode_folder(vae, folder, device)
coords = embed_2d(mu)
thumbnail_map(coords, thumbs, groups=labels, title="VAE latent space by cluster")
cluster_transitions(vae, mu, labels, thumbs, device=device)
```
