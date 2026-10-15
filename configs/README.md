# configs

Run any of these with `python run_experiment.py configs/<file>.yaml`
(`--dry-run` to check, `--set key.sub=value` to override). A config only
needs what differs from `default.yaml`.

`template.yaml` lists every setting of every step and model with its default
and options (alternative steps and models commented out): copy it to start a
new config, and delete what you don't change.

| File | |
|---|---|
| `default.yaml` | the defaults every config is merged into |
| `template.yaml` | every option; as it stands raw scans -> BackgroundEraser -> Normalizer -> Repair -> Vectorizer -> VAE -> clustering -> evaluation |
| `prep_vector_axes.yaml` | preprocessing only: vectorized shards |
| `prep_vector_axes_svg.yaml` | preprocessing only: vectorized shards, also as SVG |

## Where the outputs go

Everything a run writes is under `data/pipeline/<run id>/`, each step in its
own folder:

```
data/pipeline/<run id>/
  config.yaml                                  resolved config
  PreprocessingStep/<Step>/                    images (memory mode: only the last step's),
                                               extracted_metadata.csv, _debug/, _svg/
  FeatureLearningStep/<Model>/                 model.pt, model.yaml, features.csv
  ClusteringStep/HierarchicalClustering/       clusters.csv
  EvaluationStep/                              metrics.json
```

The preprocessing steps record per image the scan's size and DPI, the scale
and offset of the final image on the scan, px/cm, and the coordinate system
(`extracted_metadata.csv`). Folders starting with `_` are side outputs: image
collection skips them, so a step folder can be used as input as it is.

To reuse preprocessed images, copy (or move) the last step's folder by hand,
e.g. to `data/processed/<name>/`, split it with
`python split_by_labels.py data/processed/<name>`, and point
`data.preprocessed_path` at it. A trained model is reused with
`model.checkpoint: data/pipeline/<run id>/FeatureLearningStep/VAE/model.pt`.
