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
| `vae.yaml` / `vae_axes.yaml` | VAE trained on `legion/raster[_axes]/train`, features of `test` clustered and evaluated (clustering sweep) |
| `handcrafted.yaml` / `handcrafted_axes.yaml` | hand-crafted profile measures of vectorized images with their `_svg/` (nothing trained; clustering sweep) |
| `dinov3.yaml` / `dinov3_axes.yaml` | frozen pretrained DINOv3 (the sandbox.ipynb method) on `legion/raster[_axes]/test` (nothing trained; clustering sweep) |
| `dinov3_finetune.yaml` | DINOv3 fine-tuned self-supervised (NT-Xent, no labels) on `legion/raster/train`, features of `test` clustered and evaluated (clustering sweep) |
| `only_prep/*.yaml` | preprocessing only (`stop_after: preprocessing`): raster / vector images, with or without the axes |

The three feature methods are chosen with `model.type: vae | handcrafted | dinov3`
(`simclr` and `byol` also work). Switching `model.type` (also with
`--set model.type=...`) drops the `model.params` that belong to the other
kind of model. `handcrafted` and a frozen `dinov3` train nothing, so they
take no `model.checkpoint`, and the frozen dinov3 only uses
`training.batch_size`. With `model.params.finetune: true`, dinov3 trains
like the learned models (`training.*`, `model.checkpoint`). DINOv3 needs
`transformers` and a one-off `huggingface-cli login` (the weights are gated).

## Sweeps

`sweep:` tries every combination of the listed values, training each model
only once (combinations that differ only in `clustering.*` re-cluster its
features). Details: [`pipeline/README.md`](../pipeline/README.md#sweeps).

```yaml
sweep:
  model.params.k: [16, 32]
  clustering.cluster_num: [20, 30, 55]
```

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
