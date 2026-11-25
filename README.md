# Feature Learning and Clustering for Archaeological Pottery Typology

Code for my diploma thesis (MSc Data Science, TU Wien; supervisor Martin
Kampel, co-supervisor Irene Ballester Campos). The thesis proposal is in
[`docs/proposal_annalaczko_0609.pdf`](docs/proposal_annalaczko_0609.pdf).

The question is whether unsupervised feature learning and clustering can
recover **form-based groupings** of pottery sherds from fragmentary,
hand-drawn archival section drawings. The pipeline takes raw document scans
to cleaned-up single-sherd images, turns them into feature vectors, clusters
them and compares the clusters with the experts' Form annotations.

## Data

The drawings come from the [LEGION project](https://legion-hsa-2-0.github.io/):
Roman common ware from the Carnuntum excavations, documented as 2D section
drawings in a collection of about 70,000 archival documents.

- **Training data:** a subset of about 6,300 scanned documents. These are
  hard to work with: the scans vary in quality, the drawings carry
  handwritten annotations, and some pages show more than one sherd.
- **Evaluation:** an annotated subset of about 500 drawings with Form labels
  (released as CENTURIA). This is the ground truth for accuracy, NMI and ARI.
- **External benchmark:** the ROCOPOT rim profiles.

No data is included in this repository.

## Approach and status

| Stage | Implemented | Planned (see the proposal) |
|---|---|---|
| Preprocessing | YOLO segmentation of the sherd on the page, size normalization, line repair, vectorization, optional drawing of the coordinate axes | ablation of each enhancement step |
| Feature learning | pretrained DINOv3 baseline (frozen); DINOv3 fine-tuned self-supervised (NT-Xent, DINO self-distillation); VAE (baseline following Parisotto et al., 2022); SimCLR; BYOL; hand-crafted profile measures | nearest-neighbour retrieval evaluation (precision@k) |
| Clustering | hierarchical clustering, every linkage and metric, sweeps | K-means; semi-supervised clustering (prototype seeding, must-link / cannot-link) |
| Evaluation | ACC (best cluster-to-class matching), NMI, ARI on the labelled subset; silhouette on unlabelled data; t-SNE maps | |
| Metadata | per-image scan metadata (DPI, scale, coordinate system) | sherd metadata (ware, material, diameters) as an extra input |

Preprocessing is custom because existing tools did not work on these scans.
In preliminary tests PyPotteryLens did not detect the fragments, and
PyPotteryInk needed manual prompts and gave inconsistent lines.

## Repository layout

| Path | Purpose |
|---|---|
| [`pipeline/`](pipeline/README.md) | the library: preprocessing, feature learning, clustering, evaluation. **The detailed documentation is here**, with one README per subpackage |
| [`configs/`](configs/README.md) | YAML experiment configs for `run_experiment.py` (`template.yaml` lists every option) |
| `tests/` | pytest tests |
| `notebooks/` | exploration: `sandbox.ipynb` (the main experiments), YOLO label preparation, synthetic data generation |
| `docs/` | the thesis proposal |
| `run_experiment.py` | runs one config (or a sweep of configs) end to end |
| `run_centuria.py` | evaluates the trained models on the annotated subset (CENTURIA), in parallel on several GPUs |
| `cluster_folders.py` | sorts a run's images into one folder per cluster, with contact sheets, for checking clusters by eye |
| `main.py` | minimal example of the `Pipeline` API |
| `split_by_labels.py`, `split_train_val.py`, `mirror_split.py` | data-splitting helpers (labelled test / train; YOLO train / val / test) |
| `optimization.md` | notes on profiling and speeding up the preprocessing |

Run outputs (`data/pipeline/<run id>/`) and the MLflow store (`mlflow.db`,
`mlartifacts/`) are created locally and are not tracked.

## Quick start

```bash
python3.10 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

python run_experiment.py configs/dinov3.yaml --dry-run   # check the resolved config
python run_experiment.py configs/dinov3.yaml
pytest
```

Start with [`pipeline/README.md`](pipeline/README.md) for the architecture.
