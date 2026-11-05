# Night runs (DINOv3 fine-tuning, new line-drawing augmentation)

Every config trains on `<variant>/train` and clusters and evaluates `<variant>/test_top5`
(P = pagecrop, T = tightcrop, N = no_axes, A = axes). Each one sweeps clustering
(average-euclidean, average-cosine, ward-euclidean, 5 clusters). The `ntx_<v>` and
`dino_<v>` configs also sweep `unfreeze_blocks` over 2 and 3, so they train two models.
MLflow experiment: `dinov3-night`.

One tmux window per GPU, each running its queue one config after the other
(`;` = the next one starts even if one fails):

```bash
# GPU 0: NT-Xent, all variants x unfreeze_blocks 2/3, then 100 epochs on P-N and T-N  (~520 min)
cd /caa/Homes01/alaczko/Diploma/Diploma && source venv/bin/activate && export CUDA_VISIBLE_DEVICES=0 && for c in ntx_P-N ntx_T-N ntx_P-A ntx_T-A ntx_P-N_e100 ntx_T-N_e100; do python run_experiment.py configs/night/$c.yaml; done

# GPU 1: DINO, all variants x unfreeze_blocks 2/3  (~545 min)
cd /caa/Homes01/alaczko/Diploma/Diploma && source venv/bin/activate && export CUDA_VISIBLE_DEVICES=1 && for c in dino_P-N dino_T-N dino_P-A dino_T-A; do python run_experiment.py configs/night/$c.yaml; done

# GPU 2: DINO, one setting changed at a time on P-N and T-N  (~540 min)
cd /caa/Homes01/alaczko/Diploma/Diploma && source venv/bin/activate && export CUDA_VISIBLE_DEVICES=2 && for c in dino_P-N_lr1e-4 dino_T-N_lr1e-4 dino_P-N_tt0.07 dino_T-N_tt0.07 dino_P-N_e100 dino_T-N_e100; do python run_experiment.py configs/night/$c.yaml; done
```

The GPU 2 runs compare with the `unfreeze_blocks2` runs of `dino_P-N.yaml` / `dino_T-N.yaml`
(GPU 1), which differ from them in that one setting only.

A failed run shows up in the sweep's summary table at the end of its output
(`status: failed: ...`) and in `data/pipeline/sweep-<name>__<time>/summary.csv`.
