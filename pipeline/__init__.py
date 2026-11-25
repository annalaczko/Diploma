"""Pottery-sherd image analysis pipeline.

From scanned drawings of pottery sherds to groups (clusters) of similar
forms, in four stages. Each stage is a `Step` subclass, so every
part can be swapped for another:

    preprocessing        clean the scans: one sherd per image, same size
    (pipeline.preprocessing)
        -> feature learning   one vector of numbers per image
           (pipeline.feature_learning)
        -> clustering         group similar vectors
           (pipeline.clustering)
        -> evaluation         compare the groups with the Form labels
           (pipeline.evaluation)

Suggested reading order for a first look:
    1. steps/step.py                       what every step has in common
    2. preprocessing/preprocessing_step.py the two run modes (disk / memory)
    3. preprocessing/background_eraser.py, normalizer.py, repair.py
                                           the usual preprocessing chain
    4. feature_learning/vae.py             the main model (then simclr.py,
                                           byol.py, dinov3.py,
                                           handcrafted_features.py)
    5. clustering/hierarchical_clustering.py, evaluation/evaluation_step.py
    6. pipeline.py                         how Pipeline chains the steps
    7. experiment.py                       config-driven runs
                                           (run_experiment.py, configs/)

`pipeline.pipeline.Pipeline` wires a chosen set of steps into one run (see
main.py); `pipeline.experiment.run_experiment` builds and runs one from a
YAML config. Every package also has a README.md with the details.
"""
