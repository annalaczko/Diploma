"""Pottery-sherd image analysis pipeline.

Chains four stages, each defined as an interchangeable `Step` subclass
(see `pipeline.steps.Step`):

    preprocessing (pipeline.preprocessing)
        -> feature_learning (pipeline.feature_learning)
        -> clustering (pipeline.clustering)
        -> evaluation (pipeline.evaluation, optional)

`pipeline.pipeline.Pipeline` wires a chosen set of these steps together
into one runnable experiment; see `main.py` for an example.
"""
