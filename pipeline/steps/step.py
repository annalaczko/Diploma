"""The `Step` base class that every pipeline stage builds on.

Every piece of work in this project (cleaning an image, training a model,
clustering features, ...) is a `Step`. A step knows where it reads from
(`input_path`), where it writes to (`output_path`) and does its work in
`run()`. The class hierarchy has two levels:

    Step                      this file: paths and naming, shared by all
      PreprocessingStep       pipeline/preprocessing/preprocessing_step.py
      FeatureLearningStep     pipeline/feature_learning/featurelearning_step.py
      ClusteringStep          pipeline/clustering/clustering_step.py
      EvaluationStep          pipeline/evaluation/evaluation_step.py
        -> concrete steps, e.g. Normalizer, VAE, HierarchicalClustering
"""

import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


# Where steps write when no output_path is given: <repo>/data/pipeline.
# It is anchored to this file, not to the working directory, so a notebook
# in notebooks/ and a script at the repo root write to the same place.
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parents[2] / "data" / "pipeline"


class Step(ABC):
    """Abstract base class of every stage in the pipeline.

    Concrete steps implement `run()`. Most of them take no argument there:
    they read `self.input_path` (or images handed over in memory) instead.
    """

    def __init__(
        self,
        input_path: str,
        output_path: str | None = None,
        save_results: bool = True,
        name: str | None = None,
        delete_input_data: bool = False,
        run_id = "default_run",
    ):
        """
        Args:
            input_path: what the step reads: a file, a folder, or a
                step-specific value such as a CSV path.
            output_path: where the step writes. If omitted, a folder is
                created automatically (see below).
            save_results: whether the step should write its output. Each
                step decides itself how to honor this.
            name: readable name for logs; defaults to the class name.
            delete_input_data: if True, the step's `run()` may call
                `delete_input_data()` to empty `input_path` when it is done.
            run_id: name of the experiment run. All steps of one run should
                share it, so their outputs end up in one folder:
                data/pipeline/<run_id>/<stage>/<step>/, e.g.
                data/pipeline/my_run/PreprocessingStep/Normalizer/.
        """
        self.input_path = input_path
        self.save_results = save_results

        if output_path:
            self.output_path = output_path
        else:
            default_output = (
                DEFAULT_OUTPUT_ROOT
                / run_id
                / self._stage_name()
                / self.__class__.__name__
            )
            default_output.mkdir(parents=True, exist_ok=True)
            self.output_path = str(default_output)

        self.name = name or self.__class__.__name__
        self.delete_input_data_flag = delete_input_data
        print("Output path for step", self.name, ":", self.output_path)


    def _stage_name(self) -> str:
        """The stage this step belongs to, e.g. 'PreprocessingStep' for a
        Normalizer: the first class in the inheritance chain (MRO) that
        derives from `Step` directly. A step deriving from `Step` itself
        uses its own class name."""
        for cls in self.__class__.__mro__[1:]:
            if Step in cls.__bases__:
                return cls.__name__
        return self.__class__.__name__

    @abstractmethod
    def run(self, data: Any) -> Any:
        """Do the step's work and return its result (usually the output
        folder or a table). The stage base classes narrow this signature."""
        raise NotImplementedError

    def delete_input_data(self) -> None:
        """Delete everything inside `input_path` (a file is deleted, a
        folder is emptied but kept)."""
        if not self.input_path:
            return

        input_dir = Path(self.input_path)
        if not input_dir.exists():
            return

        if input_dir.is_file():
            input_dir.unlink()
            return

        for child in input_dir.iterdir():
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)
            else:
                child.unlink()

    def __call__(self, data: Any) -> Path:
        # step(data) is a shortcut for step.run(data).
        return self.run(data)

    def __repr__(self) -> str:
        return f"<{self.name}>"
