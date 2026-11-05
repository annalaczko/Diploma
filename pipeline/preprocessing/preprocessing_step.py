"""Abstract base class of the preprocessing steps (BackgroundEraser,
Normalizer, Repair, Vectorizer, AxisRenderer, Augmentor)."""

from abc import abstractmethod
from pathlib import Path

from pipeline import tracking
from pipeline.preprocessing.dataset import ImageRecord
from pipeline.steps.step import Step


class PreprocessingStep(Step):
    """Base class of the steps that clean and prepare the images.

    The steps are chained one after another, in one of two modes:
      - disk mode: `run()` reads every image under `self.input_path` and
        writes the results to `self.output_path`. That folder is then the
        next step's `input_path`.
      - in-memory mode: `process_batch(records)` takes a list of
        `ImageRecord`s (name + image) and returns the processed list;
        nothing is written (see `pipeline.pipeline.Pipeline`).
    Both modes call the same per-image method of the step, so they always
    produce the same images.

    Extra outputs (debug overlays, SVGs) go to `side_output_dir(kind)`, a
    `_<kind>` folder inside the output folder. Image collection skips `_`
    folders, so the next step never mistakes them for input.
    """

    def side_output_dir(self, kind: str) -> Path:
        """`<output folder>/_<kind>`, e.g. `_debug` or `_svg`."""
        return Path(self.output_path) / f"_{kind}"

    def __init__(
        self,
        input_path: str,
        output_path: str | None = None,
        name: str | None = None,
        delete_input_data: bool = False,
        save_results: bool = True,
        run_id: str | None = None,
    ):
        # The arguments are explained in Step.__init__.
        super().__init__(
            input_path=input_path,
            output_path=output_path,
            name=name,
            delete_input_data=delete_input_data,
            save_results=save_results,
            run_id=run_id,
        )

    @abstractmethod
    def run(self) -> Path:
        raise NotImplementedError

    def process_batch(self, records: list[ImageRecord]) -> list[ImageRecord]:
        """In-memory counterpart of `run()`: process one batch of images
        and return the results (images that fail may be dropped)."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support in-memory (batch) mode"
        )

    def begin_run(self) -> None:
        """Reset per-run bookkeeping (e.g. lists of skipped images) before
        a pass over the dataset. `Pipeline` calls it before in-memory
        preprocessing; disk-mode `run()`s call it themselves."""

    def failure_lists(self) -> dict[str, list[str]] | None:
        """Names of the images this step skipped or only partly processed
        in the current run, by reason (e.g. {'no_shard': [...]}), for
        `log_summary`. None (the default) means the step reports nothing."""
        return None

    def log_summary(self) -> None:
        """Report the run to the active MLflow run (see `pipeline.tracking`)
        after a pass over the dataset, for steps with `failure_lists`: the
        step's settings as params, `n_images` (from `self._n_seen`) and per
        reason `<reason>_count` as metrics, and the names as `<reason>.txt`
        (one per line), all under `preprocessing/<step name>/`. Counterpart
        of `begin_run`."""
        failures = self.failure_lists()
        if failures is None:
            return
        prefix = f"preprocessing/{self.name}"
        tracking.log_params(tracking.params_of(self), prefix=f"{prefix}/")
        metrics = {"n_images": getattr(self, "_n_seen", None)}
        for reason, names in failures.items():
            metrics[f"{reason}_count"] = len(names)
            tracking.log_lines(names, f"{prefix}/{reason}.txt")
            if names:
                print(f"{self.name}: {len(names)} image(s) with {reason}")
        tracking.log_metrics(metrics, prefix=f"{prefix}/")
