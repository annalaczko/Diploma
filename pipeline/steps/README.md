# `pipeline.steps`

The abstract `Step` base class shared by every concrete stage in the
pipeline (preprocessing, feature learning, clustering, evaluation).

## `Step`

```python
from pipeline.steps import Step
```

`Step` is an `ABC` (abstract base class). Every concrete step must implement
`run()`; subclasses add their own configuration via `__init__`.

### `__init__`

```python
Step(
    input_path: str,
    output_path: str | None = None,
    save_results: bool = True,
    name: str | None = None,
    delete_input_data: bool = False,
    run_id="default_run",
)
```

| Argument | Meaning |
|---|---|
| `input_path` | Where this step reads its input from — a file, directory, or step-specific value (e.g. a CSV path for clustering). |
| `output_path` | Where this step writes its output. If omitted, a directory is auto-derived (see below) and created immediately. |
| `save_results` | Whether the step should persist its output to `output_path`. This is *not enforced* by `Step` itself — each concrete step decides for itself whether/how to honor it. |
| `name` | Human-readable step name, used in logging and `__repr__`. Defaults to the concrete class name. |
| `delete_input_data` | If `True`, calling `delete_input_data()` empties `input_path` after the step considers itself done. The concrete `run()` implementation must call it explicitly — it is not automatic. |
| `run_id` | Identifies this pipeline run; namespaces the auto-derived output directory so different runs/experiments don't overwrite each other's data. |

### Auto-derived output path

If `output_path` is not given, `Step.__init__` builds one and creates it on
disk immediately:

```
<repo>/data/pipeline/<run_id>/<stage_class>/<concrete_class>/
```

The root is `DEFAULT_OUTPUT_ROOT` in `pipeline/steps/step.py`. It is
anchored to the repository, not the working directory, so a notebook in
`notebooks/` and a script run from the repo root write to the same place.
Tests redirect it to a temp folder (`tests/conftest.py`).

`<stage_class>` is the step's stage base class: the first strict `Step`
subclass in its MRO (e.g. `PreprocessingStep` for `Normalizer`,
`FeatureLearningStep` for `VAE` even though `VAE` lists `nn.Module` first).
Outputs from different kinds of steps therefore land in separate, predictable
subtrees. If a class subclasses `Step` directly, the concrete class name is
used for both segments.

### `run(data)`

```python
@abstractmethod
def run(self, data: Any) -> Any: ...
```

Executes the step and returns its output — in most cases the path to the
output data written by the step. Every concrete subclass overrides this;
some (e.g. `ClusteringStep`, `FeatureLearningStep`) narrow the signature to
take no `data` argument, since they read their input from `self.input_path`
configured at construction time instead. See each stage's own README for
its exact `run()` contract.

### `delete_input_data()`

Deletes every file and subfolder inside `self.input_path`, but preserves the
directory itself. No-ops if `input_path` is falsy or doesn't exist. Useful
for pipelines that shouldn't keep large intermediate outputs from earlier
stages around once a later stage has consumed them — opt-in per step via
`delete_input_data=True` at construction.

### `__call__` / `__repr__`

- `step_instance(data)` is shorthand for `step_instance.run(data)`.
- `repr(step_instance)` renders as `<step.name>`.
