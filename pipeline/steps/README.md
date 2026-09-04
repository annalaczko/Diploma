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
../data/pipeline/<run_id>/<parent_class>/<concrete_class>/
```

`<parent_class>` is the step's immediate abstract base (e.g.
`PreprocessingStep` for `Normalizer`), so outputs from different kinds of
steps land in separate, predictable subtrees. If a class has no such base in
its MRO (e.g. `Step` is subclassed directly, with nothing in between), the
concrete class name is used for both segments.

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
