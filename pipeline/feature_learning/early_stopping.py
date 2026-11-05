"""Early stopping shared by every training loop (`VAE`, `SimCLR`, `BYOL`,
`VAEClassifier`): stop once a smoothed metric has not improved for several
epochs in a row, and restore the weights of the best epoch.

The unsupervised models watch their training loss (mode="min"), the
classifier its validation accuracy (mode="max").
"""


class EarlyStopping:
    """Tracks a smoothed metric across epochs and decides when to stop.

    The metric is smoothed with a trailing moving average over
    `smoothing_window` epochs before being compared against the best
    value seen so far, so a single noisy epoch can't trigger a stop (or
    reset the patience counter) on its own - only a real, sustained
    plateau does.
    """

    def __init__(self, patience=15, min_delta=0.0, smoothing_window=10, mode="min"):
        """
        Args:
            patience: stop after this many consecutive epochs with no
                improvement in the smoothed metric.
            min_delta: minimum change, in the "better" direction, needed
                to count as an improvement.
            smoothing_window: number of most recent epochs averaged
                together for the early-stopping signal.
            mode: 'min' if lower metric values are better (e.g. training
                loss), 'max' if higher is better (e.g. validation
                accuracy).
        """
        if mode not in {"min", "max"}:
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")

        self.patience = patience
        self.min_delta = min_delta
        self.smoothing_window = smoothing_window
        self.mode = mode

        self.history = []
        self.best_smoothed = float("inf") if mode == "min" else float("-inf")
        self.best_epoch = 0
        self.best_state_dict = None
        self.epochs_without_improvement = 0

    def step(self, value, epoch, model=None):
        """Record this epoch's raw metric value and report whether
        training should stop now.

        Args:
            value: this epoch's raw (unsmoothed) metric value.
            epoch: 1-indexed epoch number, recorded as `best_epoch` if
                this epoch turns out to be the best one so far.
            model: if given, its `state_dict()` is snapshotted (detached,
                moved to CPU) whenever this epoch improves on the best
                smoothed value seen so far, so `restore_best` can bring
                those weights back once training stops.

        Returns:
            (smoothed_value, should_stop): the metric's current
            trailing-moving-average value, and whether
            `epochs_without_improvement` has reached `patience`.
        """
        self.history.append(value)
        window = self.history[-self.smoothing_window:]
        smoothed = sum(window) / len(window)

        if self.mode == "min":
            improved = smoothed < self.best_smoothed - self.min_delta
        else:
            improved = smoothed > self.best_smoothed + self.min_delta

        if improved:
            self.best_smoothed = smoothed
            self.best_epoch = epoch
            self.epochs_without_improvement = 0
            if model is not None:
                self.best_state_dict = {
                    key: tensor.detach().cpu().clone()
                    for key, tensor in model.state_dict().items()
                }
        else:
            self.epochs_without_improvement += 1

        should_stop = self.epochs_without_improvement >= self.patience
        return smoothed, should_stop

    def restore_best(self, model):
        """Load the snapshotted best-epoch weights back into `model`.

        No-op if `step` was never called with a `model` argument (e.g.
        training stopped before any epoch completed).
        """
        if self.best_state_dict is not None:
            model.load_state_dict(self.best_state_dict)
