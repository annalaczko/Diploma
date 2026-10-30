"""DINOv3 as a pretrained feature extractor (`model.type: dinov3`), either
frozen or fine-tuned (chosen with `finetune` at construction).

The method of notebooks/sandbox.ipynb ("DINOv3" section): every image is
converted to RGB, prepared by the model's own Hugging Face image processor,
and embedded with the pretrained backbone; its CLS token is the image's
feature vector. With `finetune=False` (default) nothing is trained. With
`finetune=True`, `run(mode="train_model")` first fine-tunes part of the
backbone - this path is still a skeleton (see the TODOs in
`_setup_finetuning`, `finetune_model` and `_finetune_loss`).

The checkpoints are gated on Hugging Face: accept the license on the model
page (e.g. https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m)
and run `huggingface-cli login` (or set HF_TOKEN) once; after the first
download the weights come from the local cache.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image

from pipeline import tracking
from pipeline.feature_learning.early_stopping import EarlyStopping
from pipeline.feature_learning.featurelearning_step import FeatureLearningStep

DEFAULT_MODEL_NAME = "facebook/dinov3-vitb16-pretrain-lvd1689m"


def short_model_name(model_name: str) -> str:
    """'facebook/dinov3-vitb16-pretrain-lvd1689m' -> 'vitb16', for run names."""
    name = model_name.rsplit("/", 1)[-1]
    name = name.removeprefix("dinov3-")
    return name.split("-pretrain")[0]


class DINOv3(FeatureLearningStep):
    """Feature step embedding each image with a DINOv3 backbone (the CLS
    token of its last hidden state), frozen or fine-tuned.

    `run()` writes `features.csv` (filename + feature vector) to `output_path`.
    """

    def __init__(
        self,
        input_path: str | None = None,
        model_name: str = DEFAULT_MODEL_NAME,
        finetune: bool = False,
        device=None,
        output_path=None,
        name=None,
        run_id="default_run",
    ):
        """
        Args:
            input_path: folder of preprocessed images (or in-memory images
                via `set_images`).
            model_name: Hugging Face model id, e.g.
                facebook/dinov3-vits16-pretrain-lvd1689m (smaller/faster),
                facebook/dinov3-vitb16-pretrain-lvd1689m,
                facebook/dinov3-vitl16-pretrain-lvd1689m (larger/slower).
            finetune: False = frozen backbone, inference only;
                `run(mode="train_model")` is refused. True = the backbone
                (the parts `_setup_finetuning` unfreezes) is trained by
                `run(mode="train_model")` before features are extracted.
            device: torch device; None = cuda if available.
        """
        super().__init__(input_path=input_path or "", output_path=output_path, name=name, run_id=run_id)
        self.model_name = model_name
        self.finetune = finetune
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.hparams = {"model_name": model_name, "finetune": finetune}
        self._processor = None
        self._model = None

    def _load(self):
        """Load the processor and backbone on first use (transformers is
        only needed when this step actually runs). The backbone starts
        fully frozen; with `finetune`, `_setup_finetuning` unfreezes the
        parts to train."""
        if self._model is None:
            from transformers import AutoImageProcessor, AutoModel

            self._processor = AutoImageProcessor.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name).to(self.device)
            self._model.requires_grad_(False)
            self._model.eval()
            if self.finetune:
                self._setup_finetuning()
        return self._processor, self._model

    def run(
        self,
        mode="extract_features",
        batch_size=128,
        epochs=20,
        lr=1e-5,
        early_stopping_patience=5,
        early_stopping_min_delta=0.0,
        early_stopping_smoothing_window=3,
        **_ignored,
    ) -> pd.DataFrame:
        """Embed every image, after fine-tuning first if
        mode='train_model' (only with `finetune=True`). Other training
        arguments are accepted so the step can stand in for a learned
        model."""
        if mode not in {"extract_features", "train_model"}:
            raise ValueError("mode must be either 'extract_features' or 'train_model'")
        if mode == "train_model":
            if not self.finetune:
                raise ValueError(f"{self.name} has a frozen backbone: create it with finetune=True to train")
            self.finetune_model(
                epochs=epochs,
                lr=lr,
                batch_size=batch_size,
                early_stopping_patience=early_stopping_patience,
                early_stopping_min_delta=early_stopping_min_delta,
                early_stopping_smoothing_window=early_stopping_smoothing_window,
            )
        else:
            tracking.log_params({"feature_step": self.name})
            tracking.log_params(self.hparams, prefix=f"feature/{self.name}/")
        features = self.extract_features(batch_size=batch_size)
        tracking.log_metrics({"n_images": len(features)}, prefix="feature/")
        return features

    @torch.no_grad()
    def extract_features(self, batch_size=128) -> pd.DataFrame:
        """The CLS embedding of every image, as a 'filename' + 'features'
        row per image (same shape as `VAE.extract_features`'s output); also
        written to `<output_path>/features.csv`."""
        processor, model = self._load()
        if self.finetune:
            model.eval()  # fine-tuning leaves it in train mode
        inputs = self._collect_inputs()
        # In-memory images (InMemoryImages) or image paths from disk.
        if hasattr(inputs, "names"):
            filenames = list(inputs.names)
            load = lambda i: inputs[i]  # noqa: E731
        else:
            filenames = [path.name for path in inputs]
            load = lambda i: _open(inputs[i])  # noqa: E731

        all_features = []
        for start in range(0, len(filenames), batch_size):
            # DINOv3 expects 3-channel input: the grayscale drawings become
            # RGB with all channels equal.
            images = [load(i).convert("RGB") for i in range(start, min(start + batch_size, len(filenames)))]
            batch = processor(images=images, return_tensors="pt").to(self.device)
            outputs = model(**batch)
            all_features.append(outputs.last_hidden_state[:, 0, :].float().cpu())

        features = torch.cat(all_features, dim=0).numpy()
        output_file = Path(self.output_path) / "features.csv"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            "filename": filenames,
            "features": [" ".join(f"{v:.6g}" for v in row) for row in features],
        }).to_csv(output_file, index=False)
        self._log_features(output_file)
        print(f"{self.name}: {len(filenames)} images, {features.shape[1]} features -> {output_file}")
        return pd.DataFrame({"filename": filenames, "features": list(features)})

    # ------------------------------------------------------------ fine-tuning
    # Skeleton: the structure is in place, the decisions marked TODO are not.

    def _setup_finetuning(self) -> None:
        """Unfreeze the parts of the (fully frozen) backbone to train, and
        build whatever the training objective needs on top of it."""
        # TODO: which parameters to train - e.g. the last N transformer
        #   blocks + the final layernorm (self._model.layer[-N:], self._model.norm;
        #   check the attribute names with print(self._model)), or LoRA.
        # TODO: a head for the objective (projection head for a contrastive
        #   loss, classifier for supervised), on self.device. Its parameters
        #   must also go to the optimizer in `finetune_model`.
        raise NotImplementedError("DINOv3 fine-tuning: choose the parameters to unfreeze")

    def _trainable_parameters(self):
        """Everything the optimizer updates."""
        # TODO: add the head's parameters once `_setup_finetuning` builds one.
        return [p for p in self._model.parameters() if p.requires_grad]

    def _train_loader(self, batch_size):
        """The training batches."""
        # TODO: dataset over self._collect_inputs(): .convert("RGB"),
        #   augmentation (e.g. SimCLRDataset + build_augmentation_transform
        #   for two views; apply self.train_transform if the pipeline set one),
        #   then the processor's resize/normalization as tensors
        #   (self._processor.image_mean / image_std).
        raise NotImplementedError("DINOv3 fine-tuning: build the training data loader")

    def _finetune_loss(self, batch) -> torch.Tensor:
        """The training objective on one batch from `_train_loader`."""
        # TODO: the objective - e.g. NT-Xent on two views (as SimCLR._nt_xent_loss),
        #   DINO-style self-distillation, or supervised on the TypeRev labels.
        raise NotImplementedError("DINOv3 fine-tuning: choose the loss")

    def finetune_model(
        self,
        epochs=20,
        lr=1e-5,
        batch_size=128,
        early_stopping_patience=5,
        early_stopping_min_delta=0.0,
        early_stopping_smoothing_window=3,
    ) -> None:
        """Fine-tune the unfrozen parts of the backbone with
        `_finetune_loss`, watching the smoothed training loss with
        `EarlyStopping` and restoring the best epoch's weights at the end.
        Features are extracted afterwards by `run`."""
        if not self.finetune:
            raise ValueError(f"{self.name} has a frozen backbone: create it with finetune=True to train")
        self._log_training_start(**{k: v for k, v in locals().items() if k != "self"})
        _, model = self._load()
        loader = self._train_loader(batch_size)
        optimizer = torch.optim.AdamW(self._trainable_parameters(), lr=lr)
        early_stopping = EarlyStopping(
            patience=early_stopping_patience,
            min_delta=early_stopping_min_delta,
            smoothing_window=early_stopping_smoothing_window,
            mode="min",
        )

        model.train()
        for epoch in range(epochs):
            train_loss = 0.0
            for batch in loader:
                optimizer.zero_grad()
                loss = self._finetune_loss(batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            average_loss = train_loss / len(loader)
            # TODO: snapshot the head too (if any), not only the backbone.
            smoothed_loss, should_stop = early_stopping.step(average_loss, epoch + 1, model=model)
            self._log_epoch(epoch + 1, loss=average_loss, smoothed_loss=smoothed_loss,
                            lr=optimizer.param_groups[0]["lr"])
            print(f"[epoch {epoch + 1}/{epochs}] loss: {average_loss:.3f} | smoothed: {smoothed_loss:.3f}")
            if should_stop:
                print(f"Early stopping at epoch {epoch + 1} "
                      f"(best smoothed={early_stopping.best_smoothed:.3f} @ epoch {early_stopping.best_epoch})")
                break

        self._log_training_end(early_stopping, epoch + 1)
        early_stopping.restore_best(model)
        # TODO: save the fine-tuned weights to <output_path>/model.pt
        #   (and model.yaml), like the other learned models.


def _open(path) -> Image.Image:
    with Image.open(path) as image:
        image.load()
        return image
