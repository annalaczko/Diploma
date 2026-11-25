"""DINOv3 as a pretrained feature extractor (`model.type: dinov3`), either
frozen or fine-tuned (chosen with `finetune` at construction).

The method of notebooks/sandbox.ipynb ("DINOv3" section): every image is
converted to RGB, prepared by the model's own Hugging Face image processor,
and embedded with the pretrained backbone; its CLS token is the image's
feature vector. With `finetune=False` (default) nothing is trained. With
`finetune=True`, `run(mode="train_model")` first fine-tunes the last
transformer blocks self-supervised on two random views of every image,
with one of two objectives (`finetune_method`):
  - "nt_xent" (SimCLR-style): a projection head on the CLS token and the
    NT-Xent contrastive loss (`_nt_xent`), the other images of the batch
    being the negatives;
  - "dino" (DINO self-distillation, the objective DINOv3 was pretrained
    with, in its basic form: CLS token only, two views, centering): a
    `DINOHead` on the student, and a teacher, an EMA copy of the backbone
    and head, whose sharpened, centered predictions for one view the
    student learns to match from the other (`_dino_loss`). No negatives;
    the features then come from the teacher.
No labels are used, so it can train on the train split without touching
the TypeRev labels the evaluation uses.

"dino" has three optional additions from DINOv2/DINOv3's recipe (Oquab et
al., 2023), each switched on by its own parameter and off by default, so
they can be compared one at a time:
  - multi-crop (`local_crops_number > 0`): besides the two full views the
    student also sees small local crops (`InkedRandomResizedCrop`) and has
    to match the teacher's prediction for the whole drawing from a piece of
    it, e.g. only the rim;
  - iBOT (`ibot_loss_weight > 0`): patches of the student's global views
    are masked, and a second head on the patch tokens has to predict the
    teacher's (unmasked) patch tokens there (`_ibot_loss`), so the local
    structure of the lines is learned too;
  - KoLeo (`koleo_loss_weight > 0`): a regularizer that pushes every
    image's CLS token away from its nearest neighbour in the batch
    (`_koleo_loss`), spreading the features out.
Independent of the method, `layerwise_lr_decay < 1` gives the earlier
unfrozen blocks a smaller learning rate than the later ones.

The checkpoints are gated on Hugging Face: accept the license on the model
page (e.g. https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m)
and run `huggingface-cli login` (or set HF_TOKEN) once; after the first
download the weights come from the local cache.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torchvision import transforms

from pipeline import tracking
from pipeline.feature_learning.early_stopping import EarlyStopping
from pipeline.feature_learning.featurelearning_step import FeatureLearningStep
from pipeline.preprocessing.dataset import image_entries, open_image, MultiCropDataset, TwoViewDataset
from pipeline.preprocessing.augmentation import build_line_drawing_augmentation, InkedRandomResizedCrop

DEFAULT_MODEL_NAME = "facebook/dinov3-vitb16-pretrain-lvd1689m"


def random_block_masks(n_images, grid_h, grid_w, ratio_range=(0.1, 0.5), probability=0.5) -> torch.Tensor:
    """iBOT masks: which patches of each image the student sees masked.

    Each image is masked with `probability` (the others keep all patches,
    as in DINOv2, so the student also sees whole images). A masked image
    loses a fraction of its patches drawn uniformly from `ratio_range`, in
    rectangular blocks rather than single patches (BEiT-style): a single
    missing patch can be guessed from its neighbours, a missing block
    has to be inferred from the rest of the drawing.

    Args:
        n_images: number of images (rows of the result).
        grid_h, grid_w: the patch grid of an image (its size / patch size).
        ratio_range: (min, max) fraction of patches masked in a masked image.
        probability: chance that an image is masked at all.

    Returns:
        (n_images, grid_h * grid_w) bool tensor, True = masked, in the
        order of the backbone's patch tokens (row by row).
    """
    masks = torch.zeros(n_images, grid_h, grid_w, dtype=torch.bool)
    n_patches = grid_h * grid_w
    for i in range(n_images):
        if float(torch.rand(())) >= probability:
            continue
        low, high = ratio_range
        target = int(round((low + (high - low) * float(torch.rand(()))) * n_patches))
        for _ in range(100):   # enough blocks to reach the target; the cap only guards odd grids
            remaining = target - int(masks[i].sum())
            if remaining <= 0:
                break
            area = min(4, remaining) + (remaining - min(4, remaining)) * float(torch.rand(()))
            aspect = math.exp(math.log(0.3) + (math.log(1 / 0.3) - math.log(0.3)) * float(torch.rand(())))
            height = max(1, min(grid_h, int(round(math.sqrt(area * aspect)))))
            width = max(1, min(grid_w, int(round(math.sqrt(area / aspect)))))
            top = int(torch.randint(0, grid_h - height + 1, ()))
            left = int(torch.randint(0, grid_w - width + 1, ()))
            masks[i, top:top + height, left:left + width] = True
    return masks.flatten(1)


def _mask_token_of(backbone) -> torch.nn.Module:
    """A module holding the backbone's `mask_token` (the same Parameter,
    not a copy), so it can sit in a ModuleDict with the trained parts:
    saved with them, given to the optimizer and paired with the teacher's
    for the EMA update."""
    holder = torch.nn.Module()
    holder.mask_token = backbone.embeddings.mask_token
    return holder


def short_model_name(model_name: str) -> str:
    """'facebook/dinov3-vitb16-pretrain-lvd1689m' -> 'vitb16', for run names."""
    name = model_name.rsplit("/", 1)[-1]
    name = name.removeprefix("dinov3-")
    return name.split("-pretrain")[0]


class DINOHead(torch.nn.Module):
    """The head on the CLS token for `finetune_method="dino"`, used only
    in fine-tuning (the features stay the CLS token).

    An MLP maps the CLS token to a small bottleneck vector, which is
    L2-normalized and compared with `num_prototypes` learned prototype
    vectors (the rows of `prototypes.weight`, also normalized). The output,
    one logit per prototype, is therefore a cosine similarity in [-1, 1]:
    dividing it by the DINO temperatures gives logits of a known range,
    which unnormalized vectors would not (Caron et al., DINO, 2021, do the
    same with weight normalization).
    """

    def __init__(self, input_dim: int, num_prototypes: int, hidden_dim: int = 2048, bottleneck_dim: int = 256):
        """
        Args:
            input_dim: size of the CLS token (the backbone's hidden size).
            num_prototypes: number of prototypes, the size of the output.
            hidden_dim: width of the MLP's hidden layers.
            bottleneck_dim: size of the vector compared with the prototypes.
        """
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, bottleneck_dim))
        self.prototypes = torch.nn.Linear(bottleneck_dim, num_prototypes, bias=False)

    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1)
        return F.linear(x, F.normalize(self.prototypes.weight, dim=1))


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
        unfreeze_blocks: int = 2,
        projection_dim: int = 128,
        temperature: float = 0.1,
        finetune_method: str = "nt_xent",
        num_prototypes: int = 4096,
        teacher_momentum: float = 0.996,
        teacher_temp: float = 0.04,
        student_temp: float = 0.1,
        center_momentum: float = 0.9,
        local_crops_number: int = 0,
        local_crops_size: int = 96,
        local_crops_scale: tuple = (0.05, 0.32),
        local_crops_min_ink: float = 0.005,
        ibot_loss_weight: float = 0.0,
        ibot_num_prototypes: int | None = None,
        ibot_mask_ratio: tuple = (0.1, 0.5),
        ibot_mask_probability: float = 0.5,
        koleo_loss_weight: float = 0.0,
        layerwise_lr_decay: float = 1.0,
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
            unfreeze_blocks: with `finetune`, how many of the last
                transformer blocks are trained (plus the final layernorm);
                the earlier ones stay frozen.
            projection_dim: with `finetune`, the output size of the
                projection head, the vectors the contrastive loss is
                computed on (not the features, see `_setup_finetuning`).
            temperature: with `finetune`, the NT-Xent temperature: lower
                values sharpen the softmax, punishing negatives that are
                almost as similar as the positive more.
            finetune_method: with `finetune`, the self-supervised objective:
                "nt_xent" (default) = SimCLR-style contrastive loss with
                the other images of the batch as negatives; "dino" =
                DINO self-distillation, a student trained to match an EMA
                teacher on the other view, without negatives (keeps a
                second copy of the backbone, so it needs more memory).
                `projection_dim` and `temperature` only apply to
                "nt_xent", the five parameters below only to "dino".
            num_prototypes: the size of the DINO head's output: every view
                is softly assigned to this many learned prototypes
                (pseudo-classes). More gives finer groups but needs more
                memory and data.
            teacher_momentum: after every step, `teacher = m * teacher +
                (1 - m) * student` (like BYOL's `tau`); close to 1, the
                teacher averages the student over the last ~1/(1 - m)
                steps, a stable target.
            teacher_temp: divides the teacher's logits before its softmax;
                low values make its targets sharp (almost one-hot), which
                together with centering prevents collapse.
            student_temp: divides the student's logits before its softmax;
                higher than `teacher_temp`, so the student has to become
                confident to match the sharp targets.
            center_momentum: momentum of `center`, the moving average of
                the teacher's logits, subtracted before its softmax so it
                cannot put every image on the same prototype.

            The parameters below are optional additions to "dino" (an
            error with "nt_xent"), each off by default:

            local_crops_number: multi-crop: how many small local crops of
                every image the student sees besides the two global views
                (0 = off). The teacher sees only the global views; the
                student matches its prediction from every other view and
                crop. DINOv2 uses 8; each costs about a quarter of a global
                view at the default sizes.
            local_crops_size: side of a local crop in pixels (a multiple
                of the patch size, 16): 96 = 6 x 6 patches.
            local_crops_scale: (min, max) fraction of the image's area a
                local crop covers, before it is resized to
                `local_crops_size`.
            local_crops_min_ink: a local crop is redrawn (up to 10 times)
                while less than this fraction of its pixels is dark, so it
                is not just blank paper (`InkedRandomResizedCrop`; 0 = no
                check).
            ibot_loss_weight: iBOT: the weight of the masked-patch loss
                added to the DINO loss (0 = off; DINOv2 uses 1.0). Also
                makes the backbone's `mask_token` trainable: it is all
                zeros in the released checkpoints.
            ibot_num_prototypes: the size of the iBOT head's output, a
                separate head on the patch tokens; None = `num_prototypes`.
            ibot_mask_ratio: (min, max) fraction of a masked view's patches
                that are masked (`random_block_masks`).
            ibot_mask_probability: the chance that a global view is masked
                at all; the rest are seen whole.
            koleo_loss_weight: KoLeo: the weight of the regularizer that
                spreads the student's CLS tokens in the batch (0 = off;
                DINOv2 uses 0.1).
            layerwise_lr_decay: with any `finetune_method`, block i of the
                L blocks gets lr * decay**(L - i) (the last block lr *
                decay, the norm and heads the full lr), so the earlier,
                more general blocks change less. 1.0 = one lr for all
                (off); DINOv2 uses 0.9.
        """
        super().__init__(input_path=input_path or "", output_path=output_path, name=name, run_id=run_id)
        self.model_name = model_name
        self.finetune = finetune
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.hparams = {"model_name": model_name, "finetune": finetune}
        if self.finetune:
            self.hparams.update({
                "unfreeze_blocks": unfreeze_blocks,
                "finetune_method": finetune_method,
            })
            match finetune_method:
                case "nt_xent":
                    self.hparams.update({
                        "projection_dim": projection_dim,
                        "temperature": temperature,})
                case "dino":
                    self.hparams.update({
                        "num_prototypes": num_prototypes,
                        "teacher_momentum": teacher_momentum,
                        "teacher_temp": teacher_temp,
                        "student_temp": student_temp,
                        "center_momentum": center_momentum,})
                    # The additions are logged only when switched on, so
                    # runs without them keep the params they had before.
                    if local_crops_number > 0:
                        self.hparams.update({
                            "local_crops_number": local_crops_number,
                            "local_crops_size": local_crops_size,
                            "local_crops_scale": list(local_crops_scale),
                            "local_crops_min_ink": local_crops_min_ink,})
                    if ibot_loss_weight > 0:
                        self.hparams.update({
                            "ibot_loss_weight": ibot_loss_weight,
                            "ibot_num_prototypes": ibot_num_prototypes or num_prototypes,
                            "ibot_mask_ratio": list(ibot_mask_ratio),
                            "ibot_mask_probability": ibot_mask_probability,})
                    if koleo_loss_weight > 0:
                        self.hparams["koleo_loss_weight"] = koleo_loss_weight
                case _:
                    raise ValueError(f"Unknown finetune_method: {finetune_method}")
            if finetune_method != "dino":
                extras = {"local_crops_number": local_crops_number, "ibot_loss_weight": ibot_loss_weight,
                          "koleo_loss_weight": koleo_loss_weight}
                switched_on = [name for name, value in extras.items() if value > 0]
                if switched_on:
                    raise ValueError(f"{', '.join(switched_on)} only work with finetune_method='dino'")
            if not 0 < layerwise_lr_decay <= 1:
                raise ValueError(f"layerwise_lr_decay must be in (0, 1], got {layerwise_lr_decay}")
            if layerwise_lr_decay < 1:
                self.hparams["layerwise_lr_decay"] = layerwise_lr_decay

        self._processor = None
        self._model = None
        self._head = None  # head on the CLS token, built by `_setup_finetuning` (only with finetune)
        self.unfreeze_blocks = unfreeze_blocks
        self.projection_dim = projection_dim
        self.temperature = temperature
        self.finetune_method = finetune_method
        self.num_prototypes = num_prototypes
        self.teacher_momentum = teacher_momentum
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.local_crops_number = local_crops_number
        self.local_crops_size = local_crops_size
        self.local_crops_scale = tuple(local_crops_scale)
        self.local_crops_min_ink = local_crops_min_ink
        self.ibot_loss_weight = ibot_loss_weight
        self.ibot_num_prototypes = ibot_num_prototypes or num_prototypes
        self.ibot_mask_ratio = tuple(ibot_mask_ratio)
        self.ibot_mask_probability = ibot_mask_probability
        self.koleo_loss_weight = koleo_loss_weight
        self.layerwise_lr_decay = layerwise_lr_decay

        # Only with finetune_method="dino", built by `_setup_finetuning`:
        self._teacher = None  # EMA copy of the backbone
        self._teacher_head = None  # EMA copy of the head
        self._center = None  # moving average of the teacher's logits (a buffer of `_teacher_head`)
        self._dino_stats = None  # this epoch's teacher-target statistics, see `_reset_dino_stats`
        # Only with ibot_loss_weight > 0:
        self._ibot_head = None  # head on the patch tokens
        self._teacher_ibot_head = None  # its EMA copy
        self._ibot_center = None  # moving average of the teacher's patch logits (a buffer of `_teacher_ibot_head`)
        # The last batch's loss terms as numbers, summed per epoch by `finetune_model`.
        self._loss_parts = {}

    @property
    def _ibot(self) -> bool:
        return self.finetune and self.finetune_method == "dino" and self.ibot_loss_weight > 0

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
        epochs=100,
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
        written to `<output_path>/features.csv`.

        With `finetune_method="dino"` the teacher's CLS token is used: it
        is an average of the student over training, which gives more
        stable features (as in the DINO paper)."""
        processor, model = self._load()
        if self._teacher is not None:
            model = self._teacher
        if self.finetune:
            model.eval()  # fine-tuning leaves it in train mode
        # In-memory images or image paths from disk, read the same way.
        entries, filenames = image_entries(self._collect_inputs())

        all_features = []
        for start in range(0, len(filenames), batch_size):
            # DINOv3 expects 3-channel input: the grayscale drawings become
            # RGB with all channels equal.
            images = [open_image(entries, i).convert("RGB")
                      for i in range(start, min(start + batch_size, len(filenames)))]
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

    def _setup_finetuning(self) -> None:
        """Unfreeze the last `unfreeze_blocks` transformer blocks and the
        final layernorm of the (fully frozen) backbone, and build the
        projection head the contrastive loss is computed on.

        The head (a small MLP, CLS token h -> z) only serves training:
        the loss pulls the two views' z together, so z has to discard
        whatever differs between the views (crop, brightness, ...). With
        the head in between, most of that discarding happens in the head
        and h keeps more information (Chen et al., SimCLR, 2020). The
        features stay h, the CLS token; the head is not used afterwards.

        With `finetune_method="dino"` the head is a `DINOHead`, and the
        backbone and head are copied into a teacher. The teacher gets no
        gradients: after every step it moves a little toward the student
        (EMA, `teacher_momentum`), so it is a slowly changing average of
        it. `center` is the moving average of the teacher's logits, which
        the loss subtracts from them. The copy is made after unfreezing,
        so student and teacher start from the same weights.
        """
        blocks = self._model.model.layer

        if self.unfreeze_blocks <= 0:
            raise ValueError(f"unfreeze_blocks must be > 0, got {self.unfreeze_blocks}")

        if self.unfreeze_blocks > len(blocks):
            raise ValueError(f"unfreeze_blocks={self.unfreeze_blocks} is larger than the number of transformer blocks "
                             f"in the backbone ({len(blocks)}). Reduce unfreeze_blocks or choose a larger backbone.")

        trainable_blocks = blocks[-self.unfreeze_blocks:]

        trainable_blocks.requires_grad_(True)
        self._model.norm.requires_grad_(True)

        match self.finetune_method:
            case "nt_xent":
                hidden_size = self._model.config.hidden_size
                linear_head = torch.nn.Linear(hidden_size, hidden_size)
                relu = torch.nn.ReLU()
                linear_head2 = torch.nn.Linear(hidden_size, self.projection_dim)
                self._head = torch.nn.Sequential(linear_head, relu, linear_head2).to(self.device)

            case "dino":
                hidden_size = self._model.config.hidden_size
                self._head = DINOHead(input_dim=hidden_size, num_prototypes=self.num_prototypes).to(self.device)
                if self._ibot:
                    # The masked patches are replaced by `mask_token`, which the
                    # released checkpoints leave at zero: it has to be learned.
                    self._model.embeddings.mask_token.requires_grad_(True)
                    self._ibot_head = DINOHead(input_dim=hidden_size,
                                               num_prototypes=self.ibot_num_prototypes).to(self.device)
                self._teacher = copy.deepcopy(self._model)
                self._teacher.requires_grad_(False)
                self._teacher_head = copy.deepcopy(self._head)
                self._teacher_head.requires_grad_(False)
                self._teacher_head.register_buffer("center", torch.zeros(1, self.num_prototypes, device=self.device))
                self._center = self._teacher_head.center   # same tensor, a shorter name
                if self._ibot:
                    self._teacher_ibot_head = copy.deepcopy(self._ibot_head)
                    self._teacher_ibot_head.requires_grad_(False)
                    self._teacher_ibot_head.register_buffer(
                        "center", torch.zeros(1, self.ibot_num_prototypes, device=self.device))
                    self._ibot_center = self._teacher_ibot_head.center

    def _student_modules(self) -> torch.nn.ModuleDict:
        """The parts the optimizer trains: the unfrozen blocks, the final
        norm and the head, with iBOT also its head and `mask_token`. They
        are shared with the backbone, not copied, so loading a state_dict
        into this updates the backbone."""
        modules = {
            "blocks": self._model.model.layer[-self.unfreeze_blocks:],
            "norm": self._model.norm,
            "head": self._head,
        }
        if self._ibot:
            modules["ibot_head"] = self._ibot_head
            modules["mask_token"] = _mask_token_of(self._model)
        return torch.nn.ModuleDict(modules)

    def _teacher_modules(self) -> torch.nn.ModuleDict:
        """The teacher's copies of the same parts (only with "dino"), in
        the same order, so their parameters pair up with the student's.
        The teacher's heads also hold their `center` as a buffer. (The
        teacher never sees masked views, so its `mask_token` is unused;
        it is kept only so the two lists pair up.)"""
        modules = {
            "blocks": self._teacher.model.layer[-self.unfreeze_blocks:],
            "norm": self._teacher.norm,
            "head": self._teacher_head,
        }
        if self._ibot:
            modules["ibot_head"] = self._teacher_ibot_head
            modules["mask_token"] = _mask_token_of(self._teacher)
        return torch.nn.ModuleDict(modules)

    def _finetuned_modules(self) -> torch.nn.ModuleDict:
        """What is saved (`state_dict`) and restored by early stopping:
        the student's parts, and with "dino" also the teacher's (with its
        `center`), since the features come from the teacher and training
        can only be continued with both. With "nt_xent" the keys stay
        those of the student's parts alone, so earlier model.pt files
        still load."""
        match self.finetune_method:
            case "nt_xent":
                return self._student_modules()
            case "dino":
                return torch.nn.ModuleDict({
                    "student": self._student_modules(),
                    "teacher": self._teacher_modules(),
                })

    def state_dict(self) -> dict:
        """The fine-tuned weights (what `_finetuned_modules` holds), as
        `run_experiment` saves them to model.pt. The frozen rest of the
        backbone is not included: it comes from `model_name` again."""
        self._load()
        return self._finetuned_modules().state_dict()

    def load_state_dict(self, state_dict) -> None:
        """Load weights saved by `state_dict` (model.pt) into the backbone."""
        self._load()
        self._finetuned_modules().load_state_dict(state_dict)

    def _trainable_parameters(self):
        """Everything the optimizer updates: the student only (the teacher
        is moved by `_update_teacher`)."""
        return list(self._student_modules().parameters())

    def _param_groups(self, lr) -> list[dict]:
        """The optimizer's parameter groups with layer-wise learning-rate
        decay (`layerwise_lr_decay`, as DINOv2's `get_vit_lr_decay_rate`):
        block i (0-based) of the backbone's L blocks gets lr * decay**(L -
        i), the final norm and the heads the full lr, and `mask_token`,
        which sits before the first block, lr * decay**(L + 1). With decay
        1 this is a single group with `lr`.

        Returns:
            A list of {"params", "lr"} dicts for torch.optim.
        """
        if self.layerwise_lr_decay == 1:
            return [{"params": self._trainable_parameters(), "lr": lr}]
        decay = self.layerwise_lr_decay
        n_blocks = len(self._model.model.layer)
        first_trained = n_blocks - self.unfreeze_blocks
        groups = [{"params": list(block.parameters()), "lr": lr * decay ** (n_blocks - (first_trained + k))}
                  for k, block in enumerate(self._model.model.layer[first_trained:])]
        modules = self._student_modules()
        for name in ("norm", "head", "ibot_head"):
            if name in modules:
                groups.append({"params": list(modules[name].parameters()), "lr": lr})
        if "mask_token" in modules:
            groups.append({"params": list(modules["mask_token"].parameters()), "lr": lr * decay ** (n_blocks + 1)})
        return groups

    def _train_loader(self, batch_size):
        """The training batches for `_finetune_loss`: (view1, view2) pairs,
        two independent random augmentations of the same images.

        The augmentation is `build_line_drawing_augmentation` (small
        rotation/shift/resize, elastic distortion, thicker lines, missing
        line pieces; no crop or mirror),
        resized to the processor's input size; then the images are prepared as the image processor does
        for `extract_features` (3 channels, [0, 1], its mean/std), so the
        backbone sees the same kind of input in training and extraction.
        The pipeline's train_transform (Augmentor) is ignored: these views
        are the augmentation.

        With multi-crop (`local_crops_number > 0`) every batch also has
        that many local crops (`MultiCropDataset`): the same augmentation,
        then an `InkedRandomResizedCrop` of `local_crops_scale` of the
        image resized to `local_crops_size`. The global views stay whole
        drawings, not crops as in DINO, so the teacher's targets still
        come from the whole sherd (its size can tell its type).
        """
        if self.train_transform is not None:
            print(
                f"Warning: {self.name} makes its own augmented views; "
                "the pipeline's train_transform (Augmentor) is ignored."
            )

        processor = self._processor
        size = processor.size["height"]

        view_transform = transforms.Compose([
            # shift/rotation/resize, distortion, line thickness, erased pieces; at the model's input size
            build_line_drawing_augmentation(output_size=size),
            # DINOv3 expects RGB: 3 equal channels, as .convert("RGB") in extract_features
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),  # [0, 1], the processor's rescaling
            transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
        ])

        if self.local_crops_number > 0:
            local_transform = transforms.Compose([
                build_line_drawing_augmentation(output_size=size),
                InkedRandomResizedCrop(self.local_crops_size, scale=self.local_crops_scale,
                                       min_ink=self.local_crops_min_ink),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
            ])
            dataset = MultiCropDataset(self._collect_inputs(), view_transform, local_transform,
                                       self.local_crops_number)
        else:
            dataset = TwoViewDataset(self._collect_inputs(), view_transform)

        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            # With "nt_xent" the other images of a batch are the loss's
            # negatives, and with "dino" the batch mean updates `center`:
            # a tiny last batch would be a poor sample for both.
            drop_last=True,
            num_workers=8,
            pin_memory=self.device.type == "cuda",
        )

    @staticmethod
    def _dino_loss(student_logits, teacher_logits, center, student_temp, teacher_temp) -> torch.Tensor:
        """DINO loss of a batch of N images seen in several views.

        There are no negatives: the teacher's probabilities for one global
        view are the target the student has to match from every other view,
        with cross-entropy. With two views that is both directions
        (teacher1 -> student2 and teacher2 -> student1); with multi-crop
        the local crops (seen only by the student) are matched to both
        global views too. The teacher's logits are centered (`center`
        subtracted) and sharpened (divided by the low `teacher_temp`)
        before the softmax; the student's are only divided by
        `student_temp`. The targets are detached: the teacher is not
        trained by the loss, only moved by EMA.

        Args:
            student_logits: list of (N, K) logits of the student's head
                (one per prototype), one per view: the global views first,
                in the same order as `teacher_logits`, then the local
                crops. Row i of every view is the same image.
            teacher_logits: list of (N, K) logits of the teacher's head
                for the global views.
            center: (1, K) moving average of the teacher's logits,
                subtracted from them before the softmax.
            student_temp: divides the student's logits before the softmax.
            teacher_temp: divides the teacher's logits before the softmax.

        Returns:
            The scalar loss, averaged over the batch and all (teacher view,
            other student view) pairs: log(K) when all distributions are
            uniform. It cannot go below the entropy of the teacher's
            targets (not 0), so its trend means more than its value.
        """
        targets = [F.softmax((t - center) / teacher_temp, dim=-1).detach() for t in teacher_logits]
        log_students = [F.log_softmax(s / student_temp, dim=-1) for s in student_logits]

        total, pairs = 0.0, 0
        for i, target in enumerate(targets):
            for j, log_student in enumerate(log_students):
                if i == j:   # the same global view: nothing to learn
                    continue
                total = total - torch.sum(target * log_student, dim=-1).mean()
                pairs += 1
        return total / pairs

    @staticmethod
    def _ibot_loss(student_logits, teacher_logits, weights, n_images, center, student_temp, teacher_temp) -> torch.Tensor:
        """iBOT loss: the DINO cross-entropy per masked patch (Zhou et al.,
        iBOT, 2022). For every patch the student saw masked, the teacher's
        centered, sharpened probabilities for that patch (it saw it) are
        the target of the student's prediction there.

        Args:
            student_logits: (M, K) logits of the student's iBOT head for
                the M masked patches of the batch.
            teacher_logits: (M, K) the teacher's iBOT-head logits for the
                same patches.
            weights: (M,) 1 / (number of masked patches of the patch's
                image), so every masked image counts the same however much
                of it is masked (as in DINOv2).
            n_images: number of images with at least one masked patch.
            center: (1, K) moving average of the teacher's patch logits.
            student_temp, teacher_temp: as in `_dino_loss`.

        Returns:
            The scalar loss, averaged over the masked images; log(K) when
            all distributions are uniform.
        """
        target = F.softmax((teacher_logits - center) / teacher_temp, dim=-1).detach()
        log_student = F.log_softmax(student_logits / student_temp, dim=-1)
        per_patch = -torch.sum(target * log_student, dim=-1)
        return (per_patch * weights).sum() / max(int(n_images), 1)

    @staticmethod
    def _koleo_loss(x, eps=1e-8) -> torch.Tensor:
        """KoLeo regularizer (Sablayrolles et al., 2019, as in DINOv2):
        -mean(log(distance of each vector to its nearest neighbour)), after
        L2 normalization. It is lowest when the batch is spread evenly
        over the sphere, so it keeps similar images from collapsing onto
        the same feature vector.

        Args:
            x: (N, D) the student's CLS tokens of one view.

        Returns:
            The scalar loss; large (towards -log(eps)) when two vectors
            coincide.
        """
        x = F.normalize(x, dim=-1, eps=eps)
        with torch.no_grad():
            # The nearest neighbour by cosine similarity, never itself.
            similarity = x @ x.T
            similarity.fill_diagonal_(-2.0)
            nearest = similarity.argmax(dim=1)
        distance = F.pairwise_distance(x, x[nearest], eps=eps)
        return -torch.log(distance + eps).mean()

    @torch.no_grad()
    def _update_center(self, teacher_logits, center=None) -> None:
        """Move `center`, the moving average of the teacher's logits, a
        step toward this batch's mean: center = m * center + (1 - m) *
        batch mean, with m = `center_momentum`.

        Called after the loss, so the loss uses the center of the earlier
        batches. It takes the raw teacher logits, not the centered ones:
        the center estimates the teacher's own average output.

        Args:
            teacher_logits: (2N, K) raw logits of the teacher's head for
                both views of the batch (or (M, K) patch logits for
                `_ibot_center`).
            center: the buffer to update; None = `_center`.
        """
        center = self._center if center is None else center
        batch_center = teacher_logits.mean(dim=0, keepdim=True)
        center.mul_(self.center_momentum).add_(batch_center, alpha=1 - self.center_momentum)

    @staticmethod
    def _nt_xent(z1, z2, temperature) -> torch.Tensor:
        """NT-Xent loss of a batch of N images seen in two views.

        Each of the 2N vectors has one positive, the other view of the same
        image, and 2N-2 negatives, the views of the other images. Each row
        of the cosine-similarity matrix is a classification over those
        2N-1 candidates, with cross-entropy, whose target is the positive.

        Args:
            z1, z2: (N, D) vectors of the two views, row i of both from
                the same image; normalized here.
            temperature: divides the similarities before the softmax.

        Returns:
            The scalar loss: log(2N-1) for random vectors, near 0 when
            every view is far closer to its positive than to the rest.
        """
        n = z1.size(0)
        z = torch.cat([z1, z2], dim=0)
        z = F.normalize(z, dim=1)

        similarity = z @ z.T / temperature

        self_mask = torch.eye(2 * n, dtype=torch.bool, device=z.device)
        similarity = similarity.masked_fill(self_mask, float("-inf"))

        targets = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)

        return F.cross_entropy(similarity, targets)

    def _finetune_loss(self, batch) -> torch.Tensor:
        """The `finetune_method` loss of one batch from `_train_loader`:
        (view1, view2), with multi-crop followed by the local crops. Both
        global views go through the backbone together, their CLS tokens
        through the head.

        "nt_xent": `_nt_xent` of the two views' projections. "dino": the
        teacher (backbone + head) sees the global views without gradients,
        and the loss is the sum of
          - `_dino_loss` of the student's logits for every view and local
            crop against the teacher's targets for the global views;
          - with iBOT, `ibot_loss_weight` * `_ibot_loss`: the student sees
            the global views with some patches masked
            (`random_block_masks`; the same masked forward pass also gives
            the CLS tokens of the DINO loss, as in DINOv2), the teacher
            sees them whole;
          - with KoLeo, `koleo_loss_weight` * `_koleo_loss` of the
            student's CLS tokens, per global view.
        Only then are the centers updated, so the loss uses the earlier
        ones. The terms of the last batch are kept in `_loss_parts`.
        """
        views = list(batch)
        n = views[0].size(0)
        # One forward pass for both global views: the first N rows are view1.
        pixel_values = torch.cat(views[:2], dim=0).to(self.device, non_blocking=True)

        if self.finetune_method == "nt_xent":
            cls = self._model(pixel_values=pixel_values).last_hidden_state[:, 0, :]
            out1, out2 = self._head(cls).chunk(2, dim=0)
            loss = self._nt_xent(out1, out2, self.temperature)
            self._loss_parts = {"nt_xent_loss": loss.item()}
            return loss

        masks = None
        if self._ibot:
            patch_size = self._model.config.patch_size
            masks = random_block_masks(2 * n, pixel_values.shape[-2] // patch_size, pixel_values.shape[-1] // patch_size,
                                       self.ibot_mask_ratio, self.ibot_mask_probability).to(self.device)
            hidden = self._model(pixel_values=pixel_values, bool_masked_pos=masks).last_hidden_state
        else:
            hidden = self._model(pixel_values=pixel_values).last_hidden_state
        cls = hidden[:, 0, :]
        student_logits = list(self._head(cls).chunk(2, dim=0))
        if len(views) > 2:
            # The local crops, all of one (smaller) size: one more forward pass, without masks.
            local_pixels = torch.cat(views[2:], dim=0).to(self.device, non_blocking=True)
            local_cls = self._model(pixel_values=local_pixels).last_hidden_state[:, 0, :]
            student_logits += list(self._head(local_cls).chunk(len(views) - 2, dim=0))

        with torch.no_grad():
            teacher_hidden = self._teacher(pixel_values=pixel_values).last_hidden_state
            teacher_logits = self._teacher_head(teacher_hidden[:, 0, :])
        dino = self._dino_loss(student_logits, list(teacher_logits.chunk(2, dim=0)), self._center,
                               self.student_temp, self.teacher_temp)
        loss = dino
        parts = {"dino_loss": dino.item()}

        teacher_patch_logits = None
        if self._ibot:
            # Patch tokens come after the CLS token and the register tokens.
            prefix = 1 + getattr(self._model.config, "num_register_tokens", 0)
            student_patch_logits = self._ibot_head(hidden[:, prefix:, :][masks])
            with torch.no_grad():
                teacher_patch_logits = self._teacher_ibot_head(teacher_hidden[:, prefix:, :][masks])
            per_image = masks.sum(dim=1, keepdim=True)
            weights = (masks / per_image.clamp_min(1))[masks]
            ibot = self._ibot_loss(student_patch_logits, teacher_patch_logits, weights, (per_image > 0).sum(),
                                   self._ibot_center, self.student_temp, self.teacher_temp)
            loss = loss + self.ibot_loss_weight * ibot
            parts["ibot_loss"] = ibot.item()

        if self.koleo_loss_weight > 0:
            koleo = sum(self._koleo_loss(view_cls) for view_cls in cls.chunk(2, dim=0))
            loss = loss + self.koleo_loss_weight * koleo
            parts["koleo_loss"] = koleo.item()

        if self._dino_stats is not None:
            # Before `_update_center`, so with the same center as the loss.
            self._record_dino_stats(teacher_logits)
        self._update_center(teacher_logits)
        if teacher_patch_logits is not None and len(teacher_patch_logits) > 0:
            self._update_center(teacher_patch_logits, self._ibot_center)
        self._loss_parts = parts
        return loss

    def _reset_dino_stats(self) -> None:
        """Start collecting a new epoch's teacher-target statistics
        (`_record_dino_stats`, read by `_dino_epoch_metrics`)."""
        self._dino_stats = {
            "entropy": 0.0,
            "batch_entropy": 0.0,
            "batches": 0,
            "counts": torch.zeros(self.num_prototypes, dtype=torch.long, device=self.device),
        }

    @torch.no_grad()
    def _record_dino_stats(self, teacher_logits) -> None:
        """Add one batch to the epoch's statistics of the teacher's targets
        (centered and sharpened, as in `_dino_loss`).

        They tell apart the two ways the DINO loss can get low: every image
        confidently assigned to its own prototype (the goal), or every
        image assigned to the same few prototypes (collapse).

        Args:
            teacher_logits: (2N, K) raw logits of the teacher's head for
                both views of the batch.
        """
        log_probs = F.log_softmax((teacher_logits - self._center) / self.teacher_temp, dim=-1)
        probs = log_probs.exp()
        # Per-image entropy: low = sharp targets (one prototype per image).
        self._dino_stats["entropy"] += -(probs * log_probs).sum(dim=-1).mean().item()
        # Entropy of the batch's average target: high = many prototypes in use, ~0 = collapse.
        mean_probs = probs.mean(dim=0)
        self._dino_stats["batch_entropy"] += -(mean_probs * mean_probs.clamp_min(1e-12).log()).sum().item()
        self._dino_stats["counts"] += torch.bincount(probs.argmax(dim=-1), minlength=self.num_prototypes)
        self._dino_stats["batches"] += 1

    def _dino_epoch_metrics(self) -> dict:
        """The epoch's teacher-target statistics, logged with the loss:

        - `teacher_entropy`: mean entropy of one image's target, from
          log(K) (uniform) down to 0 (all on one prototype).
        - `teacher_batch_entropy`: entropy of a batch's mean target, log(K)
          when the batch is spread evenly over all prototypes, 0 when every
          image picks the same one (collapse).
        - `prototypes_used`: how many prototypes are some image's most
          likely one (argmax) during the epoch, out of `num_prototypes`.
        - `top_prototype_share`: fraction of the epoch's images whose
          argmax is the most popular prototype; near 1 = collapse.
        """
        stats = self._dino_stats
        counts = stats["counts"]
        return {
            "teacher_entropy": stats["entropy"] / stats["batches"],
            "teacher_batch_entropy": stats["batch_entropy"] / stats["batches"],
            "prototypes_used": int((counts > 0).sum().item()),
            "top_prototype_share": counts.max().item() / counts.sum().item(),
        }

    @torch.no_grad()
    def _update_teacher(self) -> None:
        """Move the teacher's weights a step toward the student's:
        teacher = m * teacher + (1 - m) * student, with m =
        `teacher_momentum`. Called after every optimizer step with
        `finetune_method="dino"`.

        Only the trained parts (`_student_modules`: the last blocks, the
        final norm and the heads) are updated: the frozen blocks are the
        same in both networks and never change. In place (`mul_`, `add_`),
        so the teacher keeps its tensors and only their values change.
        """
        student = self._student_modules().parameters()
        teacher = self._teacher_modules().parameters()
        for student_param, teacher_param in zip(student, teacher, strict=True):
            teacher_param.mul_(self.teacher_momentum).add_(student_param, alpha=1 - self.teacher_momentum)

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
        `EarlyStopping` and restoring the best epoch's weights at the end
        (with "dino" the teacher's and `center` too). With "dino" the
        teacher follows the student after every optimizer step. Features
        are extracted afterwards by `run`."""
        if not self.finetune:
            raise ValueError(f"{self.name} has a frozen backbone: create it with finetune=True to train")
        self._log_training_start(**{k: v for k, v in locals().items() if k != "self"})
        _, model = self._load()
        loader = self._train_loader(batch_size)
        optimizer = torch.optim.AdamW(self._param_groups(lr), lr=lr)
        early_stopping = EarlyStopping(
            patience=early_stopping_patience,
            min_delta=early_stopping_min_delta,
            smoothing_window=early_stopping_smoothing_window,
            mode="min",
        )

        # train()/eval() only switch dropout/BatchNorm; the ViT has no
        # BatchNorm and DINOv3's dropout rates are 0, so the frozen blocks
        # behave the same in train mode.
        model.train()
        self._head.train()
        if self._ibot_head is not None:
            self._ibot_head.train()

        for epoch in range(epochs):
            train_loss = 0.0
            part_sums = {}
            if self.finetune_method == "dino":
                self._reset_dino_stats()
            for batch in loader:
                optimizer.zero_grad()
                loss = self._finetune_loss(batch)
                loss.backward()
                optimizer.step()
                if self.finetune_method == "dino":
                    self._update_teacher()
                train_loss += loss.item()
                for name, value in self._loss_parts.items():
                    part_sums[name] = part_sums.get(name, 0.0) + value

            average_loss = train_loss / len(loader)
            # The separate terms only when there is more than one (DINO + iBOT / KoLeo).
            loss_parts = {name: total / len(loader) for name, total in part_sums.items()} if len(part_sums) > 1 else {}
            # Snapshots only the trained parts, not the whole backbone.
            smoothed_loss, should_stop = early_stopping.step(
                average_loss, epoch + 1, model=self._finetuned_modules())
            dino_metrics = self._dino_epoch_metrics() if self.finetune_method == "dino" else {}
            # The highest lr of the groups: the base lr (layer-wise decay only lowers it).
            self._log_epoch(epoch + 1, loss=average_loss, smoothed_loss=smoothed_loss,
                            lr=max(group["lr"] for group in optimizer.param_groups), **loss_parts, **dino_metrics)
            message = f"[epoch {epoch + 1}/{epochs}] loss: {average_loss:.3f} | smoothed: {smoothed_loss:.3f}"
            message += "".join(f" | {name.removesuffix('_loss')}: {value:.3f}" for name, value in loss_parts.items())
            if dino_metrics:
                message += (f" | teacher H: {dino_metrics['teacher_entropy']:.2f}"
                            f" | batch H: {dino_metrics['teacher_batch_entropy']:.2f}"
                            f" | protos: {dino_metrics['prototypes_used']}/{self.num_prototypes}"
                            f" | top: {dino_metrics['top_prototype_share']:.2f}")
            print(message)
            if should_stop:
                print(f"Early stopping at epoch {epoch + 1} "
                      f"(best smoothed={early_stopping.best_smoothed:.3f} @ epoch {early_stopping.best_epoch})")
                break

        self._log_training_end(early_stopping, epoch + 1)
        early_stopping.restore_best(self._finetuned_modules())
        # Saving is run_experiment's job (save_model -> self.state_dict()).
