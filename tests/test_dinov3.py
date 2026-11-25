"""Tests of DINOv3's fine-tuning objectives (`finetune_method`): the DINO
loss and its teacher/center updates, the optional multi-crop, iBOT, KoLeo
and layer-wise lr decay, and that both "nt_xent" and "dino" save and load
their trained parts. A tiny fake backbone with the same attributes as the
Hugging Face model (`embeddings.mask_token`, `model.layer`, `norm`,
`config.hidden_size` / `patch_size` / `num_register_tokens`,
`last_hidden_state` with CLS, register and patch tokens, `bool_masked_pos`)
stands in for the real one, so nothing is downloaded and everything runs
on the CPU.

    pytest tests/test_dinov3.py -v
"""
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from pipeline.feature_learning.dinov3 import DINOHead, DINOv3, random_block_masks
from pipeline.preprocessing.augmentation import InkedRandomResizedCrop
from pipeline.preprocessing.dataset import MultiCropDataset

HIDDEN = 8
PROTOTYPES = 16
REGISTERS = 1


class FakeEmbeddings(torch.nn.Module):
    """Every pixel is a 'patch' (patch size 1): its 3 channels -> HIDDEN.
    Masked patches are replaced by `mask_token`, as in Hugging Face."""

    def __init__(self):
        super().__init__()
        self.patch = torch.nn.Linear(3, HIDDEN)
        self.cls_token = torch.nn.Parameter(torch.randn(1, 1, HIDDEN))
        self.register_tokens = torch.nn.Parameter(torch.randn(1, REGISTERS, HIDDEN))
        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, HIDDEN))

    def forward(self, pixel_values, bool_masked_pos=None):
        patches = self.patch(pixel_values.flatten(2).transpose(1, 2))   # (B, H*W, HIDDEN)
        if bool_masked_pos is not None:
            patches = torch.where(bool_masked_pos.unsqueeze(-1), self.mask_token, patches)
        batch = patches.size(0)
        # The CLS token also gets the patches' mean, so it depends on the image.
        cls = self.cls_token.expand(batch, -1, -1) + patches.mean(dim=1, keepdim=True)
        return torch.cat([cls, self.register_tokens.expand(batch, -1, -1), patches], dim=1)


class FakeBackbone(torch.nn.Module):
    """3xHxW pixel_values -> CLS, register and patch tokens through a few
    linear blocks."""

    def __init__(self, n_blocks=3):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=HIDDEN, patch_size=1, num_register_tokens=REGISTERS)
        self.embeddings = FakeEmbeddings()
        self.model = torch.nn.Module()
        self.model.layer = torch.nn.ModuleList([torch.nn.Linear(HIDDEN, HIDDEN) for _ in range(n_blocks)])
        self.norm = torch.nn.LayerNorm(HIDDEN)

    def forward(self, pixel_values, bool_masked_pos=None):
        x = self.embeddings(pixel_values, bool_masked_pos)
        for block in self.model.layer:
            x = block(x)
        return SimpleNamespace(last_hidden_state=self.norm(x))


def make_step(tmp_path, method, seed=0, unfreeze_blocks=2, **params):
    """A fine-tuning DINOv3 step on the fake backbone, set up as `_load`
    would do it (fully frozen, then `_setup_finetuning`)."""
    torch.manual_seed(seed)
    step = DINOv3(finetune=True, finetune_method=method, device="cpu", output_path=tmp_path,
                  unfreeze_blocks=unfreeze_blocks, num_prototypes=PROTOTYPES, **params)
    step._model = FakeBackbone().requires_grad_(False)
    step._setup_finetuning()
    return step


def fake_batch(n=4, n_local=0):
    """Two 2x2 global views, then `n_local` 1x1 'local crops'."""
    return tuple([torch.randn(n, 3, 2, 2) for _ in range(2)] + [torch.randn(n, 3, 1, 1) for _ in range(n_local)])


def test_unknown_finetune_method_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="finetune_method"):
        DINOv3(finetune=True, finetune_method="simclr", device="cpu", output_path=tmp_path)


def test_dino_head_outputs_cosine_similarities():
    logits = DINOHead(HIDDEN, PROTOTYPES)(torch.randn(5, HIDDEN))
    assert logits.shape == (5, PROTOTYPES)
    assert logits.abs().max() <= 1 + 1e-6


class TestDinoLoss:
    def test_uniform_distributions_give_log_k(self):
        zeros = torch.zeros(4, PROTOTYPES)
        loss = DINOv3._dino_loss([zeros, zeros], [zeros, zeros], torch.zeros(1, PROTOTYPES), 0.1, 0.04)
        assert loss.item() == pytest.approx(math.log(PROTOTYPES))

    def test_matching_student_gives_less_than_log_k(self):
        logits = torch.randn(4, PROTOTYPES)
        loss = DINOv3._dino_loss([logits, logits], [logits, logits], torch.zeros(1, PROTOTYPES), 0.04, 0.04)
        assert loss.item() < math.log(PROTOTYPES)

    def test_no_gradient_flows_into_the_teacher(self):
        student = torch.randn(4, PROTOTYPES, requires_grad=True)
        teacher = torch.randn(4, PROTOTYPES, requires_grad=True)
        DINOv3._dino_loss([student, student], [teacher, teacher], torch.zeros(1, PROTOTYPES), 0.1, 0.04).backward()
        assert student.grad is not None
        assert teacher.grad is None

    def test_two_views_cross_the_views_only(self):
        """teacher1 -> student2 and teacher2 -> student1, never a view with itself."""
        s1, s2, t1, t2 = (torch.randn(4, PROTOTYPES) for _ in range(4))
        center = torch.zeros(1, PROTOTYPES)

        def cross_entropy(t, s):
            return -(torch.softmax(t / 0.04, -1) * torch.log_softmax(s / 0.1, -1)).sum(-1).mean()

        expected = 0.5 * (cross_entropy(t1, s2) + cross_entropy(t2, s1))
        assert DINOv3._dino_loss([s1, s2], [t1, t2], center, 0.1, 0.04).item() == pytest.approx(expected.item())

    def test_local_crops_are_matched_to_both_global_views(self):
        s1, s2, local, t1, t2 = (torch.randn(4, PROTOTYPES) for _ in range(5))
        center = torch.zeros(1, PROTOTYPES)

        def cross_entropy(t, s):
            return -(torch.softmax(t / 0.04, -1) * torch.log_softmax(s / 0.1, -1)).sum(-1).mean()

        # 2 teacher views x 3 student views, minus the 2 same-view pairs.
        expected = (cross_entropy(t1, s2) + cross_entropy(t1, local)
                    + cross_entropy(t2, s1) + cross_entropy(t2, local)) / 4
        loss = DINOv3._dino_loss([s1, s2, local], [t1, t2], center, 0.1, 0.04)
        assert loss.item() == pytest.approx(expected.item())


class TestOptionalAdditions:
    @pytest.mark.parametrize("param", ["local_crops_number", "ibot_loss_weight", "koleo_loss_weight"])
    def test_rejected_without_dino(self, tmp_path, param):
        with pytest.raises(ValueError, match=param):
            DINOv3(finetune=True, finetune_method="nt_xent", device="cpu", output_path=tmp_path, **{param: 1})

    def test_logged_only_when_switched_on(self, tmp_path):
        off = DINOv3(finetune=True, finetune_method="dino", device="cpu", output_path=tmp_path)
        assert not {"local_crops_number", "ibot_loss_weight", "koleo_loss_weight",
                    "layerwise_lr_decay"} & set(off.hparams)
        on = DINOv3(finetune=True, finetune_method="dino", device="cpu", output_path=tmp_path,
                    local_crops_number=6, ibot_loss_weight=1.0, koleo_loss_weight=0.1, layerwise_lr_decay=0.9)
        assert on.hparams["local_crops_number"] == 6
        assert on.hparams["ibot_loss_weight"] == 1.0
        assert on.hparams["koleo_loss_weight"] == 0.1
        assert on.hparams["layerwise_lr_decay"] == 0.9

    def test_off_keeps_the_old_state_dict_keys(self, tmp_path):
        keys = set(make_step(tmp_path, "dino").state_dict())
        assert not any("ibot" in key or "mask_token" in key for key in keys)


class TestMultiCrop:
    def test_finetune_loss_with_local_crops(self, tmp_path):
        step = make_step(tmp_path, "dino", local_crops_number=3)
        step._reset_dino_stats()
        loss = step._finetune_loss(fake_batch(n=4, n_local=3))
        loss.backward()
        assert torch.isfinite(loss)
        assert all(s.grad is not None for s in step._student_modules().parameters())
        # The teacher saw only the 2 global views.
        assert step._dino_stats["counts"].sum() == 8

    def test_dataset_yields_two_globals_and_the_local_crops(self, tmp_path):
        paths = []
        for i in range(2):
            paths.append(tmp_path / f"{i}.png")
            Image.fromarray(np.full((32, 32), 255, dtype=np.uint8)).save(paths[-1])
        dataset = MultiCropDataset(paths, lambda im: torch.zeros(1, 32, 32), lambda im: torch.zeros(1, 8, 8), n_local=4)
        item = dataset[0]
        assert len(item) == 6
        assert [tuple(t.shape) for t in item] == [(1, 32, 32)] * 2 + [(1, 8, 8)] * 4

    def test_inked_crop_avoids_blank_paper(self):
        page = np.full((200, 200), 255, dtype=np.uint8)
        page[150:190, 150:190] = 0   # the drawing: a dark square in one corner, 4% of the page
        image = Image.fromarray(page)
        torch.manual_seed(0)
        crop = InkedRandomResizedCrop(16, scale=(0.05, 0.1), min_ink=0.01, max_tries=200)
        crops = [np.asarray(crop(image)) for _ in range(20)]
        assert all(c.shape == (16, 16) for c in crops)
        assert all((c < 128).mean() >= 0.01 for c in crops)


class TestIbot:
    def test_masks_have_the_ratio_and_probability(self):
        torch.manual_seed(0)
        masks = random_block_masks(400, 14, 14, ratio_range=(0.1, 0.5), probability=0.5)
        assert masks.shape == (400, 196) and masks.dtype == torch.bool
        masked = masks.any(dim=1)
        assert 0.4 < masked.float().mean() < 0.6
        ratios = masks[masked].float().mean(dim=1)
        assert ratios.min() >= 0.1 - 1e-6 and ratios.mean() < 0.6

    def test_setup_trains_the_mask_token_and_a_patch_head(self, tmp_path):
        step = make_step(tmp_path, "dino", ibot_loss_weight=1.0, ibot_num_prototypes=12)
        mask_token = step._model.embeddings.mask_token
        assert mask_token.requires_grad
        assert any(p is mask_token for p in step._trainable_parameters())
        assert step._ibot_head.prototypes.out_features == 12
        assert step._ibot_center is step._teacher_ibot_head.center
        assert not any(p.requires_grad for p in step._teacher_ibot_head.parameters())

    def test_finetune_loss_masks_the_student_and_updates_the_patch_center(self, tmp_path):
        step = make_step(tmp_path, "dino", ibot_loss_weight=1.0, ibot_mask_probability=1.0, ibot_mask_ratio=(0.5, 0.5))
        loss = step._finetune_loss(fake_batch(n=4))
        loss.backward()
        assert torch.isfinite(loss)
        assert set(step._loss_parts) == {"dino_loss", "ibot_loss"}
        assert step._model.embeddings.mask_token.grad is not None
        assert step._ibot_center.abs().sum() > 0
        assert all(t.grad is None for t in step._teacher_modules().parameters())

    def test_ibot_loss_weights_every_masked_image_equally(self):
        # Image 0 has 1 masked patch, image 1 has 3; per-patch losses 1 and 3 x 2.
        student = torch.zeros(4, PROTOTYPES)
        teacher = torch.zeros(4, PROTOTYPES)
        uniform = DINOv3._ibot_loss(student, teacher, torch.tensor([1.0, 1 / 3, 1 / 3, 1 / 3]), 2,
                                    torch.zeros(1, PROTOTYPES), 0.1, 0.04)
        assert uniform.item() == pytest.approx(math.log(PROTOTYPES))

    def test_state_dict_round_trip_with_ibot(self, tmp_path):
        step = make_step(tmp_path / "a", "dino", seed=0, ibot_loss_weight=1.0)
        step._update_center(torch.randn(6, PROTOTYPES), step._ibot_center)
        state = step.state_dict()
        assert "student.mask_token.mask_token" in state
        assert "teacher.ibot_head.center" in state
        other = make_step(tmp_path / "b", "dino", seed=1, ibot_loss_weight=1.0)
        other.load_state_dict(state)
        for key, value in other.state_dict().items():
            assert torch.equal(value, state[key]), key
        assert other._ibot_center is other._teacher_ibot_head.center
        assert other._model.embeddings.mask_token is other._student_modules()["mask_token"].mask_token


class TestKoleo:
    def test_spread_vectors_score_lower_than_close_ones(self):
        spread = torch.eye(4)
        close = torch.eye(4) * 0.01 + 1.0
        assert DINOv3._koleo_loss(spread) < DINOv3._koleo_loss(close)

    def test_finetune_loss_logs_the_koleo_term(self, tmp_path):
        step = make_step(tmp_path, "dino", koleo_loss_weight=0.1)
        loss = step._finetune_loss(fake_batch(n=4))
        loss.backward()
        assert set(step._loss_parts) == {"dino_loss", "koleo_loss"}
        assert loss.item() == pytest.approx(step._loss_parts["dino_loss"] + 0.1 * step._loss_parts["koleo_loss"],
                                            rel=1e-5)


class TestLayerwiseLrDecay:
    def test_off_is_one_group(self, tmp_path):
        step = make_step(tmp_path, "dino")
        groups = step._param_groups(1e-3)
        assert len(groups) == 1 and groups[0]["lr"] == 1e-3

    def test_earlier_blocks_get_a_smaller_lr(self, tmp_path):
        # 3 blocks, all trained: blocks 0, 1, 2 get decay**3, **2, **1; norm and heads the full lr.
        step = make_step(tmp_path, "dino", unfreeze_blocks=3, layerwise_lr_decay=0.5, ibot_loss_weight=1.0)
        groups = step._param_groups(1.0)
        assert [g["lr"] for g in groups] == [0.125, 0.25, 0.5, 1.0, 1.0, 1.0, 0.0625]
        grouped = [id(p) for g in groups for p in g["params"]]
        assert sorted(grouped) == sorted(id(p) for p in step._trainable_parameters())

    def test_invalid_decay_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="layerwise_lr_decay"):
            DINOv3(finetune=True, finetune_method="dino", device="cpu", output_path=tmp_path, layerwise_lr_decay=0)


class TestDinoSetup:
    def test_teacher_starts_as_a_frozen_copy_of_the_student(self, tmp_path):
        step = make_step(tmp_path, "dino")
        for s, t in zip(step._student_modules().parameters(), step._teacher_modules().parameters()):
            assert torch.equal(s, t)
            assert s.requires_grad and not t.requires_grad

    def test_center_is_a_zero_buffer_of_the_teacher_head(self, tmp_path):
        step = make_step(tmp_path, "dino")
        assert step._center is step._teacher_head.center
        assert torch.equal(step._center, torch.zeros(1, PROTOTYPES))
        assert not hasattr(step._head, "center")

    def test_optimizer_gets_only_the_student(self, tmp_path):
        step = make_step(tmp_path, "dino")
        trainable = {id(p) for p in step._trainable_parameters()}
        assert trainable == {id(p) for p in step._student_modules().parameters()}


class TestDinoUpdates:
    def test_update_teacher_moves_it_by_one_minus_momentum(self, tmp_path):
        step = make_step(tmp_path, "dino")
        before = [t.clone() for t in step._teacher_modules().parameters()]
        with torch.no_grad():
            for s in step._student_modules().parameters():
                s.add_(1.0)
        step._update_teacher()
        for old, t in zip(before, step._teacher_modules().parameters()):
            assert torch.allclose(t - old, torch.full_like(t, 1 - step.teacher_momentum), atol=1e-6)

    def test_frozen_blocks_of_the_teacher_are_not_touched(self, tmp_path):
        step = make_step(tmp_path, "dino")
        frozen_before = step._teacher.model.layer[0].weight.clone()
        step._update_teacher()
        assert torch.equal(step._teacher.model.layer[0].weight, frozen_before)

    def test_update_center_is_a_moving_average_in_place(self, tmp_path):
        step = make_step(tmp_path, "dino")
        step._update_center(torch.ones(6, PROTOTYPES))
        assert torch.allclose(step._center, torch.full((1, PROTOTYPES), 1 - step.center_momentum))
        assert step._center is step._teacher_head.center

    def test_finetune_loss_uses_the_old_center_then_updates_it(self, tmp_path):
        step = make_step(tmp_path, "dino")
        loss = step._finetune_loss(fake_batch())
        loss.backward()
        assert torch.isfinite(loss)
        assert step._center.abs().sum() > 0
        assert all(t.grad is None for t in step._teacher_modules().parameters())
        assert all(s.grad is not None for s in step._student_modules().parameters())


class TestDinoStats:
    def test_collapse_shows_one_prototype_and_zero_batch_entropy(self, tmp_path):
        step = make_step(tmp_path, "dino")
        step._reset_dino_stats()
        logits = torch.zeros(8, PROTOTYPES)
        logits[:, 3] = 1.0   # every image's target is prototype 3
        step._record_dino_stats(logits)
        metrics = step._dino_epoch_metrics()
        assert metrics["prototypes_used"] == 1
        assert metrics["top_prototype_share"] == 1.0
        assert metrics["teacher_batch_entropy"] < 0.01

    def test_one_prototype_per_image_shows_sharp_but_spread_targets(self, tmp_path):
        step = make_step(tmp_path, "dino")
        step._reset_dino_stats()
        step._record_dino_stats(torch.eye(PROTOTYPES))   # image i -> prototype i
        metrics = step._dino_epoch_metrics()
        assert metrics["prototypes_used"] == PROTOTYPES
        assert metrics["top_prototype_share"] == pytest.approx(1 / PROTOTYPES)
        assert metrics["teacher_entropy"] < 0.01
        assert metrics["teacher_batch_entropy"] == pytest.approx(math.log(PROTOTYPES), abs=0.01)

    def test_finetune_loss_records_stats_only_when_collecting(self, tmp_path):
        step = make_step(tmp_path, "dino")
        step._finetune_loss(fake_batch())          # no `_reset_dino_stats`: nothing collected
        assert step._dino_stats is None
        step._reset_dino_stats()
        step._finetune_loss(fake_batch(n=4))
        assert step._dino_stats["batches"] == 1
        assert step._dino_stats["counts"].sum() == 8   # both views of 4 images


@pytest.mark.parametrize("method, prefix", [("nt_xent", "blocks."), ("dino", "student.")])
def test_state_dict_round_trip(tmp_path, method, prefix):
    step = make_step(tmp_path / "a", method, seed=0)
    if method == "dino":
        step._update_center(torch.randn(6, PROTOTYPES))
    state = step.state_dict()
    assert next(iter(state)).startswith(prefix)
    assert ("teacher.head.center" in state) == (method == "dino")

    other = make_step(tmp_path / "b", method, seed=1)
    other.load_state_dict(state)
    for key, value in other.state_dict().items():
        assert torch.equal(value, state[key]), key
    if method == "dino":
        # Loading copies into the buffer, so `_center` still points at it.
        assert other._center is other._teacher_head.center
        assert torch.equal(other._center, state["teacher.head.center"])
