"""Tests of DINOv3's fine-tuning objectives (`finetune_method`): the DINO
loss and its teacher/center updates, and that both "nt_xent" and "dino"
save and load their trained parts. A tiny fake backbone with the same
attributes as the Hugging Face model (`model.layer`, `norm`,
`config.hidden_size`, `last_hidden_state`) stands in for the real one, so
nothing is downloaded and everything runs on the CPU.

    pytest tests/test_dinov3.py -v
"""
import math
from types import SimpleNamespace

import pytest
import torch

from pipeline.feature_learning.dinov3 import DINOHead, DINOv3

HIDDEN = 8
PROTOTYPES = 16


class FakeBackbone(torch.nn.Module):
    """3x2x2 pixel_values -> a 'CLS token' through a few linear blocks."""

    def __init__(self, n_blocks=3):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=HIDDEN)
        self.embed = torch.nn.Linear(12, HIDDEN)
        self.model = torch.nn.Module()
        self.model.layer = torch.nn.ModuleList([torch.nn.Linear(HIDDEN, HIDDEN) for _ in range(n_blocks)])
        self.norm = torch.nn.LayerNorm(HIDDEN)

    def forward(self, pixel_values):
        x = self.embed(pixel_values.flatten(1))
        for block in self.model.layer:
            x = block(x)
        return SimpleNamespace(last_hidden_state=self.norm(x).unsqueeze(1))


def make_step(tmp_path, method, seed=0):
    """A fine-tuning DINOv3 step on the fake backbone, set up as `_load`
    would do it (fully frozen, then `_setup_finetuning`)."""
    torch.manual_seed(seed)
    step = DINOv3(finetune=True, finetune_method=method, device="cpu", output_path=tmp_path,
                  unfreeze_blocks=2, num_prototypes=PROTOTYPES)
    step._model = FakeBackbone().requires_grad_(False)
    step._setup_finetuning()
    return step


def fake_batch(n=4):
    return torch.randn(n, 3, 2, 2), torch.randn(n, 3, 2, 2)


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
        loss = DINOv3._dino_loss(zeros, zeros, zeros, zeros, torch.zeros(1, PROTOTYPES), 0.1, 0.04)
        assert loss.item() == pytest.approx(math.log(PROTOTYPES))

    def test_matching_student_gives_less_than_log_k(self):
        logits = torch.randn(4, PROTOTYPES)
        loss = DINOv3._dino_loss(logits, logits, logits, logits, torch.zeros(1, PROTOTYPES), 0.04, 0.04)
        assert loss.item() < math.log(PROTOTYPES)

    def test_no_gradient_flows_into_the_teacher(self):
        student = torch.randn(4, PROTOTYPES, requires_grad=True)
        teacher = torch.randn(4, PROTOTYPES, requires_grad=True)
        DINOv3._dino_loss(student, student, teacher, teacher, torch.zeros(1, PROTOTYPES), 0.1, 0.04).backward()
        assert student.grad is not None
        assert teacher.grad is None


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
