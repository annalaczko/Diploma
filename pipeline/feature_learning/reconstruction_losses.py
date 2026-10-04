"""Interchangeable reconstruction losses for the VAE.

Every loss here has the same signature, `loss(recon_x, x) -> scalar`,
with `recon_x`/`x` of shape (batch, 1, H, W) and pixel values in [0, 1],
so `VAE(recon_loss=...)` can swap between them by name (see
`RECONSTRUCTION_LOSSES`) or take any custom callable with that signature
(e.g. `functools.partial(dice_loss, smooth=0.1)`).

Why overlap-based losses (Dice / soft-IoU): the sherd images are thin dark
outlines on a mostly white background, so a pixel-averaged loss like BCE
is dominated by the background - a reconstruction that misses most of the
outline still scores a small error. Dice and IoU instead measure how well
the *ink* of the reconstruction overlaps the ink of the target, so the
background's size doesn't matter, while ink drawn where there is none still
gets penalized (it enlarges the denominator).

Scaling: BCE is summed over pixels (and the KL term in `VAE._loss_terms`
is summed over latent dims), so the overlap losses are computed per image
in [0, 1] and then multiplied by the image's pixel count - this keeps them
on the same order of magnitude as the summed BCE, so the beta-weighted KL
term keeps roughly the same relative strength whichever loss is chosen.
"""

import torch.nn as nn


def _ink(img, ink_is_dark):
    """Map pixel values to "ink probability": ToTensor gives white
    background = 1 and black outline = 0, so dark ink needs inverting."""
    return 1.0 - img if ink_is_dark else img


def bce_loss(recon_x, x):
    """Plain pixel-wise binary cross-entropy, summed (the original VAE loss)."""
    return nn.functional.binary_cross_entropy(recon_x, x, reduction="sum")


def dice_loss(recon_x, x, smooth=1.0, ink_is_dark=True):
    """Soft Dice loss on the ink pixels: 1 - 2|P∩T| / (|P| + |T|), using
    the soft (non-thresholded) predictions so it stays differentiable.

    Args:
        smooth: added to numerator and denominator; avoids 0/0 on images
            with no ink and softens the gradient for near-empty ones.
        ink_is_dark: True if the outline is darker than the background.
    """
    p = _ink(recon_x, ink_is_dark).flatten(1)
    t = _ink(x, ink_is_dark).flatten(1)
    inter = (p * t).sum(dim=1)
    dice = 1 - (2 * inter + smooth) / (p.sum(dim=1) + t.sum(dim=1) + smooth)
    return (dice * p.shape[1]).sum()


def soft_iou_loss(recon_x, x, smooth=1.0, ink_is_dark=True):
    """Soft IoU (Jaccard) loss on the ink pixels: 1 - |P∩T| / |P∪T|.
    Same idea as `dice_loss` but penalizes partial overlap more harshly.
    See `dice_loss` for the arguments."""
    p = _ink(recon_x, ink_is_dark).flatten(1)
    t = _ink(x, ink_is_dark).flatten(1)
    inter = (p * t).sum(dim=1)
    union = p.sum(dim=1) + t.sum(dim=1) - inter
    iou = 1 - (inter + smooth) / (union + smooth)
    return (iou * p.shape[1]).sum()


def bce_dice_loss(recon_x, x, dice_weight=1.0):
    """BCE + Dice: Dice drives the outline overlap, BCE keeps per-pixel
    gradients everywhere (Dice alone can be unstable early in training,
    when the predicted ink is still a uniform gray)."""
    return bce_loss(recon_x, x) + dice_weight * dice_loss(recon_x, x)


RECONSTRUCTION_LOSSES = {
    "bce": bce_loss,
    "dice": dice_loss,
    "iou": soft_iou_loss,
    "bce_dice": bce_dice_loss,
}


def get_reconstruction_loss(loss):
    """Resolve `loss` (a name from `RECONSTRUCTION_LOSSES`, or a callable
    `loss(recon_x, x) -> scalar`) to a callable."""
    if callable(loss):
        return loss
    try:
        return RECONSTRUCTION_LOSSES[loss]
    except KeyError:
        raise ValueError(
            f"Unknown reconstruction loss {loss!r}; "
            f"choose from {sorted(RECONSTRUCTION_LOSSES)} or pass a callable"
        ) from None
