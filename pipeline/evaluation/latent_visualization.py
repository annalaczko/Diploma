"""Plots for looking at a latent (feature) space, used in the notebooks.

    thumbnail_map       - every image drawn at its 2D (t-SNE) position, its
                          ink tinted by group (cluster or true class): which
                          shapes sit next to each other, where groups blend.
    cluster_transitions - walks in a straight line from one cluster's
                          medoid to the next, decodes each step with the
                          VAE, and shows the closest real image under it:
                          how one cluster's shape morphs into the next one's.
    neighbour_paths     - the same with real images only (no decoder
                          needed, so it works for any features, e.g. DINOv3).

`encode_folder` produces the inputs (latent means + small thumbnails) from
an image folder and a trained `VAE`; `load_thumbnails` only the thumbnails.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import to_rgb
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image
from sklearn.manifold import TSNE

from pipeline.preprocessing.dataset import ImageDataset, collect_image_paths

# Fixed categorical order (validated reference palette); groups beyond
# these fold into OTHER_COLOR rather than getting generated hues.
GROUP_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
                "#008300", "#4a3aa7", "#e34948"]
OTHER_COLOR = "#9a9a94"


def encode_folder(vae, folder, device, batch_size=128, thumb_size=64):
    """Encode every image under `folder` with `vae` (in eval mode, so no
    dropout) and return what the plots need.

    Returns:
        (filenames, mu, thumbs): image file names, the (n, k) latent means,
        and (n, thumb_size, thumb_size) float thumbnails in [0, 1].
    """
    paths = collect_image_paths(folder)
    dataset = ImageDataset(paths, grayscale=True, target_size=None)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    vae.eval()
    mus = []
    with torch.no_grad():
        for x in loader:
            mu, _ = vae.encoder(x.to(device))
            mus.append(mu.cpu())
    mu = torch.cat(mus).numpy()

    thumbs = np.stack([_thumbnail(p, thumb_size) for p in dataset.paths])
    return [Path(p).name for p in dataset.paths], mu, thumbs


def load_thumbnails(folder, filenames, thumb_size=64):
    """(n, thumb_size, thumb_size) thumbnails in [0, 1] for `filenames`
    (in that order) from `folder` - for features computed elsewhere, e.g.
    a features CSV from a model without a decoder."""
    return np.stack([_thumbnail(Path(folder) / f, thumb_size) for f in filenames])


def _thumbnail(path, size):
    """A (size, size) grayscale thumbnail of an image file, values in [0, 1]."""
    image = Image.open(path).convert("L").resize((size, size), Image.LANCZOS)
    return np.asarray(image, dtype=np.float32) / 255.0


def decode(vae, z, device):
    """Decode latent vectors `z` (n, k) into (n, H, W) images in [0, 1]."""
    vae.eval()
    with torch.no_grad():
        z = torch.as_tensor(z, dtype=torch.float32, device=device)
        h = vae.decoder.fc(z).view(len(z), -1, 32, 32)
        return vae.decoder(h).squeeze(1).cpu().numpy()


def embed_2d(features, perplexity=30, random_state=0):
    """t-SNE embedding of `features` to 2D (local neighbourhoods are kept,
    global distances between far-apart groups are not meaningful)."""
    return TSNE(n_components=2, perplexity=perplexity, init="pca",
                random_state=random_state).fit_transform(features)


def _tint(thumb, color):
    """RGBA thumbnail: ink (dark pixels) in `color`, background transparent."""
    ink = 1.0 - thumb
    rgba = np.zeros((*thumb.shape, 4), dtype=np.float32)
    rgba[..., :3] = to_rgb(color)
    rgba[..., 3] = np.clip(ink * 1.5, 0, 1)
    return rgba


def thumbnail_map(coords, thumbs, groups, title, ax=None, zoom=0.45, max_groups=len(GROUP_COLORS)):
    """Draw each thumbnail at its 2D position, ink tinted by its group.

    Args:
        coords: (n, 2) embedding, e.g. from `embed_2d`.
        thumbs: (n, h, w) thumbnails in [0, 1] (dark ink on white).
        groups: (n,) group label per image (cluster id or class name).
            The `max_groups` largest groups get a color each; the rest are
            drawn gray as "other".
        title: plot title.
        zoom: thumbnail scale on the canvas.

    Returns:
        The matplotlib Axes.
    """
    groups = np.asarray(groups, dtype=object)
    names, counts = np.unique(groups, return_counts=True)
    ranked = [n for _, n in sorted(zip(-counts, names), key=lambda t: t[0])][:max_groups]
    color_of = {g: GROUP_COLORS[i] for i, g in enumerate(ranked)}

    if ax is None:
        _, ax = plt.subplots(figsize=(14, 14))
    # Draw "other" first so the colored groups sit on top.
    order = sorted(range(len(groups)), key=lambda i: groups[i] in color_of)
    for i in order:
        color = color_of.get(groups[i], OTHER_COLOR)
        box = AnnotationBbox(OffsetImage(_tint(thumbs[i], color), zoom=zoom),
                             coords[i], frameon=False, pad=0)
        ax.add_artist(box)

    # Group name at each colored group's median position, in neutral ink.
    for g in ranked:
        cx, cy = np.median(coords[groups == g], axis=0)
        ax.text(cx, cy, str(g), fontsize=13, weight="bold", color="#1a1a19", ha="center",
                va="center", bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=color_of[g], lw=2, alpha=0.9))

    handles = [plt.Line2D([], [], marker="s", ls="", ms=10, color=color_of[g],
                          label=f"{g} (n={int((groups == g).sum())})") for g in ranked]
    if len(ranked) < len(names):
        handles.append(plt.Line2D([], [], marker="s", ls="", ms=10, color=OTHER_COLOR,
                                  label=f"other (n={int((~np.isin(groups, ranked)).sum())})"))
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=11)

    pad = 0.05 * np.ptp(coords, axis=0)
    ax.set_xlim(coords[:, 0].min() - pad[0], coords[:, 0].max() + pad[0])
    ax.set_ylim(coords[:, 1].min() - pad[1], coords[:, 1].max() + pad[1])
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, fontsize=15, loc="left")
    return ax


def cluster_order(centroids):
    """Greedy nearest-neighbour path through the cluster centroids,
    starting from the one farthest from the overall mean - so consecutive
    clusters in the path are latent-space neighbours."""
    remaining = list(range(len(centroids)))
    current = int(np.argmax(np.linalg.norm(centroids - centroids.mean(0), axis=1)))
    path = [current]
    remaining.remove(current)
    while remaining:
        d = [np.linalg.norm(centroids[current] - centroids[j]) for j in remaining]
        current = remaining.pop(int(np.argmin(d)))
        path.append(current)
    return path


def cluster_transitions(vae, mu, clusters, thumbs, device=None, steps=8, title=None):
    """For each consecutive pair of clusters (in `cluster_order`), decode
    `steps` points on the straight line between their latent medoids
    (top row) and show the real image whose latent mean is closest to each
    point (bottom row).

    With `vae=None` (features from a model without a decoder, e.g. DINOv3)
    only the nearest-real-image row is drawn, with `mu` being those
    features.

    Medoid = the cluster's real image closest to its centroid. The centroid
    itself is an average of very different shapes and tends to fall in an
    empty region of the latent space, where the decoder only produces
    broken fragments; a medoid is always a real, decodable shape.

    Args:
        mu: (n, k) latent means (unscaled - the decoder's own space).
        clusters: (n,) cluster id per image.
        thumbs: (n, h, w) thumbnails, for the nearest-real-image row.

    Returns:
        The matplotlib Figure.
    """
    ids = np.unique(clusters)
    centroids = np.stack([mu[clusters == c].mean(0) for c in ids])
    path = cluster_order(centroids)
    anchors = []
    for c, centroid in zip(ids, centroids):
        members = np.flatnonzero(clusters == c)
        anchors.append(mu[members[np.argmin(np.linalg.norm(mu[members] - centroid, axis=1))]])
    anchors = np.stack(anchors)

    pairs = list(zip(path[:-1], path[1:]))
    rows = 1 if vae is None else 2
    fig, axes = plt.subplots(rows * len(pairs), steps, figsize=(1.6 * steps, 1.7 * rows * len(pairs)))
    axes = np.atleast_2d(axes)
    for r, (a, b) in enumerate(pairs):
        t = np.linspace(0, 1, steps)[:, None]
        z = (1 - t) * anchors[a] + t * anchors[b]
        nearest = [int(np.argmin(np.linalg.norm(mu - zi, axis=1))) for zi in z]
        first, bottom_row = rows * r, rows * r + rows - 1
        if vae is not None:
            decoded = decode(vae, z, device)
            for s in range(steps):
                axes[first, s].imshow(decoded[s], cmap="gray", vmin=0, vmax=1)
            axes[first, 0].set_ylabel("decoded", fontsize=9)
        for s in range(steps):
            bottom = axes[bottom_row, s]
            bottom.imshow(thumbs[nearest[s]], cmap="gray", vmin=0, vmax=1)
            if clusters[nearest[s]] in (ids[a], ids[b]):
                bottom.set_title(f"cl. {clusters[nearest[s]]}", fontsize=8, color="#52514b")
            else:
                bottom.set_title(f"cl. {clusters[nearest[s]]}", fontsize=8, color="#1a1a19", weight="bold")
        for ax in axes[first:bottom_row + 1].ravel():
            ax.set_xticks([]); ax.set_yticks([])
        axes[bottom_row, 0].set_ylabel("nearest\nreal", fontsize=9)
        axes[first, 0].set_title(f"cluster {ids[a]}", fontsize=10, loc="left", weight="bold")
        axes[first, -1].set_title(f"cluster {ids[b]}", fontsize=10, loc="right", weight="bold")
    fig.suptitle(title or "Latent-space path between neighbouring cluster medoids", fontsize=13, x=0.01, ha="left")
    fig.tight_layout()
    return fig


def neighbour_paths(features, clusters, thumbs, steps=8, n_neighbors=10, title=None):
    """Like `cluster_transitions`, but needs no decoder: for each
    consecutive pair of clusters, follows the shortest path between their
    medoids through a k-nearest-neighbour graph of the real images, so
    every step is a real sherd that is a close neighbour of the previous
    one. Shows how one cluster's shapes gradually turn into the next one's
    through actual examples (works for any features, e.g. DINOv3).

    Args:
        features: (n, d) feature vectors (as clustered).
        clusters: (n,) cluster id per image.
        thumbs: (n, h, w) thumbnails.
        steps: images shown per path (evenly resampled along it).
        n_neighbors: k of the kNN graph; larger = shorter, coarser paths.

    Returns:
        The matplotlib Figure.
    """
    from scipy.sparse.csgraph import shortest_path
    from sklearn.neighbors import kneighbors_graph

    graph = kneighbors_graph(features, n_neighbors=n_neighbors, mode="distance")
    graph = graph.maximum(graph.T)  # symmetric, so paths can go either way

    ids = np.unique(clusters)
    centroids = np.stack([features[clusters == c].mean(0) for c in ids])
    medoids = []
    for c, centroid in zip(ids, centroids):
        members = np.flatnonzero(clusters == c)
        medoids.append(int(members[np.argmin(np.linalg.norm(features[members] - centroid, axis=1))]))
    order = cluster_order(centroids)
    pairs = list(zip(order[:-1], order[1:]))

    _, pred = shortest_path(graph, directed=False, indices=[medoids[a] for a, _ in pairs],
                            return_predecessors=True)
    fig, axes = plt.subplots(len(pairs), steps, figsize=(1.6 * steps, 1.9 * len(pairs)))
    axes = np.atleast_2d(axes)
    for r, (a, b) in enumerate(pairs):
        node, path = medoids[b], []
        while node >= 0:
            path.append(node)
            node = pred[r, node]
        path = path[::-1]  # medoid a -> medoid b
        if path[0] != medoids[a]:
            raise ValueError(f"clusters {ids[a]} and {ids[b]} are disconnected in the kNN graph; raise n_neighbors")
        shown = [path[int(round(i))] for i in np.linspace(0, len(path) - 1, steps)]
        for s, idx in enumerate(shown):
            ax = axes[r, s]
            ax.imshow(thumbs[idx], cmap="gray", vmin=0, vmax=1)
            ax.set_title(f"cl. {clusters[idx]}", fontsize=8, color="#52514b")
            ax.set_xticks([]); ax.set_yticks([])
        axes[r, 0].set_ylabel(f"{ids[a]} → {ids[b]}\n({len(path)} steps)", fontsize=9)
    fig.suptitle(title or "Shortest path through real neighbours between cluster medoids",
                 fontsize=13, x=0.01, ha="left")
    fig.tight_layout()
    return fig
