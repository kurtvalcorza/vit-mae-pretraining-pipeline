"""Corpus-level measures for the two things this pipeline can be read on, in numpy.

**Reconstruction** (what a masked autoencoder is trained for): the mean squared error between the decoder's
prediction and the true pixels of the patches that were hidden from the encoder, averaged over the hidden
patches of an image and then over images — the checkpoint's own training loss (`norm_pix_loss` false: raw
pixels in the processor's normalised space). Two non-neural references a fine-tuned model must beat: the
**mean-patch fill** (every hidden patch predicted as the mean colour of the visible patches — a decoder that
knows only the image's average) and the **blur fill** (every hidden patch predicted as the mean of its visible
neighbours in the 14 x 14 patch grid — the cheapest spatial prior).

**Linear-probe classification** (how the fleet reads a self-supervised encoder): accuracy and macro F1 of a
linear head on frozen embeddings, with a per-class breakdown, against the **majority floor** and a **colour
nearest neighbour** (the label of the training photograph whose 3 x 3 mean-colour grid is closest — a
classifier that knows the image through 27 numbers) and a **k-nearest-neighbour** on the same embeddings.
"""
# ruff: noqa: E501  -- fleet metrics module written at the 110-column fleet width; this repo lints at 100

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from PIL import Image

COLOUR_GRID = 3
METRIC_DEFINITIONS = {
    "accuracy": "fraction of photographs whose highest-scoring class is the gold label; in 0..1",
    "macro_f1": "unweighted mean over classes of the F1 of predicting that class; in 0..1",
}
RECONSTRUCTION_DEFINITIONS = {
    "masked_mse": (
        "mean over images of the mean squared error between predicted and true pixels over the patches hidden "
        "from the encoder, in the processor's normalised pixel space (ImageNet mean / std); lower is better"
    ),
    "masked_mse_visible": "the same error over the patches the encoder saw (the decoder also predicts those); lower is better",
    "mask_ratio": "fraction of the patches hidden from the encoder (the seeded random mask)",
    "psnr_masked": "peak signal-to-noise ratio of the hidden patches in 0..1 pixel space, dB; higher is better",
}


# ---------------------------------------------------------------------------------- reconstruction


def masked_mse(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    """MSE over the patches with mask == 1. `prediction` / `target` are `[patches, values]`, `mask` `[patches]`."""
    pred = np.asarray(prediction, dtype=np.float64)
    true = np.asarray(target, dtype=np.float64)
    m = np.asarray(mask, dtype=np.float64)
    if pred.shape != true.shape or pred.ndim != 2 or m.shape != (pred.shape[0],):
        raise ValueError("prediction and target must be [patches, values] with one mask entry per patch")
    if m.sum() <= 0:
        raise ValueError("at least one patch must be masked")
    per_patch = ((pred - true) ** 2).mean(axis=1)
    return float((per_patch * m).sum() / m.sum())


def psnr_from_mse(mse_normalised: float, *, std: Sequence[float] = (0.229, 0.224, 0.225)) -> float:
    """PSNR (dB) in 0..1 pixel space from an MSE measured in ImageNet-normalised space (per-channel std applied
    as the mean squared std; an approximation stated as such)."""
    scale = float(np.mean(np.square(np.asarray(std, dtype=np.float64))))
    mse_pixels = max(mse_normalised * scale, 1e-12)
    return float(10.0 * math.log10(1.0 / mse_pixels))


def reconstruction_metrics(rows: Sequence[Mapping[str, Any]], *, mask_ratio: float) -> dict[str, Any]:
    """Aggregate per-image rows `{id, masked_mse, visible_mse, ...}` into the corpus reading."""
    if not rows:
        raise ValueError("no images to score")
    masked = np.asarray([r["masked_mse"] for r in rows], dtype=np.float64)
    visible = np.asarray([r["visible_mse"] for r in rows], dtype=np.float64)
    if not (np.all(np.isfinite(masked)) and np.all(np.isfinite(visible))):
        raise ValueError("reconstruction errors must be finite")
    return {
        "n": int(len(rows)),
        "masked_mse": float(masked.mean()),
        "masked_mse_median": float(np.median(masked)),
        "masked_mse_visible": float(visible.mean()),
        "psnr_masked": psnr_from_mse(float(masked.mean())),
        "mask_ratio": float(mask_ratio),
        "definitions": dict(RECONSTRUCTION_DEFINITIONS),
    }


def mean_patch_fill(patches: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Predict every hidden patch as the mean over the visible patches (per value position averaged to one
    colour per channel): the decoder that knows only the image's average."""
    p = np.asarray(patches, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    visible = p[~m]
    if not len(visible):
        raise ValueError("no visible patches")
    values = visible.shape[1]
    channels = 3
    per_channel = visible.reshape(-1, values // channels, channels).mean(axis=(0, 1))
    fill = np.tile(per_channel, values // channels)
    out = p.copy()
    out[m] = fill
    return out


def blur_fill(patches: np.ndarray, mask: np.ndarray, *, grid: int) -> np.ndarray:
    """Predict every hidden patch as the mean colour of its visible 8-neighbours in the patch grid (falling back
    to the mean-patch fill where a patch has no visible neighbour)."""
    p = np.asarray(patches, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    if p.shape[0] != grid * grid:
        raise ValueError("patch count must equal grid * grid")
    values = p.shape[1]
    channels = 3
    colours = p.reshape(grid * grid, values // channels, channels).mean(axis=1)  # one colour per patch
    fallback = colours[~m].mean(axis=0) if (~m).any() else np.zeros(channels)
    out = p.copy()
    for index in np.flatnonzero(m):
        row, col = divmod(int(index), grid)
        neighbours = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == dc == 0:
                    continue
                r2, c2 = row + dr, col + dc
                if 0 <= r2 < grid and 0 <= c2 < grid and not m[r2 * grid + c2]:
                    neighbours.append(colours[r2 * grid + c2])
        colour = np.mean(neighbours, axis=0) if neighbours else fallback
        out[index] = np.tile(colour, values // channels)
    return out


# ---------------------------------------------------------------------------------- classification


def _per_class(pred: np.ndarray, gold: np.ndarray, n_classes: int) -> dict[str, Any]:
    out = {}
    for c in range(n_classes):
        tp = int(np.sum((pred == c) & (gold == c)))
        fp = int(np.sum((pred == c) & (gold != c)))
        fn = int(np.sum((pred != c) & (gold == c)))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        out[c] = {"support": int(np.sum(gold == c)), "precision": precision, "recall": recall, "f1": f1}
    return out


def classification_metrics(scores: Any, gold: Sequence[int], classes: Sequence[str]) -> dict[str, Any]:
    """Accuracy, macro F1 and a per-class breakdown for an `[image][class]` score grid."""
    grid = np.asarray(scores, dtype=np.float64)
    gold_arr = np.asarray(gold, dtype=int)
    if grid.ndim != 2 or grid.shape != (len(gold_arr), len(classes)):
        raise ValueError(f"scores must be an [images x classes] grid, got {grid.shape}")
    if not len(gold_arr) or gold_arr.min() < 0 or gold_arr.max() >= len(classes):
        raise ValueError("gold must hold class indices for at least one photograph")
    if not np.all(np.isfinite(grid)):
        raise ValueError("scores must be finite")
    pred = grid.argmax(axis=1)
    per = _per_class(pred, gold_arr, len(classes))
    return {
        "n": int(len(gold_arr)),
        "accuracy": float(np.mean(pred == gold_arr)),
        "macro_f1": float(np.mean([per[c]["f1"] for c in range(len(classes))])),
        "per_class": {classes[c]: per[c] for c in range(len(classes))},
        "definitions": dict(METRIC_DEFINITIONS),
    }


def _scores_from_predictions(pred: Sequence[int], n_classes: int) -> np.ndarray:
    grid = np.zeros((len(pred), n_classes))
    for i, c in enumerate(pred):
        grid[i, c] = 1.0
    return grid


def majority_baseline(
    train: Sequence[Mapping[str, Any]], records: Sequence[Mapping[str, Any]], classes: Sequence[str]
) -> dict[str, Any]:
    """The most frequent training label for every photograph."""
    if not train:
        raise ValueError("the majority baseline needs training records")
    counts = Counter(str(r["label"]) for r in train)
    label = max(sorted(counts), key=counts.get)
    index = list(classes).index(label)
    gold = [list(classes).index(str(r["label"])) for r in records]
    result = classification_metrics(_scores_from_predictions([index] * len(records), len(classes)), gold, classes)
    result["baseline"] = f"majority floor ({label!r} for every photograph)"
    return result


def colour_signature(image: str | Image.Image, *, grid: int = COLOUR_GRID) -> list[float]:
    """Mean RGB of each cell of a `grid` x `grid` partition of the image, in 0..1 (27 numbers by default)."""
    handle = image if isinstance(image, Image.Image) else Image.open(image)
    small = handle.convert("RGB").resize((grid * 8, grid * 8), Image.BILINEAR)
    pixels = list(small.getdata())
    out: list[float] = []
    for row in range(grid):
        for col in range(grid):
            cell = [pixels[(row * 8 + y) * grid * 8 + col * 8 + x] for y in range(8) for x in range(8)]
            out.extend(sum(p[channel] for p in cell) / (64 * 255.0) for channel in range(3))
    return out


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b, strict=True)))


def colour_neighbour_baseline(
    train: Sequence[Mapping[str, Any]], records: Sequence[Mapping[str, Any]], classes: Sequence[str]
) -> dict[str, Any]:
    """Label every photograph with the label of the training photograph whose colour signature is closest;
    the score grid is the negative distance to the nearest training photograph of each class."""
    if not train:
        raise ValueError("the colour-neighbour baseline needs training records")
    class_list = list(classes)
    signatures = [(colour_signature(r["image"]), class_list.index(str(r["label"]))) for r in train]
    grid = np.zeros((len(records), len(class_list)))
    for i, record in enumerate(records):
        query = colour_signature(record["image"])
        best = [math.inf] * len(class_list)
        for sig, c in signatures:
            best[c] = min(best[c], _distance(sig, query))
        grid[i] = [-d for d in best]
    gold = [class_list.index(str(r["label"])) for r in records]
    result = classification_metrics(grid, gold, class_list)
    result["baseline"] = f"colour nearest neighbour ({len(train)} training photographs)"
    return result


def knn_scores(train_features: np.ndarray, train_gold: Sequence[int], features: np.ndarray, n_classes: int, *, k: int = 5) -> np.ndarray:
    """Cosine k-NN vote on L2-normalised features: the score grid is the summed similarity of the k nearest
    training features per class."""
    a = np.asarray(train_features, dtype=np.float64)
    b = np.asarray(features, dtype=np.float64)
    gold = np.asarray(train_gold, dtype=int)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1] or len(gold) != len(a):
        raise ValueError("feature matrices must be [images x dim] with one training label per row")
    sims = b @ a.T
    k = min(k, len(a))
    grid = np.zeros((len(b), n_classes))
    for i in range(len(b)):
        top = np.argsort(-sims[i])[:k]
        for j in top:
            grid[i, gold[j]] += sims[i, j]
    return grid
