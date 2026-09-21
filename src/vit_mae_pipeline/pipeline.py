"""ViT-MAE base: masked-image reconstruction with the pinned ``facebook/vit-mae-base`` weights, a
reconstruction-supervised evaluation contract, a linear-probe reading of the encoder, and a bounded
continuation of the masked-autoencoding pretraining on a labelled photograph set.

The checkpoint is the full pretraining model (``ViTMAEForPreTraining``: a ViT-B/16 encoder and an
8-layer, 512-wide decoder). Its one user-facing operation is **reconstruction**: a seeded random 75 % of the
196 patches is hidden from the encoder, the decoder predicts every patch's pixels, and the loss is the mean
squared error on the hidden patches — the model's own training objective, which is also the only number it
can be read on without labels. ``embed`` exposes the encoder as a feature extractor (mean of the patch
tokens with nothing hidden, L2-normalised), and ``fit_probe`` / ``classify`` / ``evaluate`` read those
features through a linear head the way the fleet reads self-supervised encoders. ``adapt`` continues the
masked-autoencoding objective on the caller's photographs — the decoder and the last encoder blocks — with
validation-loss epoch selection, and the probe is fitted again afterwards so the effect of that continuation
on the features is measured, not assumed.
"""
# ruff: noqa: E501  -- docstrings and record literals kept on single lines at the fleet width

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from .config import (
    DEFAULT_MASK_RATIO,
    DEFAULT_MODEL_KEY,
    ENCODER_LAYERS,
    HIDDEN_SIZE,
    IMAGE_SIZE,
    MODEL_FILENAME,
    MODEL_ID,
    MODEL_LICENSE,
    MODEL_REVISION,
    MODEL_SHA256,
    NUM_PATCHES,
    PATCH_SIZE,
)
from .metrics import (
    RECONSTRUCTION_DEFINITIONS,
    blur_fill,
    classification_metrics,
    knn_scores,
    masked_mse,
    mean_patch_fill,
    reconstruction_metrics,
)
from .model import load_components, stage_missing_files, verify_snapshot

ImageInput = str | Path | bytes | Image.Image

MAX_IMAGE_SIDE = 4096  # pixels; larger images are rejected before any decode-to-tensor work
MIN_IMAGE_SIDE = 16
MAX_BATCH = 64  # images per embed() / reconstruct() call
DEFAULT_TRAINABLE_BLOCKS = 2  # the last two encoder blocks + the encoder LayerNorm train beside the whole decoder
PARAMETER_COUNT = 111_907_840
ENCODER_PARAMETERS = 85_798_656
DECODER_PARAMETERS = 26_109_184
PROBE_STEPS = 300
PROBE_LR = 0.01
PROBE_WEIGHT_DECAY = 1e-3
KNN_K = 5
MAX_EVAL_RECORDS = 5_000
MIN_SCORED_RECORDS = 50  # below this a scored set is labelled a small sample
ARTIFACT_FORMAT = "org.valcorza.vit-mae-base.adapter.v1"
ARTIFACT_FORMAT_VERSION = "1.0"
ARTIFACT_WEIGHTS_NAME = "adapter.safetensors"
ARTIFACT_MANIFEST_NAME = "manifest.json"
GRID = IMAGE_SIZE // PATCH_SIZE

INPUT_SCHEMA: dict[str, Any] = {
    "images": (
        "PIL.Image.Image, raw bytes, or a local path decodable by Pillow; any mode, converted to RGB; "
        "remote URLs are refused"
    ),
    "image_size": (
        f"sides in [{MIN_IMAGE_SIDE}, {MAX_IMAGE_SIDE}] px; the processor resizes every image to {IMAGE_SIZE} x {IMAGE_SIZE} "
        f"(aspect ratio not kept) and normalises with the ImageNet mean / std"
    ),
    "mask_ratio": f"fraction of the {NUM_PATCHES} patches hidden from the encoder, in (0, 1); default {DEFAULT_MASK_RATIO}; the mask is seeded",
    "output": (
        "reconstruct: per image the masked-patch MSE (the model's loss), the visible-patch MSE, the mask, and the "
        "reconstructed image (decoder prediction on hidden patches, the input on visible ones); embed: one "
        f"{HIDDEN_SIZE}-d L2-normalised vector per image; classify (after fit_probe): a score per class"
    ),
    "validation": "size and decodability only; nothing checks what the image shows",
}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _coerce_image(value: ImageInput) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, bytes):
        with Image.open(BytesIO(value)) as handle:
            handle.load()
            return handle.convert("RGB")
    text = str(value)
    if text.startswith(("http://", "https://")):
        raise ValueError("remote URLs are not accepted; pass a local path, bytes or a PIL image")
    path = Path(text)
    if not path.is_file():
        raise FileNotFoundError(f"image not found: {path}")
    with Image.open(path) as handle:
        handle.load()
        return handle.convert("RGB")


def _check_size(image: Image.Image, what: str) -> None:
    w, h = image.size
    if min(w, h) < MIN_IMAGE_SIDE or max(w, h) > MAX_IMAGE_SIDE:
        raise ValueError(f"{what}: sides must be in [{MIN_IMAGE_SIDE}, {MAX_IMAGE_SIDE}] px, got {w} x {h}")


def _check_images(images: Sequence[ImageInput] | ImageInput) -> list[Image.Image]:
    if isinstance(images, str | Path | bytes | Image.Image):
        images = [images]
    if not images:
        raise ValueError("images must contain at least one image")
    if len(images) > MAX_BATCH:
        raise ValueError(f"at most {MAX_BATCH} images per call")
    out = []
    for index, image in enumerate(images):
        coerced = _coerce_image(image)
        _check_size(coerced, f"image {index}")
        out.append(coerced)
    return out


def _check_mask_ratio(mask_ratio: float) -> float:
    if isinstance(mask_ratio, bool) or not isinstance(mask_ratio, int | float) or not 0.0 < float(mask_ratio) < 1.0:
        raise ValueError("mask_ratio must be a number in (0, 1)")
    return float(mask_ratio)


def validate_inputs(
    images: Sequence[ImageInput] | ImageInput,
    *,
    mask_ratio: float = DEFAULT_MASK_RATIO,
    names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Validation stage: the input manifest (schema, per-image observations, verdict), raising exactly what
    `reconstruct` / `embed` would raise."""
    coerced = _check_images(images)
    ratio = _check_mask_ratio(mask_ratio)
    if names is not None and len(names) != len(coerced):
        raise ValueError("names must have one entry per image")
    return {
        "schema": dict(INPUT_SCHEMA),
        "inputs": [{"id": names[i] if names else f"image-{i}", "mode": image.mode, "size": list(image.size)} for i, image in enumerate(coerced)],
        "mask_ratio": ratio,
        "hidden_patches": int(NUM_PATCHES * ratio),
        "verdict": "accepted",
        "findings": [],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }


def _seeded_noise(seed: int, count: int, generator_device: str = "cpu") -> torch.Tensor:
    """The random argument HF's ViTMAE uses to choose the hidden patches, seeded per image."""
    g = torch.Generator(device=generator_device).manual_seed(int(seed))
    return torch.rand(count, NUM_PATCHES, generator=g)


def _record_seed(record_id: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{record_id}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


class ViTMAEPipeline:
    def __init__(
        self,
        model: Any,
        processor: Any,
        *,
        device: str | torch.device = "cpu",
        checkpoint_path: Path | str | None = None,
        checkpoint_source: str | None = None,
        manifest_verified: bool = False,
        weight_sha256: str | None = None,
        weight_size_bytes: int | None = None,
    ) -> None:
        self.model = model
        self.processor = processor
        self.device = torch.device(device)
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else None
        self.checkpoint_source = checkpoint_source
        self.manifest_verified = manifest_verified
        self.weight_sha256 = weight_sha256
        self.weight_size_bytes = weight_size_bytes
        self.adapter: dict[str, Any] | None = None
        self.probe: dict[str, Any] | None = None  # {"classes", "weight", "bias", ...} after fit_probe
        if hasattr(self.model, "parameters"):  # injected fakes in the offline tests carry none
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad_(False)

    @classmethod
    def from_pretrained(
        cls,
        *,
        device: str | torch.device | None = None,
        cache_dir: str | Path | None = None,
        weights_path: str | Path | None = None,
        weights_dir: str | Path | None = None,
        allow_download: bool = False,
    ) -> ViTMAEPipeline:
        """Load the one supported checkpoint (fleet snapshot directory → stage absent entries when allowed →
        verify against the manifest and the pinned digests → load the safetensors file strictly)."""
        if weights_dir is not None:
            if weights_path is not None:
                raise ValueError("pass either weights_dir or weights_path, not both")
            stage_missing_files(weights_dir, allow_download=allow_download)
            verify_snapshot(weights_dir)
            weights_path = weights_dir
        model, processor, target_device, _, metadata = load_components(device=device, cache_dir=cache_dir, weights_path=weights_path, return_metadata=True)
        return cls(
            model,
            processor,
            device=target_device,
            checkpoint_path=metadata.get("checkpoint_path"),
            checkpoint_source=metadata.get("checkpoint_source"),
            manifest_verified=metadata.get("manifest_verified", False),
            weight_sha256=metadata.get("weight_sha256"),
            weight_size_bytes=metadata.get("weight_size_bytes"),
        )

    # ------------------------------------------------------------------ tensors

    def _pixels(self, images: Sequence[Image.Image]) -> torch.Tensor:
        return self.processor(images=list(images), return_tensors="pt")["pixel_values"].to(self.device)

    def _patchify(self, pixels: torch.Tensor) -> torch.Tensor:
        return self.model.patchify(pixels)

    def _forward(self, pixels: torch.Tensor, noise: torch.Tensor, mask_ratio: float) -> Any:
        """One masked forward with the given noise; the mask ratio is set on the config for the call."""
        previous = self.model.config.mask_ratio
        self.model.config.mask_ratio = mask_ratio
        try:
            return self.model(pixel_values=pixels, noise=noise.to(pixels.device))
        finally:
            self.model.config.mask_ratio = previous

    # ------------------------------------------------------------------ inference contract

    def reconstruct(
        self,
        images: Sequence[ImageInput] | ImageInput,
        *,
        mask_ratio: float = DEFAULT_MASK_RATIO,
        seed: int = 0,
        return_images: bool = True,
    ) -> dict[str, Any]:
        """Hide a seeded random `mask_ratio` of the patches, reconstruct, and score against the true pixels."""
        coerced = _check_images(images)
        ratio = _check_mask_ratio(mask_ratio)
        pixels = self._pixels(coerced)
        noise = _seeded_noise(seed, len(coerced))
        with torch.no_grad():
            out = self._forward(pixels, noise, ratio)
            target = self._patchify(pixels)
            pred = out.logits
            mask = out.mask
            filled = pred * mask.unsqueeze(-1) + target * (1 - mask.unsqueeze(-1))
            recon = self.model.unpatchify(filled) if return_images else None
        results = []
        target_np, pred_np, mask_np = target.cpu().numpy(), pred.float().cpu().numpy(), mask.cpu().numpy()
        for i, image in enumerate(coerced):
            masked = masked_mse(pred_np[i], target_np[i], mask_np[i])
            visible = masked_mse(pred_np[i], target_np[i], 1.0 - mask_np[i])
            entry: dict[str, Any] = {
                "input_size": list(image.size),
                "mask_ratio": ratio,
                "hidden_patches": int(mask_np[i].sum()),
                "masked_mse": masked,
                "visible_mse": visible,
                "mask": mask_np[i].astype(bool).tolist(),
            }
            if return_images:
                entry["reconstruction"] = self._to_image(recon[i])
            results.append(entry)
        return {
            "results": results,
            "model_loss": float(out.loss) if out.loss is not None else None,
            "mask_ratio": ratio,
            "seed": int(seed),
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "model_license": MODEL_LICENSE,
        }

    def _to_image(self, pixels: torch.Tensor) -> Image.Image:
        mean = torch.tensor(self.processor.image_mean, dtype=pixels.dtype, device=pixels.device).view(3, 1, 1)
        std = torch.tensor(self.processor.image_std, dtype=pixels.dtype, device=pixels.device).view(3, 1, 1)
        array = ((pixels * std + mean).clamp(0, 1) * 255).round().byte().permute(1, 2, 0).cpu().numpy()
        return Image.fromarray(array)

    def embed(self, images: Sequence[ImageInput] | ImageInput) -> dict[str, Any]:
        """Encoder features with nothing hidden: the mean of the patch tokens after the final LayerNorm, L2-normalised."""
        coerced = _check_images(images)
        with torch.no_grad():
            features = self._features(self._pixels(coerced))
        return {"embeddings": features.cpu().numpy(), "dim": HIDDEN_SIZE, "normalized": True, "pooling": "mean of patch tokens, nothing hidden", "model_id": MODEL_ID, "model_revision": MODEL_REVISION}

    def _features(self, pixels: torch.Tensor) -> torch.Tensor:
        previous = self.model.config.mask_ratio
        self.model.config.mask_ratio = 0.0
        try:
            hidden = self.model.vit(pixel_values=pixels).last_hidden_state
        finally:
            self.model.config.mask_ratio = previous
        pooled = hidden[:, 1:, :].mean(dim=1)
        return torch.nn.functional.normalize(pooled.float(), dim=-1)

    def features(self, records: Sequence[Mapping[str, Any]], *, batch_size: int = 16) -> np.ndarray:
        out = []
        for start in range(0, len(records), batch_size):
            chunk = [_coerce_image(r["image"]) for r in records[start : start + batch_size]]
            with torch.no_grad():
                out.append(self._features(self._pixels(chunk)).cpu().numpy())
        return np.concatenate(out, axis=0) if out else np.zeros((0, HIDDEN_SIZE), dtype=np.float32)

    # ------------------------------------------------------------------ reconstruction evaluation

    def evaluate_reconstruction(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        mask_ratio: float = DEFAULT_MASK_RATIO,
        seed: int = 0,
        batch_size: int = 16,
    ) -> dict[str, Any]:
        """Masked-patch MSE over a validated `{id, image, ...}` set with one seeded mask per record id, beside the
        mean-patch and blur fills scored on the same masks."""
        from .samples import validate_dataset

        checked = validate_dataset(records, min_records=1, max_records=MAX_EVAL_RECORDS, require_labels=False)["records"]
        ratio = _check_mask_ratio(mask_ratio)
        rows, mean_rows, blur_rows = [], [], []
        for start in range(0, len(checked), batch_size):
            chunk = checked[start : start + batch_size]
            pixels = self._pixels([_coerce_image(r["image"]) for r in chunk])
            noise = torch.stack([_seeded_noise(_record_seed(str(r["id"]), seed), 1)[0] for r in chunk])
            with torch.no_grad():
                out = self._forward(pixels, noise, ratio)
                target = self._patchify(pixels).cpu().numpy()
            pred, mask = out.logits.float().cpu().numpy(), out.mask.cpu().numpy()
            for i, record in enumerate(chunk):
                rows.append({"id": str(record["id"]), "masked_mse": masked_mse(pred[i], target[i], mask[i]), "visible_mse": masked_mse(pred[i], target[i], 1.0 - mask[i]), "hidden_patches": int(mask[i].sum())})
                mean_rows.append({"id": str(record["id"]), "masked_mse": masked_mse(mean_patch_fill(target[i], mask[i]), target[i], mask[i]), "visible_mse": 0.0})
                blur_rows.append({"id": str(record["id"]), "masked_mse": masked_mse(blur_fill(target[i], mask[i], grid=GRID), target[i], mask[i]), "visible_mse": 0.0})
        result = reconstruction_metrics(rows, mask_ratio=ratio)
        result["per_image"] = rows
        result["seed"] = int(seed)
        result["baselines"] = {
            "mean_patch_fill": {**reconstruction_metrics(mean_rows, mask_ratio=ratio), "baseline": "every hidden patch = the mean colour of the visible patches"},
            "blur_fill": {**reconstruction_metrics(blur_rows, mask_ratio=ratio), "baseline": "every hidden patch = the mean colour of its visible neighbours"},
        }
        result["verdict"] = "measured" if len(rows) >= MIN_SCORED_RECORDS else "small-sample"
        result["adapted"] = self.adapter is not None
        return result

    # ------------------------------------------------------------------ linear probe

    def fit_probe(
        self,
        train: Sequence[Mapping[str, Any]],
        *,
        steps: int = PROBE_STEPS,
        lr: float = PROBE_LR,
        weight_decay: float = PROBE_WEIGHT_DECAY,
        seed: int = 0,
    ) -> dict[str, Any]:
        """A multinomial logistic-regression head on the frozen, standardised features of a labelled set
        (full-batch Adam, seeded)."""
        from .samples import class_names, validate_dataset

        checked = validate_dataset(train, min_records=MIN_SCORED_RECORDS // 5)["records"]
        classes = class_names(checked)
        gold = torch.tensor([classes.index(str(r["label"])) for r in checked])
        started = time.perf_counter()
        raw = torch.from_numpy(self.features(checked)).float()
        # The MAE paper probes through a BatchNorm without affine parameters; the same standardisation with the
        # training set's statistics, applied to every query, is what makes a linear head on MAE features usable.
        mean, std = raw.mean(dim=0), raw.std(dim=0) + 1e-6
        feats = (raw - mean) / std
        torch.manual_seed(seed)
        head = torch.nn.Linear(HIDDEN_SIZE, len(classes))
        optimiser = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=weight_decay)
        losses = []
        for _ in range(steps):
            optimiser.zero_grad()
            loss = torch.nn.functional.cross_entropy(head(feats), gold)
            loss.backward()
            optimiser.step()
            losses.append(float(loss.detach()))
        self.probe = {
            "classes": classes,
            "weight": head.weight.detach().clone(),
            "bias": head.bias.detach().clone(),
            "feature_mean": mean,
            "feature_std": std,
            "steps": steps,
            "lr": lr,
            "weight_decay": weight_decay,
            "n_train": len(checked),
            "train_loss": [losses[0], losses[-1]],
            "standardisation": "training-set mean / std per feature (the paper's affine-free BatchNorm)",
            "seconds": round(time.perf_counter() - started, 2),
            "train_features": raw.numpy(),
            "train_gold": gold.numpy(),
        }
        return {k: v for k, v in self.probe.items() if k not in ("weight", "bias", "feature_mean", "feature_std", "train_features", "train_gold")}

    def _scores(self, features: np.ndarray) -> np.ndarray:
        if self.probe is None:
            raise ValueError("no probe fitted: call fit_probe() first")
        feats = (torch.from_numpy(np.asarray(features)).float() - self.probe["feature_mean"]) / self.probe["feature_std"]
        logits = feats @ self.probe["weight"].T + self.probe["bias"]
        return torch.softmax(logits, dim=-1).numpy()

    def classify(self, images: Sequence[ImageInput] | ImageInput) -> dict[str, Any]:
        """Scores per class from the fitted probe on the encoder features."""
        coerced = _check_images(images)
        with torch.no_grad():
            feats = self._features(self._pixels(coerced)).cpu().numpy()
        scores = self._scores(feats)
        classes = self.probe["classes"]
        return {
            "classes": list(classes),
            "results": [{"top1": classes[int(row.argmax())], "scores": {c: float(row[j]) for j, c in enumerate(classes)}} for row in scores],
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
        }

    def evaluate(self, records: Sequence[Mapping[str, Any]], *, k: int = KNN_K) -> dict[str, Any]:
        """Probe accuracy / macro F1 on a validated labelled set, with a k-NN on the same features beside it."""
        from .samples import validate_dataset

        if self.probe is None:
            raise ValueError("no probe fitted: call fit_probe() first")
        checked = validate_dataset(records, min_records=1, max_records=MAX_EVAL_RECORDS)["records"]
        classes = list(self.probe["classes"])
        for record in checked:
            if str(record["label"]) not in classes:
                raise ValueError(f"record label {record['label']!r} is not a probe class")
        gold = [classes.index(str(r["label"])) for r in checked]
        feats = self.features(checked)
        probe = classification_metrics(self._scores(feats), gold, classes)
        if self.probe.get("train_features") is not None:
            knn = classification_metrics(knn_scores(self.probe["train_features"], self.probe["train_gold"], feats, len(classes), k=k), gold, classes)
            knn["baseline"] = f"{k}-nearest-neighbour vote on the same features"
            probe["knn"] = knn
        else:
            probe["knn"] = None  # a reloaded probe carries its head and statistics, not the training features
        probe["classes"] = classes
        probe["predictions"] = [classes[int(row.argmax())] for row in self._scores(feats)]
        probe["verdict"] = "measured" if len(checked) >= MIN_SCORED_RECORDS else "small-sample"
        probe["adapted"] = self.adapter is not None
        return probe

    # ------------------------------------------------------------------ adaptation contract

    def _trainable_names(self, trainable_blocks: int) -> list[str]:
        if isinstance(trainable_blocks, bool) or not isinstance(trainable_blocks, int) or not 0 <= trainable_blocks <= ENCODER_LAYERS:
            raise ValueError(f"trainable_blocks must be an int in 0..{ENCODER_LAYERS}")
        first = ENCODER_LAYERS - trainable_blocks
        prefixes = tuple(f"vit.encoder.layer.{k}." for k in range(first, ENCODER_LAYERS)) + ("decoder.",)
        if trainable_blocks:
            prefixes += ("vit.layernorm.",)
        return [name for name, _p in self.model.named_parameters() if name.startswith(prefixes)]

    def adapt(
        self,
        train: Sequence[Mapping[str, Any]],
        val: Sequence[Mapping[str, Any]] | None = None,
        *,
        epochs: int = 5,
        lr: float = 1e-5,
        batch_size: int = 8,
        trainable_blocks: int = DEFAULT_TRAINABLE_BLOCKS,
        mask_ratio: float = DEFAULT_MASK_RATIO,
        seed: int = 0,
        progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Bounded continuation of the masked-autoencoding objective on validated photographs.

        Trains the decoder and the last `trainable_blocks` encoder blocks (with the encoder LayerNorm) on the
        checkpoint's own loss — the masked-patch MSE under a fresh random mask every step (`mask_ratio`) — with
        AdamW at a fixed learning rate, gradient clipping at 1.0, seeded order and masks, no scheduler. Epoch 0
        records the frozen model's validation reconstruction; every epoch is scored on the validation set with
        its fixed seeded masks, and the epoch with the lowest validation masked MSE is kept. Labels are not used.
        Transactional: any failure restores the base tensors. The fitted probe (if any) is discarded, because
        the features it was fitted on no longer exist."""
        from .samples import validate_dataset

        if isinstance(epochs, bool) or not isinstance(epochs, int) or not 1 <= epochs <= 20:
            raise ValueError("epochs must be an int in 1..20")
        if not (0.0 < lr <= 1e-3):
            raise ValueError("lr must be in (0, 1e-3]")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or not 1 <= batch_size <= 64:
            raise ValueError("batch_size must be an int in 1..64")
        ratio = _check_mask_ratio(mask_ratio)
        names = self._trainable_names(trainable_blocks)
        train_checked = validate_dataset(train, require_labels=False)["records"]
        val_checked = validate_dataset(val, min_records=1, max_records=MAX_EVAL_RECORDS, require_labels=False)["records"] if val else []
        model = self.model
        wanted = set(names)
        for name, param in model.named_parameters():
            param.requires_grad_(name in wanted)
        params = [p for p in model.parameters() if p.requires_grad]
        n_trainable = sum(p.numel() for p in params)
        optimiser = torch.optim.AdamW(params, lr=lr, weight_decay=0.05)
        started = time.perf_counter()
        pixels_all = torch.cat([self._pixels([_coerce_image(r["image"])]).cpu() for r in train_checked])

        def score_val() -> dict[str, Any] | None:
            if not val_checked:
                return None
            model.eval()
            result = self.evaluate_reconstruction(val_checked, mask_ratio=ratio, seed=seed)
            return {k: result[k] for k in ("masked_mse", "masked_mse_visible", "psnr_masked", "n")}

        def key(entry: dict[str, Any]) -> float:
            return -entry["val"]["masked_mse"] if entry["val"] else -math.inf

        history: list[dict[str, Any]] = []
        entry: dict[str, Any] = {"epoch": 0, "train_loss": None, "val": score_val(), "note": "frozen model"}
        history.append(entry)
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items() if k in wanted}
        initial_state = {k: v.clone() for k, v in best_state.items()}
        best_epoch = 0
        generator = torch.Generator().manual_seed(seed)
        try:
            for epoch in range(1, epochs + 1):
                model.train()
                order = torch.randperm(len(train_checked), generator=generator).tolist()
                losses = []
                for start in range(0, len(order), batch_size):
                    chunk = order[start : start + batch_size]
                    pixels = pixels_all[chunk].to(self.device)
                    noise = torch.rand(len(chunk), NUM_PATCHES, generator=generator)
                    optimiser.zero_grad(set_to_none=True)
                    loss = self._forward(pixels, noise, ratio).loss
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(params, 1.0)
                    optimiser.step()
                    losses.append(float(loss.detach()))
                model.eval()
                entry = {"epoch": epoch, "train_loss": float(np.mean(losses)), "val": score_val()}
                history.append(entry)
                if progress is not None:
                    progress(entry)
                if val_checked:
                    if key(entry) > key(history[best_epoch]):
                        best_epoch = epoch
                        best_state = {k: v.detach().clone() for k, v in model.state_dict().items() if k in wanted}
                else:
                    best_epoch = epoch
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items() if k in wanted}
            if progress is not None and history:
                pass
        except BaseException:
            model.load_state_dict({**model.state_dict(), **initial_state}, strict=True)
            model.eval()
            for param in model.parameters():
                param.requires_grad_(False)
            raise
        model.load_state_dict({**model.state_dict(), **best_state}, strict=True)
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
        self.probe = None
        self.adapter = {
            "objective": "masked-autoencoding continuation (masked-patch MSE, the checkpoint's own loss)",
            "mask_ratio": ratio,
            "trainable_blocks": trainable_blocks,
            "n_trainable": int(n_trainable),
            "n_total": int(sum(p.numel() for p in model.parameters())),
            "epochs": epochs,
            "lr": lr,
            "batch_size": batch_size,
            "seed": seed,
            "n_train": len(train_checked),
            "n_val": len(val_checked),
            "best_epoch": best_epoch,
            "selection": "lowest validation masked MSE (validation split, fixed seeded masks)" if val_checked else "final epoch (no validation split)",
            "seconds": round(time.perf_counter() - started, 2),
            "history": history,
            "trainable_names": names,
        }
        return dict(self.adapter)

    # ------------------------------------------------------------------ artifact

    def save_artifact(self, output_dir: str | Path, metadata: Mapping[str, Any] | None = None) -> Path:
        """Write the adapted tensors (and the fitted probe, if any) as safetensors plus a base manifest."""
        if self.adapter is None:
            raise ValueError("nothing to save: call adapt() first")
        from safetensors.torch import save_file

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        names = set(self.adapter["trainable_names"])
        tensors = {k: v.detach().cpu().contiguous() for k, v in self.model.state_dict().items() if k in names}
        probe_record = None
        if self.probe is not None:
            tensors["probe.weight"] = self.probe["weight"].detach().cpu().contiguous()
            tensors["probe.bias"] = self.probe["bias"].detach().cpu().contiguous()
            tensors["probe.feature_mean"] = self.probe["feature_mean"].detach().cpu().contiguous()
            tensors["probe.feature_std"] = self.probe["feature_std"].detach().cpu().contiguous()
            probe_record = {k: v for k, v in self.probe.items() if k not in ("weight", "bias", "feature_mean", "feature_std", "train_features", "train_gold")}
        weights_path = out / ARTIFACT_WEIGHTS_NAME
        save_file(tensors, str(weights_path), metadata={"format": "pt"})
        manifest = {
            "format": ARTIFACT_FORMAT,
            "format_version": ARTIFACT_FORMAT_VERSION,
            "base_model": {"id": MODEL_ID, "revision": MODEL_REVISION, "key": DEFAULT_MODEL_KEY, "weight_file": MODEL_FILENAME, "weight_sha256": MODEL_SHA256},
            "adapter": {k: v for k, v in self.adapter.items() if k not in ("history", "trainable_names")},
            "probe": probe_record,
            "history": self.adapter["history"],
            "tensors": sorted(tensors),
            "files": [{"path": ARTIFACT_WEIGHTS_NAME, "bytes": weights_path.stat().st_size, "sha256": _sha256_file(weights_path)}],
            "metadata": dict(metadata or {}),
        }
        (out / ARTIFACT_MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        return out

    def load_artifact(self, artifact_dir: str | Path) -> dict[str, Any]:
        """Verify an adapter's manifest, digest and exact tensor set **before** deserialising, then overwrite
        exactly the tensors it carries (and restore the probe head when the artifact carries one)."""
        root = Path(artifact_dir)
        manifest = json.loads((root / ARTIFACT_MANIFEST_NAME).read_text(encoding="utf-8"))
        weights_path = _check_artifact_manifest(root, manifest)
        entry = manifest["files"][0]
        if not weights_path.is_file():
            raise FileNotFoundError(f"artifact weights missing: {weights_path}")
        if _sha256_file(weights_path) != entry["sha256"] or weights_path.stat().st_size != entry["bytes"]:
            raise ValueError(f"{entry['path']}: digest or size mismatch; refusing to load")
        expected = sorted(self._trainable_names(manifest["adapter"]["trainable_blocks"]))
        probe_record = manifest.get("probe")
        if probe_record:
            expected = sorted(expected + ["probe.weight", "probe.bias", "probe.feature_mean", "probe.feature_std"])
        if sorted(manifest["tensors"]) != expected:
            raise ValueError("artifact tensor list does not match its recorded configuration")
        from safetensors.torch import load_file

        tensors = load_file(str(weights_path))
        if sorted(tensors) != expected:
            raise ValueError("artifact tensor names differ from its manifest")
        state = self.model.state_dict()
        model_tensors = {k: v for k, v in tensors.items() if not k.startswith("probe.")}
        for key, value in model_tensors.items():
            if key not in state or not key.startswith(("vit.encoder.layer.", "vit.layernorm.", "decoder.")):
                raise ValueError(f"artifact tensor {key} is not an adaptable tensor of the base")
            if tuple(value.shape) != tuple(state[key].shape):
                raise ValueError(f"artifact tensor {key} has shape {tuple(value.shape)}, base has {tuple(state[key].shape)}")
        merged = dict(state)
        merged.update({k: v.to(state[k].dtype) for k, v in model_tensors.items()})
        self.model.load_state_dict(merged, strict=True)
        self.model.eval()
        self.adapter = {**manifest["adapter"], "trainable_names": [k for k in manifest["tensors"] if not k.startswith("probe.")], "history": manifest.get("history", [])}
        self.probe = None
        if probe_record:
            weight, bias = tensors["probe.weight"], tensors["probe.bias"]
            mean, std = tensors["probe.feature_mean"], tensors["probe.feature_std"]
            if tuple(weight.shape) != (len(probe_record["classes"]), HIDDEN_SIZE) or tuple(bias.shape) != (len(probe_record["classes"]),) or tuple(mean.shape) != (HIDDEN_SIZE,) or tuple(std.shape) != (HIDDEN_SIZE,):
                raise ValueError("artifact probe head has an unexpected shape")
            self.probe = {**probe_record, "weight": weight.float(), "bias": bias.float(), "feature_mean": mean.float(), "feature_std": std.float(), "train_features": None, "train_gold": None}
        return manifest

    @classmethod
    def from_artifact(
        cls,
        artifact_dir: str | Path,
        *,
        device: str | torch.device | None = None,
        weights_dir: str | Path | None = None,
        allow_download: bool = False,
    ) -> ViTMAEPipeline:
        """A fresh pipeline from the pinned base with an adapter overlaid."""
        pipe = cls.from_pretrained(device=device, weights_dir=weights_dir, allow_download=allow_download)
        pipe.load_artifact(artifact_dir)
        return pipe


def _check_artifact_manifest(root: Path, manifest: Mapping[str, Any]) -> Path:
    """Refuse an artifact whose manifest is not exactly the one this package writes. Nothing is deserialised
    here; the digest check that follows detects corruption or drift relative to the adjacent manifest."""
    if manifest.get("format") != ARTIFACT_FORMAT:
        raise ValueError(f"artifact format {manifest.get('format')!r} != {ARTIFACT_FORMAT!r}")
    if manifest.get("format_version") != ARTIFACT_FORMAT_VERSION:
        raise ValueError(f"artifact format_version {manifest.get('format_version')!r} is not the supported {ARTIFACT_FORMAT_VERSION!r}")
    base = manifest.get("base_model", {})
    if (base.get("id"), base.get("revision"), base.get("weight_sha256")) != (MODEL_ID, MODEL_REVISION, MODEL_SHA256):
        raise ValueError("artifact was adapted from a different base model, revision or weight file")
    if base.get("weight_file", MODEL_FILENAME) != MODEL_FILENAME:
        raise ValueError("artifact was adapted from a different base weight file")
    files = manifest.get("files")
    if not isinstance(files, list) or len(files) != 1:
        raise ValueError("artifact manifest must list exactly one file")
    entry = files[0]
    if not isinstance(entry, Mapping) or entry.get("path") != ARTIFACT_WEIGHTS_NAME:
        raise ValueError(f"artifact manifest must name exactly {ARTIFACT_WEIGHTS_NAME!r}")
    weights_path = (root / entry["path"]).resolve()
    if weights_path.parent != root.resolve():
        raise ValueError("artifact weight path must resolve inside the artifact directory")
    adapter = manifest.get("adapter")
    blocks = adapter.get("trainable_blocks") if isinstance(adapter, Mapping) else None
    if isinstance(blocks, bool) or not isinstance(blocks, int) or not 0 <= blocks <= ENCODER_LAYERS:
        raise ValueError("artifact manifest does not record an in-range integer trainable_blocks")
    probe = manifest.get("probe")
    if probe is not None and (not isinstance(probe, Mapping) or not isinstance(probe.get("classes"), list) or not probe["classes"]):
        raise ValueError("artifact probe record must list its classes")
    if not isinstance(manifest.get("tensors"), list):
        raise ValueError("artifact manifest must list its tensors")
    return weights_path


def load_pipeline(**kwargs: Any) -> ViTMAEPipeline:
    return ViTMAEPipeline.from_pretrained(**kwargs)


def evaluation_report(result: Mapping[str, Any], *, sample_kind: str = "synthetic") -> dict[str, Any]:
    """The per-call reading of a `reconstruct` result: masked and visible MSE per image and their means, with a
    `sample-sanity` verdict on a drawing (plumbing evidence) or `measured` on a scored set."""
    rows = result.get("results", [])
    if not rows:
        raise ValueError("no results to report")
    masked = [float(r["masked_mse"]) for r in rows]
    visible = [float(r["visible_mse"]) for r in rows]
    metrics = [
        {"id": "masked_mse", "value": float(np.mean(masked)), "definition": RECONSTRUCTION_DEFINITIONS["masked_mse"]},
        {"id": "masked_mse_visible", "value": float(np.mean(visible)), "definition": RECONSTRUCTION_DEFINITIONS["masked_mse_visible"]},
        {"id": "mask_ratio", "value": float(result.get("mask_ratio", DEFAULT_MASK_RATIO)), "definition": RECONSTRUCTION_DEFINITIONS["mask_ratio"]},
    ]
    return {
        "metrics": metrics,
        "n": len(rows),
        "sample_kind": sample_kind,
        "verdict": "sample-sanity" if sample_kind.startswith("synthetic") or len(rows) < MIN_SCORED_RECORDS else "measured",
        "note": "a reconstruction loss is the model's own objective, not a quality judgement; lower is better and nothing here is calibrated",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }


__all__ = [
    "ARTIFACT_FORMAT",
    "DEFAULT_TRAINABLE_BLOCKS",
    "INPUT_SCHEMA",
    "MAX_BATCH",
    "MAX_IMAGE_SIDE",
    "MIN_IMAGE_SIDE",
    "PARAMETER_COUNT",
    "ViTMAEPipeline",
    "evaluation_report",
    "load_pipeline",
    "validate_inputs",
]
