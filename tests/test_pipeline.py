"""Offline tests of the inference contract on a tiny stand-in for ViTMAEForPreTraining: the reconstruction
call, the masked / visible MSE bookkeeping, seeded masks, the embedding pooling and normalisation, input
rejections and batch ceilings. Nothing here loads the checkpoint."""
# ruff: noqa: E501

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from vit_mae_pipeline import (
    DEFAULT_MASK_RATIO,
    HIDDEN_SIZE,
    MAX_BATCH,
    MODEL_ID,
    MODEL_REVISION,
    NUM_PATCHES,
    ViTMAEPipeline,
    validate_inputs,
)

PATCH_VALUES = 16 * 16 * 3


class FakeProcessor:
    """Resizes to 224 x 224 like the real processor (no normalisation, so pixels stay readable)."""

    image_mean = [0.0, 0.0, 0.0]
    image_std = [1.0, 1.0, 1.0]

    def __call__(self, *, images=None, return_tensors="pt", **kwargs):
        arrays = [np.asarray(image.convert("RGB").resize((224, 224)), dtype=np.float32) / 255.0 for image in images]
        return {"pixel_values": torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2).contiguous()}


class _FakeVit(torch.nn.Module):
    """A stand-in encoder: 197 tokens whose patch tokens are a linear map of the patch pixels."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.proj = torch.nn.Linear(PATCH_VALUES, HIDDEN_SIZE, bias=False)
        self.encoder = torch.nn.Module()
        self.encoder.layer = torch.nn.ModuleList([torch.nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE) for _ in range(12)])
        self.layernorm = torch.nn.LayerNorm(HIDDEN_SIZE)

    def forward(self, pixel_values):
        patches = FakeModel.patchify_static(pixel_values)
        tokens = self.proj(patches)
        cls = torch.zeros(tokens.shape[0], 1, HIDDEN_SIZE)
        return SimpleNamespace(last_hidden_state=self.layernorm(torch.cat([cls, tokens], dim=1)))


class FakeModel(torch.nn.Module):
    """The surface `ViTMAEPipeline` uses: config.mask_ratio, patchify / unpatchify, a masked forward with
    `noise` returning loss / logits / mask, and `vit(pixel_values=)` for the embedding path. The decoder
    'predicts' every patch as its mean colour, so the masked MSE is computable by hand."""

    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(mask_ratio=DEFAULT_MASK_RATIO)
        self.vit = _FakeVit()
        self.decoder = torch.nn.Module()
        self.decoder.decoder_pred = torch.nn.Linear(4, PATCH_VALUES)
        self.calls: list[dict] = []

    @staticmethod
    def patchify_static(pixel_values):
        b, c, h, w = pixel_values.shape
        p = 16
        x = pixel_values.reshape(b, c, h // p, p, w // p, p).permute(0, 2, 4, 3, 5, 1)
        return x.reshape(b, (h // p) * (w // p), p * p * c)

    def patchify(self, pixel_values):
        return self.patchify_static(pixel_values)

    def unpatchify(self, patches):
        b = patches.shape[0]
        p, g = 16, 14
        x = patches.reshape(b, g, g, p, p, 3).permute(0, 5, 1, 3, 2, 4)
        return x.reshape(b, 3, g * p, g * p)

    def forward(self, *, pixel_values, noise):
        self.calls.append({"mask_ratio": self.config.mask_ratio, "noise": noise.clone()})
        target = self.patchify(pixel_values)
        len_keep = int(NUM_PATCHES * (1 - self.config.mask_ratio))
        order = torch.argsort(noise, dim=1)
        mask = torch.ones(noise.shape)
        mask.scatter_(1, order[:, :len_keep], 0.0)
        colour = target.reshape(target.shape[0], NUM_PATCHES, 256, 3).mean(dim=2, keepdim=True).expand(-1, -1, 256, -1)
        logits = colour.reshape(target.shape)
        loss = (((logits - target) ** 2).mean(dim=-1) * mask).sum() / mask.sum()
        return SimpleNamespace(loss=loss, logits=logits, mask=mask)


def _image(colour=(120, 30, 10), size=(160, 120)) -> Image.Image:
    image = Image.new("RGB", size, colour)
    image.putpixel((0, 0), (255, 255, 255))
    return image


def _pipeline() -> ViTMAEPipeline:
    return ViTMAEPipeline(FakeModel(), FakeProcessor(), device="cpu")


def test_reconstruct_reports_masked_and_visible_errors_with_a_seeded_mask():
    pipe = _pipeline()
    result = pipe.reconstruct(_image(), seed=3)
    entry = result["results"][0]
    assert entry["hidden_patches"] == int(NUM_PATCHES * DEFAULT_MASK_RATIO) == sum(entry["mask"])
    assert entry["masked_mse"] >= 0.0 and entry["visible_mse"] >= 0.0
    assert entry["reconstruction"].size == (224, 224)
    assert result["model_id"] == MODEL_ID and result["model_revision"] == MODEL_REVISION
    assert abs(result["model_loss"] - entry["masked_mse"]) < 1e-5  # the pipeline's masked MSE is the model's own loss
    again = pipe.reconstruct(_image(), seed=3)
    assert again["results"][0]["mask"] == entry["mask"]  # the same seed hides the same patches
    other = pipe.reconstruct(_image(), seed=4)
    assert other["results"][0]["mask"] != entry["mask"]


def test_mask_ratio_is_passed_for_the_call_and_restored():
    pipe = _pipeline()
    pipe.reconstruct(_image(), mask_ratio=0.5, seed=0)
    assert pipe.model.calls[-1]["mask_ratio"] == 0.5 and pipe.model.config.mask_ratio == DEFAULT_MASK_RATIO
    assert pipe.reconstruct(_image(), mask_ratio=0.5, seed=0)["results"][0]["hidden_patches"] == 98
    for bad in (0.0, 1.0, -0.1, "0.5", True):
        with pytest.raises(ValueError, match="mask_ratio"):
            pipe.reconstruct(_image(), mask_ratio=bad)


def test_embeddings_are_mean_pooled_patch_tokens_and_l2_normalised():
    pipe = _pipeline()
    result = pipe.embed([_image(), _image((10, 200, 30))])
    assert result["embeddings"].shape == (2, HIDDEN_SIZE) and result["normalized"] is True
    assert np.allclose(np.linalg.norm(result["embeddings"], axis=1), 1.0, atol=1e-5)
    assert not np.allclose(result["embeddings"][0], result["embeddings"][1])
    assert pipe.model.config.mask_ratio == DEFAULT_MASK_RATIO  # the embedding path hides nothing and restores the ratio


def test_remote_urls_empty_batches_and_oversized_batches_are_rejected():
    pipe = _pipeline()
    with pytest.raises(ValueError, match="remote URLs"):
        pipe.reconstruct("https://example.invalid/x.png")
    with pytest.raises(ValueError, match="at least one image"):
        pipe.embed([])
    with pytest.raises(ValueError, match=f"at most {MAX_BATCH}"):
        pipe.embed([_image()] * (MAX_BATCH + 1))
    with pytest.raises(FileNotFoundError):
        pipe.reconstruct("nowhere/nothing.png")
    with pytest.raises(ValueError, match="sides must be"):
        pipe.reconstruct(Image.new("RGB", (8, 8)))


def test_validate_inputs_mirrors_the_checks_and_names_inputs():
    manifest = validate_inputs([_image(), _image()], names=["a", "b"])
    assert [i["id"] for i in manifest["inputs"]] == ["a", "b"] and manifest["verdict"] == "accepted"
    assert manifest["hidden_patches"] == 147 and manifest["model_id"] == MODEL_ID
    with pytest.raises(ValueError, match="one entry per image"):
        validate_inputs(_image(), names=["a", "b"])


def test_classify_and_evaluate_need_a_fitted_probe():
    pipe = _pipeline()
    with pytest.raises(ValueError, match="fit_probe"):
        pipe.classify(_image())
    with pytest.raises(ValueError, match="fit_probe"):
        pipe.evaluate([{"id": "a", "image": _image(), "label": "x"}])
