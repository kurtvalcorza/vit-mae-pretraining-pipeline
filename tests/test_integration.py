from __future__ import annotations

import os

import numpy as np
import pytest
from PIL import Image

from vit_mae_pipeline import DEFAULT_MASK_RATIO, HIDDEN_SIZE, IMAGE_SIZE, NUM_PATCHES, load_pipeline

pytestmark = pytest.mark.integration


def test_real_checkpoint_smoke():
    if os.environ.get("RUN_INTEGRATION") != "1":
        pytest.skip("set RUN_INTEGRATION=1 to download and run the pinned checkpoint")

    pipe = load_pipeline(device="cpu")
    image = Image.new("RGB", (256, 256), (180, 30, 30))

    # Masked reconstruction: the encoder sees only the kept patches, the decoder fills the rest.
    out = pipe.reconstruct(image, seed=0)
    (entry,) = out["results"]
    hidden = NUM_PATCHES - int(NUM_PATCHES * (1 - DEFAULT_MASK_RATIO))
    assert entry["hidden_patches"] == hidden
    assert sum(entry["mask"]) == hidden
    assert np.isfinite(entry["masked_mse"]) and np.isfinite(entry["visible_mse"])
    assert entry["reconstruction"].size == (IMAGE_SIZE, IMAGE_SIZE)
    assert out["model_loss"] is not None and np.isfinite(out["model_loss"])

    # The same seed hides the same patches and gives the same error.
    again = pipe.reconstruct(image, seed=0, return_images=False)["results"][0]
    assert again["mask"] == entry["mask"]
    assert again["masked_mse"] == pytest.approx(entry["masked_mse"], rel=1e-5)

    # Encoder features with nothing hidden: one L2-normalised vector per image.
    embedded = pipe.embed([image])
    vectors = embedded["embeddings"]
    assert vectors.shape == (1, HIDDEN_SIZE)
    assert embedded["normalized"] is True
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), [1.0], atol=1e-5)
