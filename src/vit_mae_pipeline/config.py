# ruff: noqa: E501
from __future__ import annotations

MODEL_ID = "facebook/vit-mae-base"
MODEL_REVISION = "25b184bea5538bf5c4c852c79d221195fdd2778d"
MODEL_FILENAME = "model.safetensors"
MODEL_SHA256 = "479dcef4bd5df06259399027b789f21e9d9a1b79f37155a64176d55bc26fdae8"
MODEL_SIZE_BYTES = 447_670_680
MODEL_LICENSE = "Apache-2.0"

DEFAULT_MODEL_KEY = "vit-mae-base"
UNSAFE_WEIGHT_EXTENSIONS = (
    ".bin",
    ".pt",
    ".pth",
    ".ckpt",
    ".pkl",
    ".pickle",
    ".h5",
    ".msgpack",
)

ALLOWED_CHECKPOINT_FILES = (
    "config.json",
    MODEL_FILENAME,
    "preprocessor_config.json",
)

# The checkpoint's own geometry (config.json at the pinned revision).
IMAGE_SIZE = 224
PATCH_SIZE = 16
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2  # 196
HIDDEN_SIZE = 768
ENCODER_LAYERS = 12
DECODER_HIDDEN_SIZE = 512
DECODER_LAYERS = 8
DEFAULT_MASK_RATIO = 0.75  # config.json mask_ratio: 147 of the 196 patches are hidden from the encoder
NORM_PIX_LOSS = False  # config.json norm_pix_loss: the loss is a raw-pixel MSE on the masked patches
