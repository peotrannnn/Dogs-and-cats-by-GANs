"""sobelv5 Generator — the exact architecture used in notebooks/04_model_training_5th_attempt.ipynb.

`.weights.h5` files only store weights, so the model must be rebuilt with the same layers
(in the same order) before `load_weights()` can work. Keep this file in sync with the notebook.
"""

from __future__ import annotations

IMAGE_SIZE = 128
NOISE_DIM = 100
EMBEDDING_DIM = 50
SPECIES_TO_LABEL = {"dog": 0, "cat": 1}
LABEL_TO_SPECIES = {v: k for k, v in SPECIES_TO_LABEL.items()}
NUM_CLASSES = len(SPECIES_TO_LABEL)


def build_generator(noise_dim: int = NOISE_DIM,
                    num_classes: int = NUM_CLASSES,
                    embedding_dim: int = EMBEDDING_DIM,
                    name: str = "generator"):
    """(z: [B, 100], label: [B] int32) -> image [B, 128, 128, 3] in [-1, 1]."""
    from tensorflow.keras import Model, layers  # imported lazily so non-TF tools stay fast

    noise_input = layers.Input(shape=(noise_dim,), name="noise")
    label_input = layers.Input(shape=(), dtype="int32", name="label")

    label_embed = layers.Embedding(num_classes, embedding_dim)(label_input)
    x = layers.Concatenate()([noise_input, label_embed])

    x = layers.Dense(8 * 8 * 256, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Reshape((8, 8, 256))(x)

    for filters in (128, 64, 32):  # 16x16 -> 32x32 -> 64x64
        x = layers.Conv2DTranspose(filters, 4, strides=2, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)

    output = layers.Conv2DTranspose(3, 4, strides=2, padding="same", activation="tanh")(x)  # 128x128
    return Model([noise_input, label_input], output, name=name)


def to_uint8(images):
    """[-1, 1] float tensor/array -> uint8 numpy array."""
    import numpy as np

    arr = images.numpy() if hasattr(images, "numpy") else np.asarray(images)
    return ((arr + 1.0) * 127.5).clip(0, 255).astype("uint8")


def latent_from_seed(seed: int, n: int = 1):
    """Same convention as src/web_server.py: numpy default_rng(seed).standard_normal."""
    import numpy as np

    return np.random.default_rng(seed).standard_normal((n, NOISE_DIM)).astype("float32")
