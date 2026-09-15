from __future__ import annotations

import base64
import io
import json
import re
import secrets
import threading
from pathlib import Path

import numpy as np
import tensorflow as tf
from flask import Flask, Response, jsonify, request, send_from_directory, stream_with_context
from PIL import Image
from tensorflow.keras import Model, layers


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]
WEB_DIR = PROJECT_ROOT / "web"

MODEL_ROOT_CANDIDATES = [
    PROJECT_ROOT / "models" / "sobelv5",
    PROJECT_ROOT / "model" / "sobelv5",
    PROJECT_ROOT / "model",
]


def find_model_root() -> Path:
    for root in MODEL_ROOT_CANDIDATES:
        if (root / "stage1").exists() or (root / "stage2").exists() or (root / "stage3").exists():
            return root

    attempted = "\n".join(f"  - {p}" for p in MODEL_ROOT_CANDIDATES)
    raise FileNotFoundError(
        "Could not find sobelv5 model directory.\n"
        "Tried:\n"
        f"{attempted}"
    )


MODEL_ROOT = find_model_root()


# ---------------------------------------------------------------------
# sobelv5 Generator — matches 04_model_training_5th_attempt.ipynb
# ---------------------------------------------------------------------

IMAGE_SIZE = 128
NOISE_DIM = 100
EMBEDDING_DIM = 50

SPECIES_TO_LABEL = {
    "dog": 0,
    "cat": 1,
}
NUM_CLASSES = len(SPECIES_TO_LABEL)


def build_generator(
    noise_dim: int = NOISE_DIM,
    num_classes: int = NUM_CLASSES,
    embedding_dim: int = EMBEDDING_DIM,
    name: str = "generator",
) -> Model:
    noise_input = layers.Input(shape=(noise_dim,), name="noise")
    label_input = layers.Input(shape=(), dtype="int32", name="label")

    label_embed = layers.Embedding(num_classes, embedding_dim)(label_input)
    x = layers.Concatenate()([noise_input, label_embed])

    x = layers.Dense(8 * 8 * 256, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Reshape((8, 8, 256))(x)

    x = layers.Conv2DTranspose(
        128, 4, strides=2, padding="same", use_bias=False
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Conv2DTranspose(
        64, 4, strides=2, padding="same", use_bias=False
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Conv2DTranspose(
        32, 4, strides=2, padding="same", use_bias=False
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    output = layers.Conv2DTranspose(
        3,
        4,
        strides=2,
        padding="same",
        activation="tanh",
    )(x)

    return Model([noise_input, label_input], output, name=name)


# ---------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------

def epoch_from_name(path: Path) -> int | None:
    match = re.search(r"_e(\d+)\.weights\.h5$", path.name)
    return int(match.group(1)) if match else None


def stage_snapshots(stage: str) -> list[dict]:
    """
    stage1: use raw Generator snapshots.
    stage2/3: prefer EMA snapshots because training sampled/evaluated EMA.
    """
    snap_dir = MODEL_ROOT / stage / "snapshots"
    if not snap_dir.exists():
        return []

    if stage == "stage1":
        paths = sorted(
            snap_dir.glob("generator_e*.weights.h5"),
            key=lambda p: epoch_from_name(p) or -1,
        )
        kind = "G"
    else:
        ema_paths = sorted(
            snap_dir.glob("generator_ema_e*.weights.h5"),
            key=lambda p: epoch_from_name(p) or -1,
        )
        if ema_paths:
            paths = ema_paths
            kind = "EMA"
        else:
            paths = sorted(
                snap_dir.glob("generator_e*.weights.h5"),
                key=lambda p: epoch_from_name(p) or -1,
            )
            kind = "G"

    return [
        {
            "stage": stage,
            "epoch": epoch_from_name(path),
            "kind": kind,
            "path": path,
        }
        for path in paths
        if epoch_from_name(path) is not None
    ]


def pick_nearest(records: list[dict], targets: list[int]) -> list[dict]:
    if not records:
        return []

    selected = []
    used_paths = set()

    for target in targets:
        record = min(records, key=lambda item: abs(item["epoch"] - target))
        key = str(record["path"].resolve())
        if key not in used_paths:
            selected.append(record)
            used_paths.add(key)

    return sorted(selected, key=lambda item: item["epoch"])


def discover_evolution_sequence() -> list[dict]:
    """
    Keep the early/middle progression visible without loading every stored snapshot.
    These targets match v5's permanent-snapshot cadence:
      stage1: every 25 epochs
      stage2: every 25 epochs
      stage3: every 10 epochs
    """
    stage1 = pick_nearest(
        stage_snapshots("stage1"),
        [25, 50, 75, 100, 125, 150],
    )
    stage2 = pick_nearest(
        stage_snapshots("stage2"),
        [25, 75, 125, 200],
    )
    stage3 = pick_nearest(
        stage_snapshots("stage3"),
        [10, 30, 60, 100],
    )

    sequence = stage1 + stage2 + stage3

    final_best = MODEL_ROOT / "stage3" / "best" / "generator_best.weights.h5"
    if final_best.is_file():
        best_epoch = None
        info_path = MODEL_ROOT / "stage3" / "best" / "best_info.json"

        if info_path.is_file():
            try:
                info = json.loads(info_path.read_text(encoding="utf-8"))
                best_epoch = int(info.get("epoch")) if info.get("epoch") is not None else None
            except Exception:
                best_epoch = None

        sequence.append(
            {
                "stage": "stage3",
                "epoch": best_epoch,
                "kind": "BEST",
                "path": final_best,
                "final": True,
            }
        )

    # Fallback when only a final/rolling model is present.
    if not sequence:
        fallback_candidates = [
            MODEL_ROOT / "stage3" / "generator_ema.weights.h5",
            MODEL_ROOT / "stage3" / "generator.weights.h5",
            MODEL_ROOT / "stage2" / "generator_ema.weights.h5",
            MODEL_ROOT / "stage2" / "generator.weights.h5",
            MODEL_ROOT / "stage1" / "generator.weights.h5",
        ]
        for path in fallback_candidates:
            if path.is_file():
                sequence.append(
                    {
                        "stage": path.parent.name,
                        "epoch": None,
                        "kind": "LATEST",
                        "path": path,
                        "final": True,
                    }
                )
                break

    return sequence


EVOLUTION_SEQUENCE = discover_evolution_sequence()

if not EVOLUTION_SEQUENCE:
    raise FileNotFoundError(
        f"No usable Generator weights found under {MODEL_ROOT}"
    )


def checkpoint_label(item: dict) -> str:
    stage = item["stage"].replace("stage", "Stage ")
    epoch = item.get("epoch")
    kind = item.get("kind", "G")

    if kind == "BEST":
        return f"{stage} · best" + (f" (epoch {epoch})" if epoch is not None else "")
    if epoch is not None:
        return f"{stage} · epoch {epoch} · {kind}"
    return f"{stage} · {kind}"


# ---------------------------------------------------------------------
# Single reusable Generator
# ---------------------------------------------------------------------

GENERATOR = build_generator()

# Loading weights into the same Generator is much lighter than keeping
# 10–15 complete Generator objects in memory.
MODEL_LOCK = threading.Lock()

FINAL_CHECKPOINT = EVOLUTION_SEQUENCE[-1]["path"]
GENERATOR.load_weights(FINAL_CHECKPOINT)

_warmup_z = tf.zeros([1, NOISE_DIM], dtype=tf.float32)
_warmup_label = tf.constant([0], dtype=tf.int32)
_ = GENERATOR([_warmup_z, _warmup_label], training=False)

print("=" * 72)
print("sobelv5 evolution server")
print("Model root:", MODEL_ROOT)
print("Evolution checkpoints:")
for index, item in enumerate(EVOLUTION_SEQUENCE, start=1):
    print(f"  {index:02d}. {checkpoint_label(item)}")
    print(f"      {item['path']}")
print("=" * 72)


# ---------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------

def latent_from_seed(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((1, NOISE_DIM)).astype(np.float32)


def tensor_to_png_data_url(image_tensor: tf.Tensor) -> str:
    image = image_tensor.numpy()[0]
    image = ((image + 1.0) * 127.5).clip(0, 255).astype(np.uint8)

    pil_image = Image.fromarray(image, mode="RGB")
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG", optimize=True)

    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def generate_with_current_weights(z: np.ndarray, label: int) -> str:
    labels = np.array([label], dtype=np.int32)
    output = GENERATOR(
        [tf.constant(z), tf.constant(labels)],
        training=False,
    )
    return tensor_to_png_data_url(output)


# ---------------------------------------------------------------------
# Flask
# ---------------------------------------------------------------------

app = Flask(
    __name__,
    static_folder=str(WEB_DIR),
    static_url_path="",
)


@app.get("/")
def index():
    return send_from_directory(WEB_DIR, "index.html")


@app.get("/api/health")
def health():
    return jsonify(
        {
            "ok": True,
            "model": "sobelv5",
            "image_size": IMAGE_SIZE,
            "noise_dim": NOISE_DIM,
            "model_root": str(MODEL_ROOT),
            "frames": len(EVOLUTION_SEQUENCE),
            "sequence": [
                {
                    "label": checkpoint_label(item),
                    "stage": item["stage"],
                    "epoch": item.get("epoch"),
                    "kind": item.get("kind"),
                    "path": str(item["path"]),
                }
                for item in EVOLUTION_SEQUENCE
            ],
        }
    )


@app.post("/api/evolution")
def evolution():
    payload = request.get_json(silent=True) or {}

    species = str(payload.get("species", "")).strip().lower()
    if species not in SPECIES_TO_LABEL:
        return jsonify(
            {
                "ok": False,
                "error": "species must be 'dog' or 'cat'",
            }
        ), 400

    requested_seed = payload.get("seed", None)

    if requested_seed is None or requested_seed == "":
        seed = secrets.randbelow(2_147_483_647)
    else:
        try:
            seed = int(requested_seed)
        except (TypeError, ValueError):
            return jsonify(
                {
                    "ok": False,
                    "error": "seed must be an integer",
                }
            ), 400

    label = SPECIES_TO_LABEL[species]
    z = latent_from_seed(seed)

    @stream_with_context
    def stream():
        total = len(EVOLUTION_SEQUENCE)

        yield json.dumps(
            {
                "type": "start",
                "ok": True,
                "model": "sobelv5",
                "species": species,
                "label": label,
                "seed": seed,
                "total_frames": total,
            }
        ) + "\n"

        with MODEL_LOCK:
            try:
                for index, item in enumerate(EVOLUTION_SEQUENCE, start=1):
                    GENERATOR.load_weights(item["path"])
                    image = generate_with_current_weights(z, label)

                    yield json.dumps(
                        {
                            "type": "frame",
                            "index": index,
                            "total": total,
                            "progress": round(index / total * 100, 2),
                            "stage": item["stage"],
                            "epoch": item.get("epoch"),
                            "kind": item.get("kind"),
                            "label_text": checkpoint_label(item),
                            "image": image,
                            "is_final": bool(item.get("final", False)),
                        }
                    ) + "\n"

                yield json.dumps(
                    {
                        "type": "done",
                        "ok": True,
                        "seed": seed,
                        "species": species,
                    }
                ) + "\n"

            except Exception as exc:
                app.logger.exception("Evolution generation failed")
                yield json.dumps(
                    {
                        "type": "error",
                        "ok": False,
                        "error": str(exc),
                    }
                ) + "\n"

            finally:
                # Put the reusable model back on the final/best weights.
                try:
                    GENERATOR.load_weights(FINAL_CHECKPOINT)
                except Exception:
                    app.logger.exception("Could not restore final checkpoint")

    return Response(
        stream(),
        mimetype="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/<path:filename>")
def static_files(filename: str):
    return send_from_directory(WEB_DIR, filename)


if __name__ == "__main__":
    print()
    print("Open in browser:")
    print("  http://127.0.0.1:5000")
    print()
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        threaded=True,
    )
