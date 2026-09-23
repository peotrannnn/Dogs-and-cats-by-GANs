#!/usr/bin/env python
"""Lệnh tiện dụng cho dự án Dogs & Cats by GANs (sobelv5).

Chạy từ thư mục gốc của dự án:

    python scripts/gan.py info                       # kiểm tra checkpoint, dữ liệu, thư viện
    python scripts/gan.py generate --species cat -n 16 --seed 7
    python scripts/gan.py evolution --species dog --seeds 42 7
    python scripts/gan.py interpolate --species cat --seeds 1 2 3
    python scripts/gan.py history                    # vẽ loss/metric từ history.csv
    python scripts/gan.py serve                      # mở web demo

Mọi ảnh được lưu vào thư mục outputs/ (đã bị .gitignore).
Gõ `python scripts/gan.py <lệnh> -h` để xem đầy đủ tham số của từng lệnh.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import runpy
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.generator import (  # noqa: E402  (needs sys.path above)
    IMAGE_SIZE,
    SPECIES_TO_LABEL,
    build_generator,
    latent_from_seed,
    to_uint8,
)

MODEL_ROOT = PROJECT_ROOT / "models" / "sobelv5"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
STAGES = ("stage1", "stage2", "stage3")
EVOLUTION_TARGETS = {  # same milestones as src/web_server.py
    "stage1": [25, 50, 75, 100, 125, 150],
    "stage2": [25, 75, 125, 200],
    "stage3": [10, 30, 60, 100],
}


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def best_path(stage: str) -> Path:
    return MODEL_ROOT / stage / "best" / "generator_best.weights.h5"


def best_info(stage: str) -> dict | None:
    path = MODEL_ROOT / stage / "best" / "best_info.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def snapshot_records(stage: str) -> list[dict]:
    """stage1 -> raw G snapshots; stage2/3 -> EMA snapshots if present (that is what training sampled)."""
    snap_dir = MODEL_ROOT / stage / "snapshots"
    if not snap_dir.is_dir():
        return []

    def collect(pattern: str, kind: str) -> list[dict]:
        records = []
        for path in snap_dir.glob(pattern):
            match = re.search(r"_e(\d+)\.weights\.h5$", path.name)
            if match:
                records.append({"stage": stage, "epoch": int(match.group(1)), "kind": kind, "path": path})
        return sorted(records, key=lambda r: r["epoch"])

    if stage != "stage1":
        ema = collect("generator_ema_e*.weights.h5", "EMA")
        if ema:
            return ema
    return collect("generator_e*.weights.h5", "G")


def resolve_checkpoint(spec: str) -> Path:
    """'best' | 'stage1' | 'stage2' | 'stage3' | 'stage2:e150' | path/to/file.weights.h5"""
    if spec in ("best", "final"):
        spec = "stage3"
    if spec in STAGES:
        path = best_path(spec)
    elif re.fullmatch(r"stage[123]:e?\d+", spec):
        stage, epoch = spec.split(":")
        epoch = int(epoch.lstrip("e"))
        matches = [r for r in snapshot_records(stage) if r["epoch"] == epoch]
        if not matches:
            available = [r["epoch"] for r in snapshot_records(stage)] or "không có snapshot"
            sys.exit(f"Không tìm thấy snapshot {stage} epoch {epoch}. Có sẵn: {available}")
        path = matches[0]["path"]
    else:
        path = Path(spec)
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
    if not path.is_file():
        sys.exit(f"Không tìm thấy checkpoint: {path}\nChạy `python scripts/gan.py info` để xem checkpoint nào đang có.")
    return path


def evolution_sequence() -> list[dict]:
    """Snapshots nearest to the milestones; falls back to the three stage-best files (GitHub clone)."""
    sequence = []
    for stage in STAGES:
        records = snapshot_records(stage)
        chosen = {}
        for target in EVOLUTION_TARGETS[stage]:
            if records:
                rec = min(records, key=lambda r: abs(r["epoch"] - target))
                chosen[rec["epoch"]] = rec
        sequence.extend(chosen[e] for e in sorted(chosen))

    if not sequence:  # no snapshots in the repo -> use each stage's best checkpoint
        for stage in STAGES:
            if best_path(stage).is_file():
                info = best_info(stage) or {}
                sequence.append({"stage": stage, "epoch": info.get("epoch"), "kind": "best",
                                 "path": best_path(stage)})
        return sequence

    if best_path("stage3").is_file():
        info = best_info("stage3") or {}
        sequence.append({"stage": "stage3", "epoch": info.get("epoch"), "kind": "best",
                         "path": best_path("stage3")})
    return sequence


def frame_label(rec: dict) -> str:
    stage = rec["stage"].replace("stage", "S")
    epoch = rec.get("epoch")
    if rec["kind"] == "best":
        return f"{stage} best" + (f" e{epoch}" if epoch is not None else "")
    return f"{stage} e{epoch}"


# ---------------------------------------------------------------------------
# Image helpers (Pillow only)
# ---------------------------------------------------------------------------

def make_grid(images, cols: int, labels=None, upscale: int = 1, pad: int = 4):
    from PIL import Image, ImageDraw

    n = len(images)
    cols = max(1, min(cols, n))
    rows = (n + cols - 1) // cols
    size = IMAGE_SIZE * upscale
    label_h = 18 if labels else 0
    canvas = Image.new("RGB", (cols * (size + pad) + pad, rows * (size + pad + label_h) + pad), "white")
    draw = ImageDraw.Draw(canvas)
    for i, arr in enumerate(images):
        tile = Image.fromarray(arr)
        if upscale > 1:
            tile = tile.resize((size, size), Image.NEAREST)
        x = pad + (i % cols) * (size + pad)
        y = pad + (i // cols) * (size + pad + label_h)
        canvas.paste(tile, (x, y + label_h))
        if labels:
            draw.text((x + 2, y + 3), str(labels[i]), fill=(40, 40, 40))
    return canvas


def output_path(user_path: str | None, prefix: str) -> Path:
    if user_path:
        path = Path(user_path)
    else:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = OUTPUT_DIR / f"{prefix}_{stamp}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def load_generator(path: Path):
    generator = build_generator()
    generator.load_weights(str(path))
    return generator


def run_generator(generator, z, labels):
    import numpy as np
    import tensorflow as tf

    labels = np.asarray(labels, dtype="int32")
    return to_uint8(generator([tf.constant(z), tf.constant(labels)], training=False))


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_info(_args) -> None:
    print(f"Dự án     : {PROJECT_ROOT}")
    print(f"Model root: {MODEL_ROOT}  ({'có' if MODEL_ROOT.is_dir() else 'KHÔNG có'})\n")

    print("Checkpoint:")
    for stage in STAGES:
        info = best_info(stage)
        best = best_path(stage)
        best_txt = (f"best e{info['epoch']} (score {info['score']}, health {info['health']})"
                    if best.is_file() and info else ("best (không có best_info.json)" if best.is_file() else "không có best"))
        snaps = snapshot_records(stage)
        snap_txt = (f"{len(snaps)} snapshot {snaps[0]['kind']} (e{snaps[0]['epoch']}–e{snaps[-1]['epoch']})"
                    if snaps else "0 snapshot")
        print(f"  {stage}: {best_txt:<42} | {snap_txt}")

    seq = evolution_sequence()
    print(f"\nChuỗi 'evolution' sẽ dùng {len(seq)} khung: " + ", ".join(frame_label(r) for r in seq))

    print("\nDữ liệu:")
    processed = PROJECT_ROOT / "data" / "processed"
    for name in ("train_manifest.csv", "val_manifest.csv", "images_128", "images_64"):
        path = processed / name
        if path.is_dir():
            status = f"{sum(1 for _ in path.iterdir())} file"
        elif path.is_file():
            status = f"{sum(1 for _ in path.open(encoding='utf-8')) - 1} dòng"
        else:
            status = "chưa có (chạy notebook 01 → 03)"
        print(f"  data/processed/{name:<20} {status}")

    print("\nThư viện:")
    for module in ("tensorflow", "flask", "numpy", "PIL", "pandas", "matplotlib"):
        found = importlib.util.find_spec(module) is not None
        print(f"  {module:<11} {'OK' if found else 'thiếu'}")


def cmd_generate(args) -> None:
    ckpt = resolve_checkpoint(args.ckpt)
    species = ["dog", "cat"] if args.species == "both" else [args.species]
    z = latent_from_seed(args.seed, args.n)

    generator = load_generator(ckpt)
    images, labels = [], []
    for sp in species:
        imgs = run_generator(generator, z, [SPECIES_TO_LABEL[sp]] * args.n)
        images.extend(imgs)
        labels.extend(f"{sp} #{i}" for i in range(args.n))

    out = output_path(args.out, f"generate_{args.species}_seed{args.seed}")
    make_grid(images, args.cols, labels if args.labels else None, args.upscale).save(out)
    print(f"Checkpoint: {ckpt}")
    print(f"Đã lưu {len(images)} ảnh → {out}")

    if args.separate:
        from PIL import Image
        folder = out.with_suffix("")
        folder.mkdir(parents=True, exist_ok=True)
        for arr, label in zip(images, labels):
            Image.fromarray(arr).save(folder / f"{label.replace(' #', '_')}.png")
        print(f"Và từng ảnh riêng → {folder}")


def cmd_evolution(args) -> None:
    sequence = evolution_sequence()
    if not sequence:
        sys.exit("Không có checkpoint nào trong models/sobelv5/.")

    seeds = args.seeds
    species = args.species
    generator = build_generator()
    rows = {seed: [] for seed in seeds}
    for rec in sequence:
        generator.load_weights(str(rec["path"]))
        for seed in seeds:
            rows[seed].append(run_generator(generator, latent_from_seed(seed), [SPECIES_TO_LABEL[species]])[0])
        print(f"  ✓ {frame_label(rec):<14} {rec['path'].name}")

    images = [img for seed in seeds for img in rows[seed]]
    labels = [frame_label(r) for _ in seeds for r in sequence]
    out = output_path(args.out, f"evolution_{species}_seed{'-'.join(map(str, seeds))}")
    make_grid(images, len(sequence), labels, args.upscale).save(out)
    print(f"Đã lưu {len(sequence)} khung × {len(seeds)} seed → {out}")

    if args.gif:
        from PIL import Image
        gif_path = out.with_suffix(".gif")
        frames = [Image.fromarray(img).resize((IMAGE_SIZE * 2, IMAGE_SIZE * 2), Image.NEAREST)
                  for img in rows[seeds[0]]]
        frames[0].save(gif_path, save_all=True, append_images=frames[1:] + [frames[-1]] * 3,
                       duration=350, loop=0)
        print(f"GIF (seed {seeds[0]}) → {gif_path}")


def cmd_interpolate(args) -> None:
    import numpy as np

    if len(args.seeds) < 2:
        sys.exit("Cần ít nhất 2 seed, ví dụ: --seeds 1 2")
    ckpt = resolve_checkpoint(args.ckpt)
    anchors = [latent_from_seed(s)[0] for s in args.seeds]

    zs = []
    for a, b in zip(anchors[:-1], anchors[1:]):
        for t in np.linspace(0.0, 1.0, args.steps, endpoint=False):
            zs.append((1 - t) * a + t * b)
    zs.append(anchors[-1])
    z = np.stack(zs).astype("float32")

    generator = load_generator(ckpt)
    images = run_generator(generator, z, [SPECIES_TO_LABEL[args.species]] * len(z))
    out = output_path(args.out, f"interpolate_{args.species}_{'-'.join(map(str, args.seeds))}")
    make_grid(list(images), min(len(images), 16), None, args.upscale).save(out)
    print(f"Đã lưu {len(images)} ảnh nội suy → {out}")


def cmd_history(args) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    frames, offset, boundaries = [], 0, []
    for stage in STAGES:
        path = MODEL_ROOT / stage / "history.csv"
        if path.is_file():
            df = pd.read_csv(path)
            df["global_epoch"] = df["epoch"] + offset
            if offset:
                boundaries.append(offset)
            offset = int(df["global_epoch"].max())
            frames.append(df)
    if not frames:
        sys.exit("Không có history.csv trong models/sobelv5/stage*/")
    hist = pd.concat(frames, ignore_index=True)

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    axes[0].plot(hist["global_epoch"], hist["gen_loss"], label="Generator")
    axes[0].plot(hist["global_epoch"], hist["disc_loss"], label="Discriminator")
    axes[0].set_title("Loss")
    for sp in ("dog", "cat"):
        axes[1].plot(hist["global_epoch"], hist[f"struct_{sp}"], label=f"struct {sp}")
    axes[1].set_title("Độ đa dạng cấu trúc (struct)")
    axes[2].plot(hist["global_epoch"], hist["sharp_ratio"], label="sharp / real")
    axes[2].plot(hist["global_epoch"], hist["disc_acc"], label="D accuracy")
    axes[2].plot(hist["global_epoch"], hist["score"], label="score")
    axes[2].set_title("Độ sắc nét, D accuracy, score")
    for ax in axes:
        for b in boundaries:
            ax.axvline(b, color="grey", ls=":", lw=1)
        ax.set_xlabel("epoch (cả 3 stage)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    plt.tight_layout()

    out = output_path(args.out, "history")
    fig.savefig(out, dpi=130)
    print(f"Đã lưu biểu đồ → {out}")

    summary = hist.groupby("stage").agg(
        epochs=("epoch", "max"), disc_acc_last=("disc_acc", "last"),
        health_last=("health", "last"), sharp_last=("sharp_ratio", "last"))
    print(summary.round(3).to_string())


def cmd_serve(_args) -> None:
    print("Mở http://127.0.0.1:5000  (Ctrl+C để dừng)")
    runpy.run_path(str(PROJECT_ROOT / "src" / "web_server.py"), run_name="__main__")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/gan.py",
        description="Lệnh tiện dụng cho mô hình sobelv5 (sinh ảnh chó/mèo 128×128).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Checkpoint (--ckpt): best | stage1 | stage2 | stage3 | stage2:e150 | đường/dẫn/file.weights.h5",
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="<lệnh>")

    sub.add_parser("info", help="Liệt kê checkpoint, dữ liệu và thư viện đang có").set_defaults(func=cmd_info)

    p = sub.add_parser("generate", help="Sinh một lưới ảnh chó/mèo")
    p.add_argument("--species", choices=["dog", "cat", "both"], default="both")
    p.add_argument("-n", type=int, default=8, help="số ảnh mỗi loài (mặc định 8)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ckpt", default="best")
    p.add_argument("--cols", type=int, default=8)
    p.add_argument("--upscale", type=int, default=1, help="phóng to mỗi ô (nearest), vd 2")
    p.add_argument("--labels", action="store_true", help="ghi nhãn trên từng ô")
    p.add_argument("--separate", action="store_true", help="lưu thêm từng ảnh riêng lẻ")
    p.add_argument("--out", help="file PNG đầu ra (mặc định outputs/...)")
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("evolution", help="Cùng một z qua các checkpoint: xem mô hình 'lớn lên'")
    p.add_argument("--species", choices=["dog", "cat"], default="cat")
    p.add_argument("--seeds", type=int, nargs="+", default=[42], help="một hoặc nhiều seed (mỗi seed một hàng)")
    p.add_argument("--upscale", type=int, default=1)
    p.add_argument("--gif", action="store_true", help="xuất thêm GIF cho seed đầu tiên")
    p.add_argument("--out")
    p.set_defaults(func=cmd_evolution)

    p = sub.add_parser("interpolate", help="Nội suy tuyến tính giữa các vector nhiễu")
    p.add_argument("--species", choices=["dog", "cat"], default="dog")
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    p.add_argument("--steps", type=int, default=8, help="số bước giữa hai seed liên tiếp")
    p.add_argument("--ckpt", default="best")
    p.add_argument("--upscale", type=int, default=1)
    p.add_argument("--out")
    p.set_defaults(func=cmd_interpolate)

    p = sub.add_parser("history", help="Vẽ loss/metric từ history.csv của 3 stage")
    p.add_argument("--out")
    p.set_defaults(func=cmd_history)

    sub.add_parser("serve", help="Chạy web demo tại http://127.0.0.1:5000").set_defaults(func=cmd_serve)
    return parser


def main(argv=None) -> None:
    for stream in (sys.stdout, sys.stderr):  # Windows console: tránh lỗi in tiếng Việt
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
