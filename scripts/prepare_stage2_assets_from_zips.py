from __future__ import annotations

import argparse
import hashlib
import shutil
import zipfile
from pathlib import Path


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Normalize Stage2 assets from zip packages.")
    p.add_argument("--zip-dir", type=str, required=True, help="Directory containing source zip files.")
    p.add_argument(
        "--out-root",
        type=str,
        default="data/assets/source_stage2",
        help="Normalized output root directory.",
    )
    p.add_argument(
        "--tmp-raw-dir",
        type=str,
        default="data/assets/_tmp_raw_stage2",
        help="Temporary raw extraction directory.",
    )
    p.add_argument("--clean", action="store_true", help="Delete existing out-root/tmp-raw-dir before processing.")
    p.add_argument("--train-ratio", type=float, default=0.7)
    p.add_argument("--val-ratio", type=float, default=0.15)
    return p.parse_args()


def stable_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def infer_split(path: Path, token: str, train_ratio: float, val_ratio: float) -> str:
    low_parts = [p.lower() for p in path.parts]
    for s in SPLITS:
        if s in low_parts:
            return s
    h = int(stable_hash(token), 16) % 1000
    t_cut = int(train_ratio * 1000)
    v_cut = int((train_ratio + val_ratio) * 1000)
    if h < t_cut:
        return "train"
    if h < v_cut:
        return "val"
    return "test"


def safe_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        shutil.copy2(src, dst)


def infer_background_category(path: Path) -> str | None:
    low = str(path).lower().replace("\\", "/")
    name = path.name.lower()
    if "/open_sea" in low or name.startswith("s2_open_sea_"):
        return "open_sea_processed"
    if "/coastal" in low or name.startswith("s2_coastal_"):
        return "coastal"
    if "/island_complex" in low or name.startswith("s2_island_complex_"):
        return "island_complex"
    if "/port" in low or name.startswith("s2_port_"):
        return "port"
    return None


def ensure_layout(root: Path) -> None:
    for p in [
        root / "backgrounds" / "open_sea_processed",
        root / "backgrounds" / "coastal",
        root / "backgrounds" / "island_complex",
        root / "backgrounds" / "port",
        root / "target_templates" / "alpha_png" / "train",
        root / "target_templates" / "alpha_png" / "val",
        root / "target_templates" / "alpha_png" / "test",
        root / "distractor_templates" / "splits" / "train",
        root / "distractor_templates" / "splits" / "val",
        root / "distractor_templates" / "splits" / "test",
        root / "distractor_templates" / "alpha_png",
        root / "manifests",
        root / "qc",
    ]:
        p.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    zip_dir = Path(args.zip_dir).resolve()
    out_root = Path(args.out_root).resolve()
    tmp_raw = Path(args.tmp_raw_dir).resolve()

    if args.clean:
        if out_root.exists():
            shutil.rmtree(out_root)
        if tmp_raw.exists():
            shutil.rmtree(tmp_raw)

    out_root.mkdir(parents=True, exist_ok=True)
    tmp_raw.mkdir(parents=True, exist_ok=True)
    ensure_layout(out_root)

    zip_files = sorted(zip_dir.glob("*.zip"))
    if not zip_files:
        raise FileNotFoundError(f"No zip files found under {zip_dir}")

    for zf in zip_files:
        with zipfile.ZipFile(zf, "r") as z:
            z.extractall(tmp_raw)

    num_bg = 0
    num_target = 0
    num_dst_split = 0
    num_dst_alpha = 0
    num_manifest = 0
    num_qc = 0

    for p in tmp_raw.rglob("*"):
        if not p.is_file():
            continue
        suffix = p.suffix.lower()
        name = p.name.lower()
        low = str(p).lower().replace("\\", "/")
        norm_name = low.rsplit("/", 1)[-1]

        if "/manifests/" in low and suffix in {".csv", ".jsonl", ".txt"}:
            safe_copy(p, out_root / "manifests" / p.name)
            num_manifest += 1
            continue
        if "/qc/" in low and suffix in {".jpg", ".jpeg", ".png"}:
            safe_copy(p, out_root / "qc" / p.name)
            num_qc += 1
            continue
        if name == "open_sea_processed_manifest.csv":
            safe_copy(p, out_root / "manifests" / p.name)
            num_manifest += 1
            continue

        if suffix == ".png" and (
            name.startswith("target_")
            or norm_name.startswith("target_")
            or "/target_" in low
        ):
            split = infer_split(p, f"target::{p.stem}", args.train_ratio, args.val_ratio)
            out_name = norm_name if norm_name.endswith(".png") else p.name
            safe_copy(p, out_root / "target_templates" / "alpha_png" / split / out_name)
            num_target += 1
            continue

        if suffix == ".png" and (
            name.startswith("distractor_")
            or norm_name.startswith("distractor_")
            or "/distractor_" in low
        ):
            split = infer_split(p, f"distractor::{p.stem}", args.train_ratio, args.val_ratio)
            out_name = norm_name if norm_name.endswith(".png") else p.name
            safe_copy(p, out_root / "distractor_templates" / "splits" / split / out_name)
            safe_copy(p, out_root / "distractor_templates" / "alpha_png" / out_name)
            num_dst_split += 1
            num_dst_alpha += 1
            continue

        if suffix in IMAGE_EXTS:
            cat = infer_background_category(p)
            if cat is not None:
                safe_copy(p, out_root / "backgrounds" / cat / p.name)
                num_bg += 1

    print("[DONE] Stage2 assets prepared")
    print(f"[INFO] zip_dir={zip_dir}")
    print(f"[INFO] out_root={out_root}")
    print(f"[INFO] tmp_raw={tmp_raw}")
    print(
        "[INFO] counts "
        f"background={num_bg} target={num_target} "
        f"distractor_split={num_dst_split} distractor_alpha={num_dst_alpha} "
        f"manifest={num_manifest} qc={num_qc}"
    )


if __name__ == "__main__":
    main()
