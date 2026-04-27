from __future__ import annotations

import argparse
import math
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
        root / "backgrounds" / "open_sea_processed" / "train",
        root / "backgrounds" / "open_sea_processed" / "val",
        root / "backgrounds" / "open_sea_processed" / "test",
        root / "backgrounds" / "coastal" / "train",
        root / "backgrounds" / "coastal" / "val",
        root / "backgrounds" / "coastal" / "test",
        root / "backgrounds" / "island_complex" / "train",
        root / "backgrounds" / "island_complex" / "val",
        root / "backgrounds" / "island_complex" / "test",
        root / "backgrounds" / "port" / "train",
        root / "backgrounds" / "port" / "val",
        root / "backgrounds" / "port" / "test",
        root / "target_templates" / "alpha_png" / "train",
        root / "target_templates" / "alpha_png" / "val",
        root / "target_templates" / "alpha_png" / "test",
        root / "distractor_templates" / "splits" / "train",
        root / "distractor_templates" / "splits" / "val",
        root / "distractor_templates" / "splits" / "test",
        root / "distractor_templates" / "alpha_png" / "train",
        root / "distractor_templates" / "alpha_png" / "val",
        root / "distractor_templates" / "alpha_png" / "test",
        root / "manifests",
        root / "qc",
    ]:
        p.mkdir(parents=True, exist_ok=True)


def _desired_counts(n: int, train_ratio: float, val_ratio: float) -> tuple[int, int, int]:
    test_ratio = max(0.0, 1.0 - train_ratio - val_ratio)
    raw = [n * train_ratio, n * val_ratio, n * test_ratio]
    base = [int(math.floor(x)) for x in raw]
    rem = n - sum(base)
    frac_idx = sorted([(raw[i] - base[i], i) for i in range(3)], reverse=True)
    for k in range(rem):
        base[frac_idx[k % 3][1]] += 1
    n_train, n_val, n_test = base
    if n >= 3:
        if n_train == 0:
            n_train, n_test = 1, max(0, n_test - 1)
        if n_val == 0:
            n_val, n_train = 1, max(1, n_train - 1)
        if n_test == 0:
            n_test, n_train = 1, max(1, n_train - 1)
        while n_train + n_val + n_test < n:
            n_train += 1
        while n_train + n_val + n_test > n and n_train > 1:
            n_train -= 1
    else:
        n_train = min(1, n)
        n_val = 1 if n >= 2 else 0
        n_test = n - n_train - n_val
    return n_train, n_val, n_test


def _rebalance_split_pngs(root: Path, train_ratio: float, val_ratio: float) -> tuple[int, int, int, int]:
    split_dirs = [root / "train", root / "val", root / "test"]
    for d in split_dirs:
        d.mkdir(parents=True, exist_ok=True)
    files = []
    for d in split_dirs:
        files.extend([p for p in d.glob("*.png") if p.is_file()])
    files = sorted(files, key=lambda p: p.name.lower())
    n = len(files)
    n_train, n_val, n_test = _desired_counts(n, train_ratio, val_ratio)

    tmp = root / ".rebalance_tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    staged: list[Path] = []
    for i, p in enumerate(files):
        t = tmp / f"{i:06d}__{p.name}"
        shutil.move(str(p), str(t))
        staged.append(t)

    cursor = 0
    for split, count in (("train", n_train), ("val", n_val), ("test", n_test)):
        out_dir = root / split
        for _ in range(count):
            src = staged[cursor]
            cursor += 1
            out = out_dir / src.name.split("__", 1)[1]
            shutil.move(str(src), str(out))

    shutil.rmtree(tmp)
    return n, n_train, n_val, n_test


def _sync_distractor_alpha_from_splits(splits_root: Path, alpha_root: Path) -> None:
    for sp in SPLITS:
        d = alpha_root / sp
        if d.exists():
            for p in d.glob("*.png"):
                p.unlink()
        d.mkdir(parents=True, exist_ok=True)
        for src in sorted((splits_root / sp).glob("*.png"), key=lambda p: p.name.lower()):
            safe_copy(src, d / src.name)


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
            safe_copy(p, out_root / "distractor_templates" / "alpha_png" / split / out_name)
            num_dst_split += 1
            num_dst_alpha += 1
            continue

        if suffix in IMAGE_EXTS:
            cat = infer_background_category(p)
            if cat is not None:
                split = infer_split(p, f"background::{cat}::{norm_name}", args.train_ratio, args.val_ratio)
                safe_copy(p, out_root / "backgrounds" / cat / split / norm_name)
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

    # Enforce deterministic 70/15/15 split for template assets.
    t_total, t_train, t_val, t_test = _rebalance_split_pngs(
        out_root / "target_templates" / "alpha_png", args.train_ratio, args.val_ratio
    )
    d_total, d_train, d_val, d_test = _rebalance_split_pngs(
        out_root / "distractor_templates" / "splits", args.train_ratio, args.val_ratio
    )
    _sync_distractor_alpha_from_splits(
        out_root / "distractor_templates" / "splits",
        out_root / "distractor_templates" / "alpha_png",
    )
    print(
        "[INFO] rebalanced target "
        f"total={t_total} train={t_train} val={t_val} test={t_test}"
    )
    print(
        "[INFO] rebalanced distractor "
        f"total={d_total} train={d_train} val={d_val} test={d_test}"
    )


if __name__ == "__main__":
    main()
