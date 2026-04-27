from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

import cv2

from paper2.render.asset_registry import stable_hash

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
SPLITS = {"train", "val", "test"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--assets-root", type=str, default="D:/paper2_assets")
    p.add_argument("--out-csv", type=str, default="data/assets/stage2/asset_inventory.csv")
    p.add_argument("--train-ratio", type=float, default=0.7)
    p.add_argument("--val-ratio", type=float, default=0.15)
    return p.parse_args()


def _read_shape_and_mode(path: Path) -> tuple[int, int, str]:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    h, w = img.shape[:2]
    if img.ndim == 2:
        mode = "GRAY"
    elif img.shape[2] == 4:
        mode = "RGBA"
    else:
        mode = "RGB"
    return int(w), int(h), mode


def _file_hash(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _infer_split(path: Path) -> str | None:
    parts = [p.lower() for p in path.parts]
    for s in SPLITS:
        if s in parts:
            return s
    return None


def _hash_split(token: str, train_ratio: float, val_ratio: float) -> str:
    h = int(stable_hash(token), 16) % 1000
    t_cut = int(train_ratio * 1000)
    v_cut = int((train_ratio + val_ratio) * 1000)
    if h < t_cut:
        return "train"
    if h < v_cut:
        return "val"
    return "test"


def _infer_source(path: Path) -> str:
    low = path.name.lower()
    if low.startswith("s2_"):
        return "Copernicus"
    if "wiki" in low:
        return "Wikimedia"
    if "oi_" in low or "openimages" in str(path).lower():
        return "OpenImages"
    return "manual"


def _iter_backgrounds(root: Path):
    # Prefer explicitly split folders when available.
    patterns = [
        root / "backgrounds_split",
        root / "backgrounds" / "island_complex",
        root / "backgrounds" / "open_sea_processed",
        root / "backgrounds" / "open_sea",
        root / "backgrounds" / "coastal",
        root / "backgrounds" / "port",
    ]
    seen: set[Path] = set()
    for base in patterns:
        if not base.exists():
            continue
        for p in base.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                rp = p.resolve()
                if rp in seen:
                    continue
                seen.add(rp)
                yield p


def _iter_targets(root: Path):
    base = root / "target_templates" / "alpha_png"
    if not base.exists():
        return
    for p in base.rglob("*.png"):
        if p.is_file():
            yield p


def _iter_distractors(root: Path):
    preferred = root / "distractor_templates" / "splits"
    base = preferred if preferred.exists() else (root / "distractor_templates" / "alpha_png")
    if not base.exists():
        return
    for p in base.rglob("*.png"):
        if p.is_file():
            yield p


def _bg_category(path: Path) -> str:
    low = str(path).lower().replace("\\", "/")
    if "/open_sea" in low:
        return "open_sea"
    if "/coastal" in low:
        return "coastal"
    if "/island_complex" in low:
        return "island_complex"
    if "/port" in low:
        return "port"
    return "background"


def _target_category(path: Path) -> str:
    name = path.stem.lower()
    if "_top_" in name:
        return "boat_top"
    if "_side_" in name:
        return "boat_side"
    if "_oblique_" in name:
        return "boat_oblique"
    return "boat"


def _distractor_category(path: Path) -> str:
    name = path.stem.lower()
    if "ship_like" in name:
        return "ship_like"
    if "small" in name:
        return "small_watercraft"
    return "distractor"


def main() -> None:
    args = parse_args()
    assets_root = Path(args.assets_root).resolve()
    out_csv = Path(args.out_csv).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str | int]] = []
    seen_hash_by_type: dict[str, set[str]] = {"background": set(), "target": set(), "distractor": set()}

    for p in _iter_backgrounds(assets_root):
        digest = _file_hash(p)
        if digest in seen_hash_by_type["background"]:
            continue
        seen_hash_by_type["background"].add(digest)
        split = _infer_split(p) or _hash_split(f"background::{p.stem}", args.train_ratio, args.val_ratio)
        w, h, mode = _read_shape_and_mode(p)
        rows.append(
            {
                "asset_id": f"bg_{stable_hash(str(p.resolve()))}",
                "asset_type": "background",
                "category": _bg_category(p),
                "file_path": str(p.resolve()),
                "width": w,
                "height": h,
                "mode": mode,
                "split": split,
                "source": _infer_source(p),
                "status": "accept",
                "reason": "",
            }
        )

    for p in _iter_targets(assets_root):
        digest = _file_hash(p)
        if digest in seen_hash_by_type["target"]:
            continue
        seen_hash_by_type["target"].add(digest)
        split = _infer_split(p) or _hash_split(f"target::{p.stem}", args.train_ratio, args.val_ratio)
        w, h, mode = _read_shape_and_mode(p)
        rows.append(
            {
                "asset_id": f"tgt_{stable_hash(str(p.resolve()))}",
                "asset_type": "target",
                "category": _target_category(p),
                "file_path": str(p.resolve()),
                "width": w,
                "height": h,
                "mode": mode,
                "split": split,
                "source": _infer_source(p),
                "status": "accept",
                "reason": "",
            }
        )

    for p in _iter_distractors(assets_root):
        digest = _file_hash(p)
        if digest in seen_hash_by_type["distractor"]:
            continue
        seen_hash_by_type["distractor"].add(digest)
        split = _infer_split(p) or _hash_split(f"distractor::{p.stem}", args.train_ratio, args.val_ratio)
        w, h, mode = _read_shape_and_mode(p)
        rows.append(
            {
                "asset_id": f"dst_{stable_hash(str(p.resolve()))}",
                "asset_type": "distractor",
                "category": _distractor_category(p),
                "file_path": str(p.resolve()),
                "width": w,
                "height": h,
                "mode": mode,
                "split": split,
                "source": _infer_source(p),
                "status": "accept",
                "reason": "",
            }
        )

    fieldnames = [
        "asset_id",
        "asset_type",
        "category",
        "file_path",
        "width",
        "height",
        "mode",
        "split",
        "source",
        "status",
        "reason",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    by_type_split: dict[str, dict[str, int]] = {}
    for r in rows:
        at = str(r["asset_type"])
        sp = str(r["split"])
        by_type_split.setdefault(at, {}).setdefault(sp, 0)
        by_type_split[at][sp] += 1

    print(f"[DONE] wrote inventory: {out_csv}")
    print(f"[INFO] total assets: {len(rows)}")
    print(f"[INFO] counts by type/split: {by_type_split}")


if __name__ == "__main__":
    main()
