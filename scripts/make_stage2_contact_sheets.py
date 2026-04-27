from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", type=str, default="data/rendered/stage2_smoke_v0")
    p.add_argument("--sequence-id", type=str, default="0000")
    p.add_argument("--frames", type=int, default=20)
    p.add_argument("--cols", type=int, default=5)
    p.add_argument("--draw-labels", action="store_true", help="Draw bbox/center/target_id overlay")
    return p.parse_args()


def _load_labels(dataset_root: Path) -> dict[tuple[str, str], dict]:
    out: dict[tuple[str, str], dict] = {}
    for split in ("train", "val", "test"):
        lp = dataset_root / "labels" / f"{split}.jsonl"
        if not lp.exists():
            continue
        with lp.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                img_name = Path(str(row.get("image_path", ""))).name
                out[(split, img_name)] = row
    return out


def _draw_overlay(img: np.ndarray, row: dict | None) -> np.ndarray:
    if row is None:
        return img
    out = img.copy()
    try:
        x, y, w, h = [int(round(float(v))) for v in row.get("bbox_xywh", [0, 0, 0, 0])]
        cx, cy = [int(round(float(v))) for v in row.get("target_center_px", [0, 0])]
        cv2.rectangle(out, (x, y), (x + max(1, w), y + max(1, h)), (0, 255, 255), 1)
        cv2.drawMarker(out, (cx, cy), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=8, thickness=1)
        tid = str(row.get("target_asset_id", "NA"))[:16]
        cv2.putText(out, f"tgt:{tid}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1, cv2.LINE_AA)
    except Exception:
        return img
    return out


def make_sheet(
    paths: list[Path],
    out_path: Path,
    cols: int,
    split: str,
    labels: dict[tuple[str, str], dict] | None = None,
    draw_labels: bool = False,
) -> None:
    imgs = []
    for p in paths:
        if not p.exists():
            continue
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            continue
        if draw_labels and labels is not None:
            img = _draw_overlay(img, labels.get((split, p.name)))
        imgs.append(img)
    if not imgs:
        return
    h, w = imgs[0].shape[:2]
    rows = int(np.ceil(len(imgs) / max(1, cols)))
    sheet = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for i, img in enumerate(imgs):
        r = i // cols
        c = i % cols
        sheet[r * h : (r + 1) * h, c * w : (c + 1) * w] = img
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), sheet)


def main() -> None:
    args = parse_args()
    root = Path(args.dataset_root).resolve()
    labels = _load_labels(root) if args.draw_labels else None
    for split in ("train", "val", "test"):
        paths = [
            root / "images" / split / f"seq_{args.sequence_id}_frame_{i:04d}.png"
            for i in range(int(args.frames))
        ]
        suffix = "_overlay" if args.draw_labels else ""
        out = root / "reports" / "contact_sheets" / f"{split}_seq{args.sequence_id}{suffix}.jpg"
        make_sheet(paths, out, cols=int(args.cols), split=split, labels=labels, draw_labels=args.draw_labels)
        print(f"[DONE] {out}")


if __name__ == "__main__":
    main()
