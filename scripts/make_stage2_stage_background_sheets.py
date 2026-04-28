from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", type=str, required=True)
    p.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    p.add_argument(
        "--background-categories",
        type=str,
        default="open_sea,coastal,island_complex,port",
        help="Comma-separated categories to export.",
    )
    p.add_argument("--frames", type=int, default=20, help="Frames per sheet")
    p.add_argument("--cols", type=int, default=5)
    p.add_argument("--draw-labels", action="store_true")
    p.add_argument("--inventory-csv", type=str, default="data/assets/stage2/asset_inventory.csv")
    return p.parse_args()


def _load_asset_category_map(inventory_csv: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not inventory_csv.exists():
        return out
    with inventory_csv.open("r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            out[str(r.get("asset_id", ""))] = str(r.get("category", "unknown"))
    return out


def _draw_overlay(img: np.ndarray, row: dict) -> np.ndarray:
    out = img.copy()
    try:
        x, y, w, h = [int(round(float(v))) for v in row.get("bbox_xywh", [0, 0, 0, 0])]
        cx, cy = [int(round(float(v))) for v in row.get("target_center_px", [0, 0])]
        cv2.rectangle(out, (x, y), (x + max(1, w), y + max(1, h)), (0, 255, 255), 1)
        cv2.drawMarker(out, (cx, cy), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=8, thickness=1)
    except Exception:
        pass
    return out


def _make_sheet(imgs: list[np.ndarray], out_path: Path, cols: int) -> None:
    if not imgs:
        return
    h, w = imgs[0].shape[:2]
    cols = max(1, int(cols))
    rows = int(np.ceil(len(imgs) / cols))
    canvas = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for i, im in enumerate(imgs):
        r = i // cols
        c = i % cols
        canvas[r * h : (r + 1) * h, c * w : (c + 1) * w] = im
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)


def main() -> None:
    args = parse_args()
    root = Path(args.dataset_root).resolve()
    split = str(args.split)
    frames = max(1, int(args.frames))
    cols = max(1, int(args.cols))
    wanted_bg = [x.strip().lower() for x in str(args.background_categories).split(",") if x.strip()]
    asset2cat = _load_asset_category_map(Path(args.inventory_csv).resolve())

    labels_path = root / "labels" / f"{split}.jsonl"
    if not labels_path.exists():
        raise FileNotFoundError(f"Missing labels: {labels_path}")

    # stage -> bg -> sequence_id -> rows
    grouped: dict[str, dict[str, dict[str, list[dict]]]] = {
        s: {b: defaultdict(list) for b in wanted_bg} for s in ("far", "mid", "terminal")
    }
    with labels_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            st = str(r.get("stage", "")).lower()
            if st not in grouped:
                continue
            bg = str(r.get("meta", {}).get("background_category", "")).lower()
            if not bg:
                bg = str(asset2cat.get(str(r.get("background_asset_id", "")), "unknown")).lower()
            if bg not in grouped[st]:
                continue
            grouped[st][bg][str(r.get("sequence_id", ""))].append(r)

    out_dir = root / "reports" / "contact_sheets" / f"{split}_stage_background"
    for st in ("far", "mid", "terminal"):
        for bg in wanted_bg:
            seq_map = grouped[st][bg]
            if not seq_map:
                print(f"[SKIP] no samples for split={split}, stage={st}, bg={bg}")
                continue
            # pick a stable sequence with most frames in this stage
            seq_id = sorted(seq_map.keys(), key=lambda k: len(seq_map[k]), reverse=True)[0]
            rows = sorted(seq_map[seq_id], key=lambda x: int(x.get("frame_id", 0)))
            rows = rows[:frames]
            imgs: list[np.ndarray] = []
            for r in rows:
                p = Path(str(r.get("image_path", "")))
                img_path = p if p.is_absolute() else (Path.cwd() / p)
                if not img_path.exists():
                    img_path = root / "images" / split / f"seq_{seq_id.split('_')[-1]}_frame_{int(r.get('frame_id',0)):04d}.png"
                img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
                if img is None:
                    continue
                if args.draw_labels:
                    img = _draw_overlay(img, r)
                cv2.putText(
                    img,
                    f"{split} {st} {bg} {seq_id}",
                    (6, 16),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
                imgs.append(img)
            out_path = out_dir / f"{split}_{st}_{bg}.jpg"
            _make_sheet(imgs, out_path, cols=cols)
            print(f"[DONE] {out_path}")


if __name__ == "__main__":
    main()
