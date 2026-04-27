from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", type=str, default="data/rendered/stage2_smoke_v0")
    p.add_argument("--sequence-id", type=str, default="0000")
    p.add_argument("--frames", type=int, default=20)
    p.add_argument("--cols", type=int, default=5)
    return p.parse_args()


def make_sheet(paths: list[Path], out_path: Path, cols: int) -> None:
    imgs = [cv2.imread(str(p), cv2.IMREAD_COLOR) for p in paths if p.exists()]
    imgs = [x for x in imgs if x is not None]
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
    for split in ("train", "val", "test"):
        paths = [
            root / "images" / split / f"seq_{args.sequence_id}_frame_{i:04d}.png"
            for i in range(int(args.frames))
        ]
        out = root / "reports" / "contact_sheets" / f"{split}_seq{args.sequence_id}.jpg"
        make_sheet(paths, out, cols=int(args.cols))
        print(f"[DONE] {out}")


if __name__ == "__main__":
    main()
