from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class AssetRecord:
    asset_id: str
    asset_type: str
    category: str
    file_path: str
    width: int
    height: int
    mode: str
    split: str
    source: str
    status: str
    reason: str

    @property
    def path(self) -> Path:
        return Path(self.file_path)


def stable_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def load_asset_inventory(path: str | Path) -> list[AssetRecord]:
    rows: list[AssetRecord] = []
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                AssetRecord(
                    asset_id=str(row["asset_id"]),
                    asset_type=str(row["asset_type"]),
                    category=str(row["category"]),
                    file_path=str(row["file_path"]),
                    width=int(row["width"]),
                    height=int(row["height"]),
                    mode=str(row["mode"]),
                    split=str(row["split"]),
                    source=str(row.get("source", "")),
                    status=str(row.get("status", "")),
                    reason=str(row.get("reason", "")),
                )
            )
    return rows


class AssetRegistry:
    def __init__(self, assets: Iterable[AssetRecord]):
        self.assets = [a for a in assets if a.status.lower() in {"accept", "accepted", ""}]
        self._by_type_split: dict[tuple[str, str], list[AssetRecord]] = {}
        for a in self.assets:
            self._by_type_split.setdefault((a.asset_type, a.split), []).append(a)

    def get(self, asset_type: str, split: str) -> list[AssetRecord]:
        return list(self._by_type_split.get((asset_type, split), []))

    def sample_one(self, asset_type: str, split: str, rng) -> AssetRecord:
        pool = self.get(asset_type, split)
        if not pool:
            raise ValueError(f"No assets for type={asset_type}, split={split}")
        idx = int(rng.integers(0, len(pool)))
        return pool[idx]

    def sample_many(self, asset_type: str, split: str, n: int, rng) -> list[AssetRecord]:
        pool = self.get(asset_type, split)
        if n <= 0:
            return []
        if not pool:
            raise ValueError(f"No assets for type={asset_type}, split={split}")
        if n >= len(pool):
            idx = rng.choice(len(pool), size=n, replace=True)
        else:
            idx = rng.choice(len(pool), size=n, replace=False)
        return [pool[int(i)] for i in idx]
