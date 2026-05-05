"""Utilities for label ID -> name lookups.

`Dataset/label_definitions.txt` embeds the authoritative label lists.
We extract them into `Dataset/label_mappings.json` via `pipeline/extract_label_mappings.py`.

This module loads that JSON and provides simple lookup helpers for the pipeline.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LabelInfo:
    label_id: int
    name: str
    group: str  # "cortex" | "noncortex"


def load_label_mappings(repo_root: Path) -> dict[int, LabelInfo]:
    """Load `Dataset/label_mappings.json` and return a mapping by label id.

    If an ID appears multiple times (unlikely), we prefer `noncortex` over `cortex`
    because those are typically the structure IDs used for subcortical meshes.
    """

    path = repo_root / "Dataset" / "label_mappings.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Run pipeline/extract_label_mappings.py first."
        )

    payload = json.loads(path.read_text(encoding="utf-8"))
    labels = payload.get("labels", [])
    out: dict[int, LabelInfo] = {}
    for item in labels:
        try:
            lid = int(item["id"])
            name = str(item["name"])
            group = str(item.get("group", "unknown"))
        except Exception:
            continue

        info = LabelInfo(label_id=lid, name=name, group=group)
        prev = out.get(lid)
        if prev is None:
            out[lid] = info
        else:
            # Prefer noncortex names when available.
            if prev.group != "noncortex" and group == "noncortex":
                out[lid] = info
    return out


def label_name_for_id(mappings: dict[int, LabelInfo], label_id: int) -> str | None:
    info = mappings.get(int(label_id))
    return info.name if info else None

