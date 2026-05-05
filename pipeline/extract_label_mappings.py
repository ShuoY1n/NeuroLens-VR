"""Extract label ID -> name mappings from Dataset/label_definitions.txt.

The source file includes embedded Python-style lists:
- cortex_numbers_names = [[1002, "left ..."], ...]
- noncortex_numbers_names = [[16, "Brain stem"], ...]

We parse these lists (without executing them) and emit a single JSON file:
`Dataset/label_mappings.json`.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LabelMapping:
    label_id: int
    name: str
    group: str  # "cortex" | "noncortex"


# Match non-commented list entries like: [1002, "left ..."]
# Reject lines that start with optional whitespace then '#'.
_ENTRY_RE = re.compile(
    r"(?m)^(?!\s*#).*?\[\s*(?P<id>\d+)\s*,\s*\"(?P<name>(?:[^\"\\]|\\.)*)\"\s*\]"
)


def _unescape_python_string(s: str) -> str:
    # The source uses double-quoted python strings. We only need a minimal unescape.
    return bytes(s, "utf-8").decode("unicode_escape")


def extract_mappings(label_definitions_text: str) -> list[LabelMapping]:
    mappings: list[LabelMapping] = []

    def extract_between(mark_start: str, mark_end: str, group: str) -> None:
        start = label_definitions_text.find(mark_start)
        if start < 0:
            return
        end = label_definitions_text.find(mark_end, start + len(mark_start))
        if end < 0:
            end = len(label_definitions_text)
        body = label_definitions_text[start:end]
        for em in _ENTRY_RE.finditer(body):
            lid = int(em.group("id"))
            name_raw = em.group("name")
            name = _unescape_python_string(name_raw).strip()
            mappings.append(LabelMapping(label_id=lid, name=name, group=group))

    extract_between(
        "cortex_numbers_names = [",
        "noncortex_numbers_names = [",
        "cortex",
    )
    extract_between(
        "noncortex_numbers_names = [",
        "sulcus_names = [",
        "noncortex",
    )

    # De-dupe by (id, group, name) while preserving order.
    seen: set[tuple[int, str, str]] = set()
    unique: list[LabelMapping] = []
    for x in mappings:
        key = (x.label_id, x.group, x.name)
        if key in seen:
            continue
        seen.add(key)
        unique.append(x)
    return unique


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    src_path = repo_root / "Dataset" / "label_definitions.txt"
    if not src_path.is_file():
        raise SystemExit(f"Missing {src_path}")

    text = src_path.read_text(encoding="utf-8", errors="replace")
    mappings = extract_mappings(text)
    if not mappings:
        raise SystemExit("No mappings extracted. Source format may have changed.")

    out_path = repo_root / "Dataset" / "label_mappings.json"
    payload = {
        "source": str(src_path).replace("\\", "/"),
        "extractedFrom": "cortex_numbers_names + noncortex_numbers_names",
        "count": len(mappings),
        "labels": [
            {"id": m.label_id, "name": m.name, "group": m.group}
            for m in sorted(mappings, key=lambda x: (x.group, x.label_id))
        ],
    }
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path} ({len(mappings)} labels).")


if __name__ == "__main__":
    main()

