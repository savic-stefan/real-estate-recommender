from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable


def ensure_directories(paths: Iterable[Path]) -> None:
    """Create a sequence of directories if they do not already exist."""

    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def discover_csv_files(search_roots: Iterable[Path], patterns: Iterable[str]) -> list[Path]:
    """Find CSV datasets across multiple roots while preserving uniqueness."""

    discovered: dict[str, Path] = {}
    for root in search_roots:
        if not root.exists():
            continue
        for pattern in patterns:
            for candidate in root.glob(pattern):
                if candidate.is_file():
                    discovered[str(candidate.resolve())] = candidate.resolve()
    return sorted(discovered.values())


def serialize_to_json(data: Any, output_path: Path) -> None:
    """Persist metadata objects in UTF-8 JSON format."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(data) if is_dataclass(data) else data
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def project_tree_snapshot(root: Path, max_depth: int = 2) -> list[str]:
    """Return a compact textual snapshot of the current project structure."""

    lines: list[str] = []
    root = root.resolve()
    for path in sorted(root.rglob("*")):
        depth = len(path.relative_to(root).parts)
        if depth > max_depth:
            continue
        prefix = "  " * (depth - 1)
        label = f"{prefix}{path.name}/" if path.is_dir() else f"{prefix}{path.name}"
        lines.append(label)
    return lines
