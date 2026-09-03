from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from clinidoc.findings import STRUCTURE_BAD_ROW, STRUCTURE_UNREADABLE, Finding, finding

ANN_LINE = re.compile(
    r"^T(\d+)\t(\S+)\s+(\d+)\s+(\d+)(?:\s+\d+\s+\d+)*\t(.*)$"
)


def _split_from_path(path: Path, root: Path) -> str:
    from clinidoc.dataset import normalize_split

    parts = {p.lower() for p in path.relative_to(root).parts[:-1]} if path.is_relative_to(root) else set()
    for name in ("train", "val", "valid", "test"):
        if name in parts:
            return normalize_split(name)
    stem = path.stem.lower()
    for name in ("train", "val", "valid", "test"):
        if stem == name or stem.startswith(f"{name}_") or stem.endswith(f"_{name}"):
            return normalize_split(name)
    return "train"


def _parse_ann(path: Path, text: str) -> tuple[list[Any], list[Finding]]:
    from clinidoc.dataset import Entity

    entities: list[Any] = []
    findings: list[Finding] = []
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        findings.append(
            finding(STRUCTURE_UNREADABLE, f"Cannot read {path.name}: {exc}", evidence={"path": str(path)})
        )
        return entities, findings
    for lineno, line in enumerate(raw.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped[0] not in {"T", "A", "R", "N", "#", "*"}:
            continue
        if not stripped.startswith("T"):
            continue
        match = ANN_LINE.match(stripped)
        if not match:
            # Multi-span BRAT uses extra offsets; try a looser parse.
            parts = stripped.split("\t")
            if len(parts) >= 3 and parts[0].startswith("T"):
                meta = parts[1].split()
                if len(meta) >= 3:
                    try:
                        entities.append(
                            Entity(
                                start=int(meta[1]),
                                end=int(meta[2]),
                                label=meta[0],
                                text=parts[2] if len(parts) > 2 else None,
                            )
                        )
                        continue
                    except ValueError:
                        pass
            findings.append(
                finding(
                    STRUCTURE_BAD_ROW,
                    f"Unparseable BRAT entity at {path.name}:{lineno}",
                    evidence={"path": str(path), "row": lineno},
                )
            )
            continue
        _tid, label, start, end, span_text = match.groups()
        entities.append(Entity(start=int(start), end=int(end), label=label, text=span_text or None))
    return entities, findings


def _collect_pairs(root: Path) -> list[tuple[Path, Path]]:
    if root.is_file():
        if root.suffix.lower() == ".ann":
            txt = root.with_suffix(".txt")
            return [(txt, root)] if txt.exists() else []
        if root.suffix.lower() == ".txt":
            ann = root.with_suffix(".ann")
            return [(root, ann)] if ann.exists() else []
        return []
    pairs: list[tuple[Path, Path]] = []
    for ann in sorted(root.rglob("*.ann")):
        txt = ann.with_suffix(".txt")
        if txt.exists():
            pairs.append((txt, ann))
    return pairs


def load(
    root: Path,
    *,
    mapping: dict[str, str],
    config: dict[str, Any],
) -> tuple[list[Any], list[Path], list[Finding]]:
    from clinidoc.dataset import Document

    documents = []
    findings: list[Finding] = []
    files: list[Path] = []
    base = root if root.is_dir() else root.parent
    for txt_path, ann_path in _collect_pairs(root):
        files.extend([txt_path, ann_path])
        try:
            text = txt_path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            findings.append(
                finding(STRUCTURE_UNREADABLE, f"Cannot read {txt_path.name}: {exc}", evidence={"path": str(txt_path)})
            )
            continue
        entities, ann_findings = _parse_ann(ann_path, text)
        findings.extend(ann_findings)
        documents.append(
            Document(
                id=txt_path.stem,
                text=text,
                split=_split_from_path(txt_path, base),
                entities=entities or None,
                source_path=str(txt_path),
            )
        )
    return documents, files, findings
