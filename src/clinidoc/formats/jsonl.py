from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from clinidoc.findings import STRUCTURE_BAD_ROW, STRUCTURE_UNREADABLE, Finding, finding

SPLIT_FILE_NAMES = {
    "train": ("train.jsonl", "train.json"),
    "val": ("val.jsonl", "valid.jsonl", "val.json", "valid.json", "dev.jsonl"),
    "test": ("test.jsonl", "test.json"),
}


def _iter_json_records(path: Path) -> tuple[list[tuple[int, dict[str, Any]]], list[Finding]]:
    findings: list[Finding] = []
    records: list[tuple[int, dict[str, Any]]] = []
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        findings.append(
            finding(STRUCTURE_UNREADABLE, f"Cannot read {path.name}: {exc}", evidence={"path": str(path)})
        )
        return records, findings
    if path.suffix.lower() == ".json" and raw.lstrip().startswith("["):
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            findings.append(
                finding(
                    STRUCTURE_BAD_ROW,
                    f"Invalid JSON array in {path.name}: {exc}",
                    evidence={"path": str(path)},
                )
            )
            return records, findings
        if isinstance(payload, list):
            for i, item in enumerate(payload, start=1):
                if isinstance(item, dict):
                    records.append((i, item))
                else:
                    findings.append(
                        finding(
                            STRUCTURE_BAD_ROW,
                            f"Non-object JSON element at index {i} in {path.name}",
                            evidence={"path": str(path), "row": i},
                        )
                    )
            return records, findings
    for lineno, line in enumerate(raw.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            item = json.loads(stripped)
        except json.JSONDecodeError as exc:
            findings.append(
                finding(
                    STRUCTURE_BAD_ROW,
                    f"Invalid JSONL at {path.name}:{lineno}: {exc}",
                    evidence={"path": str(path), "row": lineno},
                )
            )
            continue
        if not isinstance(item, dict):
            findings.append(
                finding(
                    STRUCTURE_BAD_ROW,
                    f"Non-object JSONL row at {path.name}:{lineno}",
                    evidence={"path": str(path), "row": lineno},
                )
            )
            continue
        records.append((lineno, item))
    return records, findings


def _collect_files(root: Path) -> list[tuple[str, Path]]:
    if root.is_file():
        return [("train", root)]
    found: list[tuple[str, Path]] = []
    for split, names in SPLIT_FILE_NAMES.items():
        for name in names:
            path = root / name
            if path.is_file():
                found.append((split, path))
                break
    if found:
        return found
    extras = sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in {".jsonl", ".json"})
    return [("train", p) for p in extras]


def load(
    root: Path,
    *,
    mapping: dict[str, str],
    config: dict[str, Any],
) -> tuple[list[Any], list[Path], list[Finding]]:
    from clinidoc.dataset import record_to_document

    documents = []
    findings: list[Finding] = []
    files: list[Path] = []
    index = 0
    for default_split, path in _collect_files(root):
        files.append(path)
        rows, row_findings = _iter_json_records(path)
        findings.extend(row_findings)
        for lineno, record in rows:
            index += 1
            documents.append(
                record_to_document(
                    record,
                    default_split=default_split,
                    source_path=str(path),
                    source_row=lineno,
                    index=index,
                    mapping=mapping,
                )
            )
    return documents, files, findings
