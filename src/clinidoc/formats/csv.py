from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from clinidoc.findings import STRUCTURE_BAD_ROW, STRUCTURE_UNREADABLE, Finding, finding

SPLIT_FILE_NAMES = {
    "train": ("train.csv",),
    "val": ("val.csv", "valid.csv", "dev.csv"),
    "test": ("test.csv",),
}


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
    extras = sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() == ".csv")
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
    delimiter = str(config.get("delimiter") or ",")
    for default_split, path in _collect_files(root):
        files.append(path)
        try:
            handle = path.open("r", encoding="utf-8-sig", errors="replace", newline="")
        except OSError as exc:
            findings.append(
                finding(STRUCTURE_UNREADABLE, f"Cannot read {path.name}: {exc}", evidence={"path": str(path)})
            )
            continue
        with handle:
            try:
                reader = csv.DictReader(handle, delimiter=delimiter)
                if reader.fieldnames is None:
                    findings.append(
                        finding(
                            STRUCTURE_BAD_ROW,
                            f"CSV has no header: {path.name}",
                            evidence={"path": str(path)},
                        )
                    )
                    continue
                for lineno, row in enumerate(reader, start=2):
                    if row is None:
                        continue
                    if not any((value or "").strip() for value in row.values()):
                        continue
                    index += 1
                    documents.append(
                        record_to_document(
                            {k: v for k, v in row.items() if k is not None},
                            default_split=default_split,
                            source_path=str(path),
                            source_row=lineno,
                            index=index,
                            mapping=mapping,
                        )
                    )
            except csv.Error as exc:
                findings.append(
                    finding(
                        STRUCTURE_BAD_ROW,
                        f"CSV parse error in {path.name}: {exc}",
                        evidence={"path": str(path)},
                    )
                )
    return documents, files, findings
