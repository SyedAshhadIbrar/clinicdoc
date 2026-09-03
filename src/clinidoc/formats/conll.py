from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from clinidoc.findings import STRUCTURE_UNREADABLE, Finding, finding

META_RE = re.compile(r"(\w+)\s*=\s*([^\s]+)")
SPLIT_FILE_NAMES = {
    "train": ("train.conll", "train.bio", "train.tsv"),
    "val": ("val.conll", "valid.conll", "dev.conll", "val.bio", "valid.bio", "dev.bio", "val.tsv"),
    "test": ("test.conll", "test.bio", "test.tsv"),
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
    extras = sorted(
        p for p in root.iterdir() if p.is_file() and p.suffix.lower() in {".conll", ".bio", ".tsv"}
    )
    return [("train", p) for p in extras]


def _parse_meta(line: str) -> dict[str, str]:
    return {k: v for k, v in META_RE.findall(line)}


def entities_from_bio(tokens: list[tuple[str, str]]) -> tuple[str, list[Any], list[str]]:
    from clinidoc.dataset import Entity

    pieces: list[str] = []
    entities: list[Any] = []
    problems: list[str] = []
    offset = 0
    i = 0
    n = len(tokens)
    while i < n:
        token, tag = tokens[i]
        if pieces:
            pieces.append(" ")
            offset += 1
        start = offset
        pieces.append(token)
        offset += len(token)
        prefix, _, label = tag.partition("-")
        prefix = prefix.upper()
        if tag.upper() in {"O", "0"} or tag == "":
            prev = tokens[i - 1][1] if i else "O"
            i += 1
            continue
        if prefix == "I":
            prev_tag = tokens[i - 1][1] if i else "O"
            prev_p, _, prev_l = prev_tag.partition("-")
            if i == 0 or prev_tag.upper() in {"O", "0"} or prev_l != label:
                problems.append(f"I-{label} without matching B- at token {i}")
        if prefix in {"B", "I", "U", "S"} or (prefix == "I" and i == 0):
            label = label or tag
            j = i + 1
            end = offset
            while j < n:
                nprefix, _, nlabel = tokens[j][1].partition("-")
                if nprefix.upper() in {"I", "L", "E"} and nlabel == label:
                    pieces.append(" ")
                    end += 1
                    pieces.append(tokens[j][0])
                    end += len(tokens[j][0])
                    offset = end
                    j += 1
                    continue
                break
            entities.append(Entity(start=start, end=end, label=label, text="".join(pieces)[start:end]))
            i = j
            continue
        i += 1
    return "".join(pieces), entities, problems


def _flush_doc(
    tokens: list[tuple[str, str]],
    meta: dict[str, str],
    default_split: str,
    source_path: str,
    index: int,
) -> Any:
    from clinidoc.dataset import Document, normalize_split, parse_timestamp

    text, entities, problems = entities_from_bio(tokens)
    doc_id = meta.get("id") or meta.get("doc_id") or f"{Path(source_path).stem}-{index}"
    return Document(
        id=str(doc_id),
        text=text,
        split=normalize_split(meta.get("split"), default=default_split),
        patient_id=meta.get("patient_id") or meta.get("subject_id"),
        encounter_id=meta.get("encounter_id"),
        timestamp=parse_timestamp(meta.get("timestamp") or meta.get("date")),
        label=meta.get("label"),
        entities=entities or None,
        tokens=tokens or None,
        source_path=source_path,
    ), problems


def load(
    root: Path,
    *,
    mapping: dict[str, str],
    config: dict[str, Any],
) -> tuple[list[Any], list[Path], list[Finding]]:
    documents = []
    findings: list[Finding] = []
    files: list[Path] = []
    index = 0
    for default_split, path in _collect_files(root):
        files.append(path)
        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            findings.append(
                finding(STRUCTURE_UNREADABLE, f"Cannot read {path.name}: {exc}", evidence={"path": str(path)})
            )
            continue
        tokens: list[tuple[str, str]] = []
        meta: dict[str, str] = {}
        for line in raw.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                meta.update(_parse_meta(stripped.lstrip("#").strip()))
                continue
            if stripped.startswith("-DOCSTART-"):
                if tokens:
                    index += 1
                    doc, _problems = _flush_doc(tokens, meta, default_split, str(path), index)
                    documents.append(doc)
                    tokens = []
                    meta = _parse_meta(stripped)
                else:
                    meta.update(_parse_meta(stripped))
                continue
            if not stripped:
                if tokens:
                    index += 1
                    doc, _problems = _flush_doc(tokens, meta, default_split, str(path), index)
                    documents.append(doc)
                    tokens = []
                    meta = {}
                continue
            parts = re.split(r"\s+", stripped)
            if len(parts) == 1:
                token, tag = parts[0], "O"
            else:
                token, tag = parts[0], parts[-1]
            tokens.append((token, tag))
        if tokens:
            index += 1
            doc, _problems = _flush_doc(tokens, meta, default_split, str(path), index)
            documents.append(doc)
    return documents, files, findings
