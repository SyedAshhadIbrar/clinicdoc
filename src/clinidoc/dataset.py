from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from clinidoc.findings import (
    STRUCTURE_MISSING_SPLIT,
    STRUCTURE_NO_DOCUMENTS,
    STRUCTURE_UNREADABLE,
    Finding,
    finding,
)
from clinidoc.formats import brat, conll, csv as csv_format, jsonl as jsonl_format

SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "val": "val",
    "valid": "val",
    "validation": "val",
    "dev": "val",
    "development": "val",
    "test": "test",
    "testing": "test",
    "holdout": "test",
}

TEXT_KEYS = ("text", "note", "document", "clinical_text", "content", "note_text")
ID_KEYS = ("id", "doc_id", "document_id", "note_id")
LABEL_KEYS = ("label", "class", "category", "gold_label")
LABELS_KEYS = ("labels", "classes")
PATIENT_KEYS = ("patient_id", "patientId", "subject_id", "hadm_subject", "pt_id")
ENCOUNTER_KEYS = ("encounter_id", "hadm_id", "visit_id", "admission_id")
TIMESTAMP_KEYS = ("timestamp", "time", "datetime", "charttime", "date")
ENTITY_KEYS = ("entities", "ner", "spans", "annotations")
SPLIT_KEYS = ("split", "partition", "subset")
TOKENS_KEYS = ("tokens", "bio", "tags")


@dataclass
class Entity:
    start: int
    end: int
    label: str
    text: str | None = None


@dataclass
class Document:
    id: str
    text: str
    split: str
    patient_id: str | None = None
    encounter_id: str | None = None
    timestamp: datetime | None = None
    label: str | None = None
    labels: list[str] | None = None
    entities: list[Entity] | None = None
    tokens: list[tuple[str, str]] | None = None
    source_path: str | None = None
    source_row: int | None = None

    def class_labels(self) -> list[str]:
        if self.labels:
            return list(self.labels)
        if self.label is not None and str(self.label) != "":
            return [str(self.label)]
        return []

    def has_classification(self) -> bool:
        return bool(self.class_labels())

    def has_ner(self) -> bool:
        return bool(self.entities) or bool(self.tokens)


@dataclass
class LoadedDataset:
    documents: list[Document]
    root: Path
    detected_format: str
    files: list[Path]
    load_findings: list[Finding] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)
    expected_splits: list[str] = field(default_factory=list)

    def by_split(self) -> dict[str, list[Document]]:
        out: dict[str, list[Document]] = {}
        for doc in self.documents:
            out.setdefault(doc.split, []).append(doc)
        return out


def normalize_split(value: Any, default: str = "train") -> str:
    if value is None or str(value).strip() == "":
        return default
    key = str(value).strip().lower()
    return SPLIT_ALIASES.get(key, key)


def parse_timestamp(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        except (OSError, OverflowError, ValueError):
            return None
    text = str(value).strip()
    if not text:
        return None
    iso = text.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(iso)
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def pick(record: dict[str, Any], keys: tuple[str, ...], mapping: dict[str, str] | None = None) -> Any:
    if mapping:
        for canonical in keys:
            mapped = mapping.get(canonical)
            if mapped and mapped in record:
                return record[mapped]
    for key in keys:
        if key in record:
            return record[key]
    lower = {str(k).lower(): v for k, v in record.items()}
    for key in keys:
        if key.lower() in lower:
            return lower[key.lower()]
    return None


def parse_entities(raw: Any) -> list[Entity] | None:
    if raw is None or raw == "":
        return None
    if isinstance(raw, str):
        import json

        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return None
    if not isinstance(raw, list):
        return None
    entities: list[Entity] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        start = item.get("start", item.get("begin", item.get("start_offset")))
        end = item.get("end", item.get("finish", item.get("end_offset")))
        label = item.get("label", item.get("type", item.get("entity")))
        text = item.get("text", item.get("span", item.get("value")))
        span = item.get("span") or item.get("offsets")
        if start is None and isinstance(span, (list, tuple)) and len(span) >= 2:
            start, end = span[0], span[1]
        if start is None or end is None or label is None:
            continue
        try:
            entities.append(
                Entity(start=int(start), end=int(end), label=str(label), text=None if text is None else str(text))
            )
        except (TypeError, ValueError):
            continue
    return entities or None


def record_to_document(
    record: dict[str, Any],
    *,
    default_split: str,
    source_path: str,
    source_row: int,
    index: int,
    mapping: dict[str, str] | None = None,
) -> Document:
    doc_id = pick(record, ID_KEYS, mapping)
    if doc_id is None or str(doc_id).strip() == "":
        doc_id = f"{Path(source_path).stem}-{index}"
    text = pick(record, TEXT_KEYS, mapping)
    text = "" if text is None else str(text)
    split = normalize_split(pick(record, SPLIT_KEYS, mapping), default=default_split)
    label_val = pick(record, LABEL_KEYS, mapping)
    labels_val = pick(record, LABELS_KEYS, mapping)
    labels: list[str] | None = None
    if isinstance(labels_val, list):
        labels = [str(x) for x in labels_val]
    elif isinstance(labels_val, str) and labels_val.strip():
        if "," in labels_val:
            labels = [part.strip() for part in labels_val.split(",") if part.strip()]
        else:
            labels = [labels_val]
    label = None if label_val is None or str(label_val).strip() == "" else str(label_val)
    patient = pick(record, PATIENT_KEYS, mapping)
    encounter = pick(record, ENCOUNTER_KEYS, mapping)
    entities = parse_entities(pick(record, ENTITY_KEYS, mapping))
    return Document(
        id=str(doc_id),
        text=text,
        split=split,
        patient_id=None if patient is None or str(patient).strip() == "" else str(patient),
        encounter_id=None if encounter is None or str(encounter).strip() == "" else str(encounter),
        timestamp=parse_timestamp(pick(record, TIMESTAMP_KEYS, mapping)),
        label=label,
        labels=labels,
        entities=entities,
        source_path=source_path,
        source_row=source_row,
    )


def field_mapping(config: dict[str, Any]) -> dict[str, str]:
    fields = config.get("fields")
    mapping: dict[str, str] = {}
    if isinstance(fields, dict):
        mapping.update({str(k): str(v) for k, v in fields.items() if v is not None})
    for key in ("text", "id", "label", "labels", "patient_id", "encounter_id", "timestamp", "split", "entities"):
        if key in config and isinstance(config[key], str):
            mapping[key] = config[key]
    return mapping


def load_config(root: Path) -> dict[str, Any]:
    for name in ("clinicdoc.yaml", "clinicdoc.yml", "clinidoc.yaml", "clinidoc.yml"):
        path = root / name if root.is_dir() else root.parent / name
        if path.is_file():
            import yaml

            with path.open("r", encoding="utf-8") as handle:
                data = yaml.safe_load(handle) or {}
            if isinstance(data, dict):
                return data
    return {}


def _split_files_from_config(root: Path, config: dict[str, Any]) -> dict[str, Path]:
    splits = config.get("splits")
    if not isinstance(splits, dict):
        return {}
    out: dict[str, Path] = {}
    for name, rel in splits.items():
        split = normalize_split(name)
        path = Path(str(rel))
        out[split] = path if path.is_absolute() else root / path
    return out


def detect_format(path: Path, override: str | None = None) -> str:
    if override:
        key = override.lower().strip()
        aliases = {"json": "jsonl", "bio": "conll", "tsv": "conll", "ann": "brat"}
        return aliases.get(key, key)
    if path.is_file():
        suffix = path.suffix.lower()
        if suffix in {".jsonl", ".json"}:
            return "jsonl"
        if suffix in {".ann", ".txt"}:
            return "brat"
        if suffix in {".conll", ".bio"}:
            return "conll"
        if suffix == ".tsv":
            return "conll"
        if suffix == ".csv":
            return "csv"
        return "jsonl"
    names = [p.name.lower() for p in path.iterdir() if p.is_file() or p.is_dir()]
    if any(name.endswith(".ann") for name in names) or any(
        (path / sub).is_dir() and any(child.suffix.lower() == ".ann" for child in (path / sub).glob("*"))
        for sub in ("train", "val", "valid", "test")
        if (path / sub).exists()
    ):
        return "brat"
    jsonl_hits = (
        "train.jsonl",
        "val.jsonl",
        "valid.jsonl",
        "test.jsonl",
        "train.json",
        "val.json",
        "test.json",
    )
    if any(name in jsonl_hits or name.endswith(".jsonl") for name in names):
        return "jsonl"
    if any(name.endswith(".conll") or name.endswith(".bio") for name in names):
        return "conll"
    csv_hits = ("train.csv", "val.csv", "valid.csv", "test.csv")
    if any(name in csv_hits or name.endswith(".csv") for name in names):
        return "csv"
    if any(name.endswith(".tsv") for name in names):
        return "conll"
    return "jsonl"


def load_dataset(path: str | Path, input_format: str | None = None) -> LoadedDataset:
    root = Path(path).expanduser().resolve()
    if not root.exists():
        return LoadedDataset(
            documents=[],
            root=root,
            detected_format=input_format or "unknown",
            files=[],
            load_findings=[
                finding(
                    STRUCTURE_UNREADABLE,
                    f"Path does not exist: {root}",
                    evidence={"path": str(root)},
                )
            ],
        )
    config = load_config(root if root.is_dir() else root.parent)
    fmt = detect_format(root, override=input_format or config.get("format"))
    mapping = field_mapping(config)
    loaders = {
        "jsonl": jsonl_format.load,
        "csv": csv_format.load,
        "brat": brat.load,
        "conll": conll.load,
    }
    loader = loaders.get(fmt)
    if loader is None:
        return LoadedDataset(
            documents=[],
            root=root,
            detected_format=fmt,
            files=[],
            load_findings=[
                finding(
                    STRUCTURE_UNREADABLE,
                    f"Unsupported format: {fmt}",
                    evidence={"format": fmt},
                )
            ],
            config=config,
        )
    documents, files, load_findings = loader(root, mapping=mapping, config=config)
    expected = list(_split_files_from_config(root if root.is_dir() else root.parent, config).keys())
    for split, split_path in _split_files_from_config(root if root.is_dir() else root.parent, config).items():
        if not split_path.exists():
            load_findings.append(
                finding(
                    STRUCTURE_MISSING_SPLIT,
                    f"Declared {split} file is missing: {split_path.name}",
                    split=split,
                    evidence={"path": str(split_path)},
                )
            )
    if not documents and not load_findings:
        load_findings.append(
            finding(
                STRUCTURE_NO_DOCUMENTS,
                f"No documents found under {root}",
                evidence={"path": str(root), "format": fmt},
            )
        )
    return LoadedDataset(
        documents=documents,
        root=root,
        detected_format=fmt,
        files=files,
        load_findings=load_findings,
        config=config,
        expected_splits=expected,
    )
