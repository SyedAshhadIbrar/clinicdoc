from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from clinidoc.dataset import Document, LoadedDataset, load_dataset
from clinidoc.detectors.duplicates import text_hash
from clinidoc.detectors.leakage import scan as scan_leakage
from clinidoc.findings import Finding


class ResplitError(RuntimeError):
    pass


def _find(parent: dict[int, int], x: int) -> int:
    parent.setdefault(x, x)
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def _union(parent: dict[int, int], a: int, b: int) -> None:
    ra, rb = _find(parent, a), _find(parent, b)
    if ra != rb:
        parent[rb] = ra


def grouped_split(
    documents: list[Document],
    *,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 13,
) -> list[Document]:
    parent: dict[int, int] = {}
    by_patient: dict[str, list[int]] = defaultdict(list)
    by_hash: dict[str, list[int]] = defaultdict(list)
    for i, doc in enumerate(documents):
        _find(parent, i)
        if doc.patient_id:
            by_patient[doc.patient_id].append(i)
        if (doc.text or "").strip():
            by_hash[text_hash(doc.text)].append(i)
    for indexes in by_patient.values():
        for other in indexes[1:]:
            _union(parent, indexes[0], other)
    for indexes in by_hash.values():
        for other in indexes[1:]:
            _union(parent, indexes[0], other)

    groups: dict[int, list[Document]] = defaultdict(list)
    for i, doc in enumerate(documents):
        groups[_find(parent, i)].append(doc)
    keys = list(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(keys)
    n = len(keys)
    n_train = max(1, int(round(n * ratios[0]))) if n else 0
    n_val = int(round(n * ratios[1])) if n > 2 else 0
    if n_train + n_val >= n and n > 1:
        n_val = max(0, n - n_train - 1)
    n_train = min(n_train, n - (1 if n > 1 and n_val == 0 else 0))
    if n >= 3 and n_val == 0:
        n_val = 1
        n_train = min(n_train, n - 2)
    train_keys = set(keys[:n_train])
    val_keys = set(keys[n_train : n_train + n_val])
    out: list[Document] = []
    for key, docs in groups.items():
        if key in train_keys:
            split = "train"
        elif key in val_keys:
            split = "val"
        else:
            split = "test" if n > 2 else "val"
        for doc in docs:
            out.append(
                Document(
                    id=doc.id,
                    text=doc.text,
                    split=split,
                    patient_id=doc.patient_id,
                    encounter_id=doc.encounter_id,
                    timestamp=doc.timestamp,
                    label=doc.label,
                    labels=doc.labels,
                    entities=doc.entities,
                    tokens=doc.tokens,
                    source_path=doc.source_path,
                    source_row=doc.source_row,
                )
            )
    return out


def _doc_to_record(doc: Document) -> dict:
    record: dict = {
        "id": doc.id,
        "text": doc.text,
        "split": doc.split,
    }
    if doc.patient_id is not None:
        record["patient_id"] = doc.patient_id
    if doc.encounter_id is not None:
        record["encounter_id"] = doc.encounter_id
    if doc.timestamp is not None:
        record["timestamp"] = doc.timestamp.isoformat()
    if doc.label is not None:
        record["label"] = doc.label
    if doc.labels is not None:
        record["labels"] = doc.labels
    if doc.entities:
        record["entities"] = [
            {"start": e.start, "end": e.end, "label": e.label, **({"text": e.text} if e.text is not None else {})}
            for e in doc.entities
        ]
    return record


def write_jsonl_splits(documents: list[Document], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    by_split: dict[str, list[Document]] = defaultdict(list)
    for doc in documents:
        by_split[doc.split].append(doc)
    written: list[Path] = []
    for split in ("train", "val", "test"):
        docs = by_split.get(split) or []
        if not docs:
            continue
        path = out_dir / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for doc in docs:
                handle.write(json.dumps(_doc_to_record(doc), ensure_ascii=False) + "\n")
        written.append(path)
    return written


def resplit(
    path: str | Path,
    out_dir: str | Path,
    *,
    by: str = "patient_id",
    seed: int = 13,
    input_format: str | None = None,
) -> tuple[LoadedDataset, list[Finding], list[Path]]:
    if by not in {"patient_id"}:
        raise ResplitError(f"Unsupported grouping: {by}. v1 supports --by patient_id.")
    dataset = load_dataset(path, input_format=input_format)
    if not dataset.documents:
        raise ResplitError("No documents to resplit.")
    rewritten = grouped_split(dataset.documents, seed=seed)
    probe = LoadedDataset(
        documents=rewritten,
        root=Path(out_dir),
        detected_format="jsonl",
        files=[],
        config=dataset.config,
    )
    leaks = scan_leakage(probe)
    if leaks:
        raise ResplitError(
            "Refusing to write a leaking split: " + "; ".join(item.message for item in leaks)
        )
    written = write_jsonl_splits(rewritten, Path(out_dir))
    return dataset, leaks, written
