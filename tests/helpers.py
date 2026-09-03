from __future__ import annotations

from pathlib import Path

from clinidoc.dataset import Document, Entity, LoadedDataset


def dataset(documents: list[Document], tmp_path: Path | None = None) -> LoadedDataset:
    root = tmp_path if tmp_path is not None else Path(".")
    return LoadedDataset(
        documents=documents,
        root=root,
        detected_format="jsonl",
        files=[],
        config={},
    )


def doc(
    doc_id: str,
    text: str,
    *,
    split: str = "train",
    patient_id: str | None = None,
    label: str | None = None,
    timestamp=None,
    entities: list[Entity] | None = None,
    tokens: list[tuple[str, str]] | None = None,
) -> Document:
    return Document(
        id=doc_id,
        text=text,
        split=split,
        patient_id=patient_id,
        timestamp=timestamp,
        label=label,
        entities=entities,
        tokens=tokens,
    )
