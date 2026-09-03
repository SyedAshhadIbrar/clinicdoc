from __future__ import annotations

from clinidoc.dataset import LoadedDataset
from clinidoc.findings import (
    STRUCTURE_EMPTY_CORPUS,
    STRUCTURE_EMPTY_TEXT,
    STRUCTURE_REPLACEMENT,
    Finding,
    finding,
)


def scan(dataset: LoadedDataset) -> list[Finding]:
    findings: list[Finding] = []
    if not dataset.documents:
        if not any(f.id in {"structure.no_documents", "structure.unreadable_file"} for f in dataset.load_findings):
            findings.append(
                finding(
                    STRUCTURE_EMPTY_CORPUS,
                    "No documents were loaded; the corpus cannot be trained on.",
                    evidence={"path": str(dataset.root)},
                )
            )
        return findings

    empty_ids: list[str] = []
    replacement_ids: list[str] = []
    for doc in dataset.documents:
        if not (doc.text or "").strip():
            empty_ids.append(doc.id)
            findings.append(
                finding(
                    STRUCTURE_EMPTY_TEXT,
                    f"Document {doc.id} has empty text",
                    document_id=doc.id,
                    split=doc.split,
                    evidence={"path": doc.source_path, "row": doc.source_row},
                )
            )
        if "\ufffd" in (doc.text or ""):
            replacement_ids.append(doc.id)
            findings.append(
                finding(
                    STRUCTURE_REPLACEMENT,
                    f"Document {doc.id} contains Unicode replacement characters",
                    document_id=doc.id,
                    split=doc.split,
                    evidence={"path": doc.source_path},
                )
            )

    usable = [d for d in dataset.documents if (d.text or "").strip()]
    if not usable:
        findings.append(
            finding(
                STRUCTURE_EMPTY_CORPUS,
                "Every document has empty text; the corpus cannot be trained on.",
                evidence={"document_ids": empty_ids, "count": len(empty_ids)},
                count=len(empty_ids),
            )
        )
    return findings
