from __future__ import annotations

from collections import defaultdict
from datetime import datetime

from clinidoc.dataset import LoadedDataset
from clinidoc.findings import TEMPORAL_TEST_BEFORE_TRAIN, Finding, finding

HELD_OUT = {"val", "test"}


def _key(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        return ts
    return ts.replace(tzinfo=None)


def scan(dataset: LoadedDataset) -> list[Finding]:
    findings: list[Finding] = []
    by_patient: dict[str, list] = defaultdict(list)
    for doc in dataset.documents:
        if doc.patient_id and doc.timestamp is not None:
            by_patient[doc.patient_id].append(doc)
    for patient_id, docs in by_patient.items():
        train_times = [_key(d.timestamp) for d in docs if d.split == "train" and d.timestamp]
        held = [d for d in docs if d.split in HELD_OUT and d.timestamp]
        if not train_times or not held:
            continue
        earliest_train = min(train_times)
        latest_train = max(train_times)
        for doc in held:
            held_time = _key(doc.timestamp)
            if held_time < earliest_train:
                findings.append(
                    finding(
                        TEMPORAL_TEST_BEFORE_TRAIN,
                        f"patient_id {patient_id} has {doc.split} note {doc.id} before any train note",
                        document_id=doc.id,
                        split=doc.split,
                        evidence={
                            "patient_id": patient_id,
                            "held_timestamp": doc.timestamp.isoformat(),
                            "earliest_train": min(train_times).isoformat(),
                            "latest_train": max(train_times).isoformat(),
                        },
                    )
                )
    return findings
