from __future__ import annotations

from collections import Counter

from clinidoc.dataset import LoadedDataset
from clinidoc.findings import SPLIT_CLASS_MISSING, SPLIT_UNUSABLE, Finding, finding

MIN_VAL_SHARE = 0.05
MIN_DOCS_FOR_SHARE = 20


def scan(dataset: LoadedDataset) -> list[Finding]:
    findings: list[Finding] = []
    by_split = dataset.by_split()
    total = len(dataset.documents)
    if total == 0:
        return findings

    for split, docs in sorted(by_split.items()):
        if len(docs) == 0:
            findings.append(
                finding(
                    SPLIT_UNUSABLE,
                    f"Split {split} is empty",
                    split=split,
                    evidence={"size": 0, "total": total},
                )
            )
        elif split in {"val", "test"} and total >= MIN_DOCS_FOR_SHARE:
            share = len(docs) / total
            if share < MIN_VAL_SHARE and len(docs) < 2:
                findings.append(
                    finding(
                        SPLIT_UNUSABLE,
                        f"Split {split} has only {len(docs)} document(s) ({share:.1%} of corpus)",
                        split=split,
                        evidence={"size": len(docs), "share": round(share, 4), "total": total},
                    )
                )

    train = by_split.get("train") or []
    val = by_split.get("val")
    if train and val is not None:
        train_labels: set[str] = set()
        val_labels: set[str] = set()
        train_counts: Counter[str] = Counter()
        for doc in train:
            for lab in doc.class_labels():
                train_labels.add(lab)
                train_counts[lab] += 1
        for doc in val:
            val_labels.update(doc.class_labels())
        if train_labels:
            missing = sorted(train_labels - val_labels)
            for lab in missing:
                findings.append(
                    finding(
                        SPLIT_CLASS_MISSING,
                        f"Class {lab!r} is present in train but absent from val",
                        split="val",
                        evidence={"label": lab, "train_count": train_counts[lab]},
                    )
                )
    return findings
