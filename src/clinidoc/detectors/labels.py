from __future__ import annotations

from collections import Counter, defaultdict

from clinidoc.dataset import LoadedDataset
from clinidoc.findings import (
    LABELS_CONFLICT,
    LABELS_IMBALANCE,
    LABELS_MISSING,
    LABELS_THIN_CLASS,
    LABELS_UNKNOWN,
    Finding,
    finding,
)

MIN_CLASS_COUNT = 2
IMBALANCE_MAJORITY = 0.9
IMBALANCE_MIN_DOCS = 8


def _is_classification_dataset(dataset: LoadedDataset) -> bool:
    task = str(dataset.config.get("task", "")).strip().lower()
    if task in {"ner", "token_classification"}:
        return False
    if task in {"classification", "classify"}:
        return True
    labeled = sum(1 for d in dataset.documents if d.has_classification())
    if labeled == 0:
        return False
    ner = sum(1 for d in dataset.documents if d.has_ner())
    total = len(dataset.documents)
    if ner and labeled < total * 0.5:
        return labeled >= ner
    return True


def scan(dataset: LoadedDataset) -> list[Finding]:
    findings: list[Finding] = []
    labeled_docs = [d for d in dataset.documents if d.has_classification()]
    if not labeled_docs:
        return findings

    classification_mode = _is_classification_dataset(dataset)
    if classification_mode:
        for doc in dataset.documents:
            if not doc.has_classification() and not doc.has_ner():
                findings.append(
                    finding(
                        LABELS_MISSING,
                        f"Document {doc.id} has no classification label",
                        document_id=doc.id,
                        split=doc.split,
                    )
                )

    allowed = dataset.config.get("labels") or dataset.config.get("label_list")
    if isinstance(allowed, list) and allowed:
        allowed_set = {str(x) for x in allowed}
        for doc in labeled_docs:
            for lab in doc.class_labels():
                if lab not in allowed_set:
                    findings.append(
                        finding(
                            LABELS_UNKNOWN,
                            f"Document {doc.id} has unknown label {lab!r}",
                            document_id=doc.id,
                            split=doc.split,
                            evidence={"label": lab, "allowed": sorted(allowed_set)},
                        )
                    )

    by_id: dict[str, set[str]] = defaultdict(set)
    id_docs: dict[str, list[str]] = defaultdict(list)
    for doc in labeled_docs:
        labs = tuple(sorted(doc.class_labels()))
        by_id[doc.id].add("|".join(labs))
        id_docs[doc.id].append(doc.split)
    for doc_id, variants in by_id.items():
        if len(variants) > 1:
            findings.append(
                finding(
                    LABELS_CONFLICT,
                    f"Document id {doc_id} has conflicting labels: {sorted(variants)}",
                    document_id=doc_id,
                    evidence={"variants": sorted(variants), "splits": id_docs[doc_id]},
                )
            )

    train_docs = [d for d in labeled_docs if d.split == "train"]
    pool = train_docs or labeled_docs
    counts: Counter[str] = Counter()
    for doc in pool:
        for lab in doc.class_labels():
            counts[lab] += 1
    if counts:
        for lab, n in sorted(counts.items()):
            if n < MIN_CLASS_COUNT:
                findings.append(
                    finding(
                        LABELS_THIN_CLASS,
                        f"Class {lab!r} has only {n} example(s) in {'train' if train_docs else 'the corpus'}",
                        split="train" if train_docs else None,
                        evidence={"label": lab, "count": n, "min": MIN_CLASS_COUNT},
                        count=n,
                    )
                )
        total = sum(counts.values())
        if total >= IMBALANCE_MIN_DOCS:
            majority_label, majority_n = counts.most_common(1)[0]
            share = majority_n / total
            if share >= IMBALANCE_MAJORITY:
                findings.append(
                    finding(
                        LABELS_IMBALANCE,
                        f"Class {majority_label!r} is {share:.0%} of labeled documents",
                        evidence={
                            "label": majority_label,
                            "share": round(share, 4),
                            "counts": dict(counts),
                        },
                    )
                )
    return findings
