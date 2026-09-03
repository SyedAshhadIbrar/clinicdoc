from __future__ import annotations

from datetime import datetime

from clinidoc.dataset import Entity
from clinidoc.detectors import scan_all
from clinidoc.detectors.duplicates import scan as scan_duplicates
from clinidoc.detectors.labels import scan as scan_labels
from clinidoc.detectors.leakage import scan as scan_leakage
from clinidoc.detectors.phi import scan as scan_phi
from clinidoc.detectors.spans import scan as scan_spans
from clinidoc.detectors.split import scan as scan_split
from clinidoc.detectors.structure import scan as scan_structure
from clinidoc.detectors.temporal import scan as scan_temporal

from helpers import dataset, doc


def _ids(findings) -> set[str]:
    return {item.id for item in findings}


def test_structure_empty_text() -> None:
    findings = scan_structure(
        dataset(
            [
                doc("ok", "synthetic cough note", label="cough"),
                doc("empty", "", label="cough"),
            ]
        )
    )
    assert "structure.empty_text" in _ids(findings)


def test_structure_empty_corpus() -> None:
    findings = scan_structure(dataset([]))
    assert "structure.empty_corpus" in _ids(findings)


def test_structure_replacement_chars() -> None:
    findings = scan_structure(dataset([doc("bad", "note with \ufffd damage", label="x")]))
    assert "structure.replacement_chars" in _ids(findings)


def test_labels_thin_and_conflict() -> None:
    findings = scan_labels(
        dataset(
            [
                doc("a", "note one about cough", label="cough"),
                doc("a", "note one about cough", label="fever"),
                doc("b", "note two about cough", label="cough"),
                doc("c", "rare syndrome note", label="rare"),
            ]
        )
    )
    assert "labels.conflict" in _ids(findings)
    assert "labels.thin_class" in _ids(findings)


def test_labels_missing() -> None:
    findings = scan_labels(
        dataset(
            [
                doc("a", "labeled cough note", label="cough"),
                doc("b", "unlabeled cough note"),
            ]
        )
    )
    assert "labels.missing" in _ids(findings)


def test_spans_out_of_range_and_inverted() -> None:
    findings = scan_spans(
        dataset(
            [
                doc(
                    "bad",
                    "short",
                    entities=[
                        Entity(start=99, end=120, label="CONDITION"),
                        Entity(start=4, end=1, label="CONDITION"),
                    ],
                )
            ]
        )
    )
    assert "spans.out_of_range" in _ids(findings)
    assert "spans.invalid_range" in _ids(findings)


def test_spans_mismatch_overlap_bio() -> None:
    text = "Patient has pneumonia today"
    start = text.index("pneumonia")
    findings = scan_spans(
        dataset(
            [
                doc(
                    "m",
                    text,
                    entities=[
                        Entity(start=start, end=start + 9, label="CONDITION", text="asthma"),
                        Entity(start=start, end=start + 5, label="CONDITION"),
                    ],
                ),
                doc(
                    "bio",
                    "x y",
                    tokens=[("x", "O"), ("y", "I-CONDITION")],
                ),
            ]
        )
    )
    assert "spans.text_mismatch" in _ids(findings)
    assert "spans.overlap_same_type" in _ids(findings)
    assert "spans.invalid_bio" in _ids(findings)


def test_duplicates_exact() -> None:
    text = "Discharge summary: keep the incision dry."
    findings = scan_duplicates(
        dataset(
            [
                doc("d1", text, label="other"),
                doc("d2", text, label="other"),
            ]
        )
    )
    assert "duplicates.exact" in _ids(findings)


def test_leakage_patient_and_document() -> None:
    findings = scan_leakage(
        dataset(
            [
                doc("t1", "train cough note", split="train", patient_id="P1", label="cough"),
                doc("v1", "val fever note", split="val", patient_id="P1", label="fever"),
                doc("t2", "shared template note", split="train", patient_id="P2"),
                doc("v2", "shared template note", split="val", patient_id="P3"),
            ]
        )
    )
    assert "leakage.patient_across_splits" in _ids(findings)
    assert "leakage.document_across_splits" in _ids(findings)


def test_split_class_missing() -> None:
    findings = scan_split(
        dataset(
            [
                doc("t1", "train cough", split="train", label="cough"),
                doc("t2", "train rare", split="train", label="rare"),
                doc("v1", "val cough", split="val", label="cough"),
            ]
        )
    )
    assert "split.class_missing_from_val" in _ids(findings)


def test_temporal_inversion() -> None:
    findings = scan_temporal(
        dataset(
            [
                doc(
                    "t",
                    "later train note",
                    split="train",
                    patient_id="P1",
                    timestamp=datetime(2024, 6, 1),
                ),
                doc(
                    "v",
                    "earlier val note",
                    split="val",
                    patient_id="P1",
                    timestamp=datetime(2024, 1, 1),
                ),
            ]
        )
    )
    assert "temporal.test_before_train" in _ids(findings)


def test_phi_redacts_ssn() -> None:
    findings = scan_phi(
        dataset([doc("p", "Patient SSN 078-05-1120 was admitted for pneumonia.")])
    )
    assert "phi.ssn" in _ids(findings)
    snippet = findings[0].evidence["snippet"]
    assert "078-05-1120" not in snippet
    assert "***SSN***" in snippet


def test_scan_all_includes_load_findings() -> None:
    ds = dataset([doc("a", "ok note", label="cough")])
    from clinidoc.findings import STRUCTURE_BAD_ROW, finding

    ds.load_findings.append(finding(STRUCTURE_BAD_ROW, "bad row", evidence={"row": 1}))
    findings = scan_all(ds)
    assert "structure.bad_row" in _ids(findings)
