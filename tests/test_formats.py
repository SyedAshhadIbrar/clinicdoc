from __future__ import annotations

import json
from pathlib import Path

from clinidoc.dataset import load_dataset


def test_jsonl_splits(tmp_path: Path) -> None:
    (tmp_path / "train.jsonl").write_text(
        json.dumps({"id": "a", "text": "alpha note about cough", "label": "cough", "patient_id": "p1"})
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "val.jsonl").write_text(
        json.dumps({"id": "b", "text": "beta note about fever", "label": "fever", "patient_id": "p2", "split": "val"})
        + "\n",
        encoding="utf-8",
    )
    loaded = load_dataset(tmp_path)
    assert loaded.detected_format == "jsonl"
    assert {d.id: d.split for d in loaded.documents} == {"a": "train", "b": "val"}
    assert loaded.documents[0].patient_id == "p1"


def test_csv_loader(tmp_path: Path) -> None:
    (tmp_path / "train.csv").write_text(
        "id,text,label,patient_id\n"
        "c1,synthetic cough note,cough,p9\n",
        encoding="utf-8",
    )
    loaded = load_dataset(tmp_path, input_format="csv")
    assert loaded.detected_format == "csv"
    assert loaded.documents[0].id == "c1"
    assert loaded.documents[0].label == "cough"


def test_brat_loader(tmp_path: Path) -> None:
    text = "SYNTHETIC NOTE. Patient has pneumonia today."
    start = text.index("pneumonia")
    end = start + len("pneumonia")
    train = tmp_path / "train"
    train.mkdir()
    (train / "note1.txt").write_text(text, encoding="utf-8")
    (train / "note1.ann").write_text(
        f"T1\tCONDITION {start} {end}\tpneumonia\n",
        encoding="utf-8",
    )
    loaded = load_dataset(tmp_path, input_format="brat")
    assert loaded.detected_format == "brat"
    assert loaded.documents[0].split == "train"
    assert loaded.documents[0].entities[0].label == "CONDITION"
    assert loaded.documents[0].text[start:end] == "pneumonia"


def test_conll_loader(tmp_path: Path) -> None:
    (tmp_path / "train.conll").write_text(
        "# id = n1\n# patient_id = p1\n# label = pneumonia\n"
        "The O\n"
        "patient O\n"
        "has O\n"
        "pneumonia B-CONDITION\n"
        "today O\n",
        encoding="utf-8",
    )
    loaded = load_dataset(tmp_path, input_format="conll")
    assert loaded.detected_format == "conll"
    doc = loaded.documents[0]
    assert doc.id == "n1"
    assert doc.patient_id == "p1"
    assert any(ent.label == "CONDITION" for ent in (doc.entities or []))
    assert "pneumonia" in doc.text


def test_malformed_jsonl_row(tmp_path: Path) -> None:
    (tmp_path / "train.jsonl").write_text("{not json\n", encoding="utf-8")
    loaded = load_dataset(tmp_path)
    assert any(item.id == "structure.bad_row" for item in loaded.load_findings)


def test_clinicdoc_yaml_mapping(tmp_path: Path) -> None:
    (tmp_path / "clinicdoc.yaml").write_text(
        "text: note\nlabel: dx\npatient_id: subject\n",
        encoding="utf-8",
    )
    (tmp_path / "train.jsonl").write_text(
        json.dumps({"id": "m1", "note": "synthetic fever note", "dx": "fever", "subject": "s1"})
        + "\n",
        encoding="utf-8",
    )
    loaded = load_dataset(tmp_path)
    assert loaded.documents[0].text == "synthetic fever note"
    assert loaded.documents[0].label == "fever"
    assert loaded.documents[0].patient_id == "s1"
