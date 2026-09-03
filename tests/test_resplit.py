from __future__ import annotations

import json
from pathlib import Path

from clinidoc.cli import main
from clinidoc.resplit import ResplitError, resplit

import pytest


def test_resplit_keeps_patient_in_one_split(tmp_path: Path) -> None:
    source = tmp_path / "src"
    source.mkdir()
    lines = []
    for i in range(8):
        pid = f"P{i // 2}"
        lines.append(
            json.dumps(
                {
                    "id": f"n{i}",
                    "text": f"synthetic note number {i} about cough",
                    "patient_id": pid,
                    "label": "cough",
                    "split": "train",
                }
            )
        )
    (source / "train.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    out = tmp_path / "split"
    _ds, leaks, written = resplit(source, out, by="patient_id", seed=7)
    assert leaks == []
    assert written
    from clinidoc.dataset import load_dataset
    from clinidoc.detectors.leakage import scan as scan_leakage

    loaded = load_dataset(out)
    assert not scan_leakage(loaded)
    by_patient: dict[str, set[str]] = {}
    for item in loaded.documents:
        by_patient.setdefault(item.patient_id or "", set()).add(item.split)
    assert all(len(splits) == 1 for splits in by_patient.values())


def test_resplit_refuses_unsupported_group(tmp_path: Path) -> None:
    (tmp_path / "train.jsonl").write_text(
        json.dumps({"id": "a", "text": "note", "patient_id": "p"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ResplitError):
        resplit(tmp_path, tmp_path / "out", by="encounter_id")


def test_cli_resplit(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    rows = []
    for i in range(6):
        rows.append(
            json.dumps(
                {
                    "id": f"n{i}",
                    "text": f"unique synthetic cough note {i}",
                    "patient_id": f"P{i}",
                    "label": "cough",
                }
            )
        )
    (src / "train.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")
    out = tmp_path / "out"
    assert main(["resplit", str(src), "--by", "patient_id", "--out", str(out)]) == 0
    assert list(out.glob("*.jsonl"))
