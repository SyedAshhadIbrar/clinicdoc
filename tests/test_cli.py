from __future__ import annotations

import json
from pathlib import Path

from clinidoc.cli import main


def test_scan_json_clean_dataset(tmp_path: Path, capsys) -> None:
    rows = []
    for split, prefix, start in (("train", "t", 0), ("val", "v", 10), ("test", "s", 20)):
        name = "train.jsonl" if split == "train" else ("val.jsonl" if split == "val" else "test.jsonl")
        chunk = []
        for i in range(4):
            label = "cough" if i % 2 == 0 else "fever"
            chunk.append(
                json.dumps(
                    {
                        "id": f"{prefix}{i}",
                        "text": f"synthetic unique {split} {label} note {start + i} with enough tokens",
                        "label": label,
                        "patient_id": f"{prefix}P{i}",
                        "split": split,
                    }
                )
            )
        (tmp_path / name).write_text("\n".join(chunk) + "\n", encoding="utf-8")
        rows.extend(chunk)
    code = main(["scan", str(tmp_path), "--format", "json"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["verdict"] in {"pass", "caution"}
    assert code in {0, 1}
    if payload["verdict"] == "pass":
        assert code == 0
