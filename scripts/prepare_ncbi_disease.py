"""Download NCBI Disease (public PubMed abstracts) and convert to Clinicdoc JSONL."""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

BASE = "https://raw.githubusercontent.com/spyysalo/ncbi-disease/master/original-data"
FILES = {
    "train": f"{BASE}/train/NCBItrainset_corpus.txt",
    "val": f"{BASE}/devel/NCBIdevelopset_corpus.txt",
    "test": f"{BASE}/test/NCBItestset_corpus.txt",
}
LABELS = {"DiseaseClass", "SpecificDisease", "Modifier", "CompositeMention"}


def parse_ann_line(line: str) -> tuple[int, int, str, str] | None:
    if "\t" in line:
        parts = line.split("\t")
        if len(parts) < 5:
            return None
        return int(parts[1]), int(parts[2]), parts[3], parts[4]
    parts = line.split()
    if len(parts) < 5:
        return None
    label_idx = next((i for i, token in enumerate(parts) if token in LABELS), None)
    if label_idx is None or label_idx < 4:
        return None
    return int(parts[1]), int(parts[2]), " ".join(parts[3:label_idx]), parts[label_idx]


def parse_pubtator(raw: str, split: str) -> list[dict]:
    documents: list[dict] = []
    blocks = raw.replace("\r\n", "\n").split("\n\n")
    for block in blocks:
        lines = [line for line in block.split("\n") if line.strip()]
        if not lines:
            continue
        title = ""
        abstract = ""
        pmid: str | None = None
        anns: list[tuple[int, int, str, str]] = []
        for line in lines:
            if "|t|" in line[:20]:
                pmid, _, title = line.partition("|t|")
                pmid = pmid.strip()
            elif "|a|" in line[:20]:
                pmid_a, _, abstract = line.partition("|a|")
                pmid = pmid or pmid_a.strip()
            else:
                parsed = parse_ann_line(line)
                if parsed:
                    anns.append(parsed)
        if not pmid:
            continue
        spaced = f"{title} {abstract}".rstrip()
        joined = f"{title}{abstract}".rstrip()
        text_body = spaced
        entities = []
        for start, end, mention, label in anns:
            slice_spaced = spaced[start:end] if 0 <= start <= end <= len(spaced) else ""
            if slice_spaced == mention:
                text_body = spaced
                entities.append({"start": start, "end": end, "label": label, "text": mention})
                continue
            slice_joined = joined[start:end] if 0 <= start <= end <= len(joined) else ""
            if slice_joined == mention:
                text_body = joined
                entities.append({"start": start, "end": end, "label": label, "text": mention})
            else:
                entities.append({"start": start, "end": end, "label": label, "text": mention})
        documents.append(
            {
                "id": pmid,
                "text": text_body,
                "split": split,
                "entities": entities,
            }
        )
    return documents


def main() -> None:
    out_dir = Path(__file__).resolve().parents[1] / "examples" / "ncbi_disease"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "clinicdoc.yaml").write_text(
        "text: text\nlabel: label\nid: id\nsplit: split\n",
        encoding="utf-8",
    )
    (out_dir / "README.md").write_text(
        "Public NCBI Disease corpus (PubMed abstracts), converted to JSONL for Clinicdoc.\n"
        "Source: https://github.com/spyysalo/ncbi-disease (original NCBI/NLM release).\n"
        "These are scientific abstracts, not hospital EHR notes, and contain no patient identifiers from a medical record.\n",
        encoding="utf-8",
    )
    for split, url in FILES.items():
        print(f"downloading {split} ...")
        with urllib.request.urlopen(url, timeout=60) as response:
            raw = response.read().decode("utf-8")
        docs = parse_pubtator(raw, split)
        path = out_dir / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for doc in docs:
                handle.write(json.dumps(doc, ensure_ascii=False) + "\n")
        print(f"  wrote {len(docs)} documents -> {path}")


if __name__ == "__main__":
    main()
