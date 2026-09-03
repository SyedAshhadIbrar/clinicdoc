# Clinicdoc

Clinicdoc audits labelled clinical NLP datasets (document classification and NER) before you train. It answers: **is this dataset safe to train on, and what must be fixed first?**

It runs **locally**. Findings come from the files on disk. Possible identifier matches are shown as **redacted snippets only**. Note text is not sent to a cloud model.

```
JSONL / CSV / BRAT / BIO
        │
        ▼
  format auto-detect + normalize
        │
        ▼
  detectors
        │
        ▼
  verdict: pass | caution | blocked
  report.md + findings.json + fix_plan.json
```

## Install

Python 3.10+

```bash
pip install -e ".[dev]"
```

## Commands

```bash
clinicdoc scan  ./dataset
clinicdoc audit ./dataset --out out
clinicdoc resplit ./dataset --by patient_id --out ./split
clinicdoc detectors
```

CI:

```bash
clinicdoc audit ./dataset --fail-on critical --format sarif --out out
```

`--fail-on major` also fails on caution. `--fail-on-phi` treats identifier hits as a blocker. `audit` / `scan` exit `2` when the verdict is `blocked`, `0` otherwise.

## Formats (auto-detect, `--input-format` to override)

- Folder with `train.jsonl` / `val.jsonl` / `test.jsonl` (or `valid.jsonl`)
- Single JSONL or CSV with a `split` column
- BRAT: paired `.txt` + `.ann`
- CoNLL / BIO token files

Optional `clinicdoc.yaml` for column mapping (`text`, `label`, `patient_id`, `split`) and a `label_list`.

After load, each document has `id`, `text`, `split`, optional `patient_id` / `encounter_id` / `timestamp`, a classification `label` or `labels[]`, and/or NER `entities[{start,end,label}]`.

## Detectors

**Trainability** (drives the verdict)

| Group | What it catches |
|---|---|
| structure | unreadable files, bad rows, empty text, encoding replacement chars |
| labels | missing/unknown/conflicting labels, thin classes, severe imbalance |
| spans | bad offsets, invalid BIO, same-type overlaps |
| duplicates | exact SHA-256 duplicates; near-duplicates via MinHash |
| leakage | same document across splits; **same `patient_id` in train and val/test** |
| split | class in train missing from val; unusable split sizes |
| temporal | val/test note earlier than train for the same patient |

**Governance** (reported; does not alone block training)

- `phi` — regex for SSN, phone, email, MRN-like tokens, and IP. Evidence is redacted.

**Verdict**

- `blocked` — any critical trainability finding (patient/doc leakage, malformed spans, empty corpus)
- `caution` — only major/minor issues
- `pass` — none of the above
- Identifier hits are `governance` unless `--fail-on-phi`

## Demo

Synthetic notes only (no real patient data):

```bash
clinicdoc audit examples/demo_dataset --out out
```

Planted defects: empty note, `patient_id` in train+val, exact duplicate, bad NER offset, thin class, fake SSN. Verdict should be **blocked**.

## License

Apache-2.0
