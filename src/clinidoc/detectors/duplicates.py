from __future__ import annotations

import hashlib
import re
from collections import defaultdict

from clinidoc.dataset import Document, LoadedDataset
from clinidoc.findings import DUPLICATES_EXACT, DUPLICATES_NEAR, Finding, finding

NUM_PERM = 64
NEAR_JACCARD = 0.8
SHINGLE_N = 3


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9']+", text.lower())


def shingles(text: str, n: int = SHINGLE_N) -> set[str]:
    tokens = _tokens(text)
    if not tokens:
        return set()
    if len(tokens) < n:
        return {" ".join(tokens)}
    return {" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def minhash(values: set[str], num_perm: int = NUM_PERM) -> tuple[int, ...]:
    if not values:
        return tuple()
    sig = [2**64 - 1] * num_perm
    for item in values:
        digest = hashlib.sha1(item.encode("utf-8")).digest()
        base = int.from_bytes(digest[:8], "big")
        extra = int.from_bytes(digest[8:16], "big")
        for i in range(num_perm):
            # Independent-ish permutations from two 64-bit halves.
            h = (base + (i + 1) * extra + i * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
            if h < sig[i]:
                sig[i] = h
    return tuple(sig)


def estimated_jaccard(a: tuple[int, ...], b: tuple[int, ...]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    hits = sum(1 for x, y in zip(a, b) if x == y)
    return hits / len(a)


def scan(dataset: LoadedDataset) -> list[Finding]:
    findings: list[Finding] = []
    by_hash: dict[str, list[Document]] = defaultdict(list)
    signatures: list[tuple[Document, tuple[int, ...]]] = []
    for doc in dataset.documents:
        text = doc.text or ""
        if not text.strip():
            continue
        digest = text_hash(text)
        by_hash[digest].append(doc)
        signatures.append((doc, minhash(shingles(text))))

    for digest, group in by_hash.items():
        if len(group) < 2:
            continue
        ids = [d.id for d in group]
        splits = sorted({d.split for d in group})
        findings.append(
            finding(
                DUPLICATES_EXACT,
                f"{len(group)} documents share identical text ({', '.join(ids)})",
                evidence={"document_ids": ids, "splits": splits, "hash": digest[:12]},
                count=len(group),
            )
        )

    exact_pairs = set()
    for group in by_hash.values():
        if len(group) < 2:
            continue
        ids = [d.id for d in group]
        for i, left in enumerate(ids):
            for right in ids[i + 1 :]:
                exact_pairs.add(tuple(sorted((left, right))))

    seen_near: set[tuple[str, str]] = set()
    for i, (left, lsig) in enumerate(signatures):
        if not lsig:
            continue
        for right, rsig in signatures[i + 1 :]:
            pair = tuple(sorted((left.id, right.id)))
            if pair in exact_pairs or pair in seen_near:
                continue
            score = estimated_jaccard(lsig, rsig)
            if score >= NEAR_JACCARD:
                seen_near.add(pair)
                findings.append(
                    finding(
                        DUPLICATES_NEAR,
                        f"Near-duplicate notes {left.id} and {right.id} (Jaccard≈{score:.2f})",
                        evidence={
                            "document_ids": [left.id, right.id],
                            "splits": sorted({left.split, right.split}),
                            "jaccard": round(score, 3),
                        },
                    )
                )
    return findings
