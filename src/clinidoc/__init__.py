"""Clinicdoc: offline clinical NLP dataset auditor.

Findings come from files on disk. Clinical note text is never sent to a network API.
"""

__version__ = "0.1.0"

from clinidoc.dataset import Document, Entity, LoadedDataset, load_dataset
from clinidoc.findings import Finding

__all__ = [
    "Document",
    "Entity",
    "Finding",
    "LoadedDataset",
    "load_dataset",
    "__version__",
]
