"""Read-only, bounded ingestion. The original dataset is never modified."""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import pandas as pd

from datasage.models import DatasetMetadata

SUPPORTED = {".csv", ".xlsx", ".parquet"}
MAX_BYTES = 50 * 1024 * 1024
MAX_ROWS = 1_000_000
MAX_COLUMNS = 2_000


def load_dataset(path: str | Path) -> tuple[pd.DataFrame, DatasetMetadata]:
    source = Path(path)
    if not source.is_file():
        raise ValueError("Dataset file not found")
    if source.suffix.lower() not in SUPPORTED:
        raise ValueError("Unsupported dataset format")
    if source.stat().st_size > MAX_BYTES:
        raise ValueError("Dataset exceeds the 50 MiB ingestion limit")

    suffix = source.suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(source, nrows=MAX_ROWS + 1)
    elif suffix == ".xlsx":
        frame = pd.read_excel(source, nrows=MAX_ROWS + 1)
    else:
        frame = pd.read_parquet(source)
    if len(frame) > MAX_ROWS or len(frame.columns) > MAX_COLUMNS:
        raise ValueError("Dataset exceeds row or column limits")

    # Hash the input bytes rather than a potentially lossy dataframe serialization.
    digest = sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    metadata = DatasetMetadata(
        rows=len(frame),
        columns=len(frame.columns),
        schema={str(name): str(dtype) for name, dtype in frame.dtypes.items()},
        fingerprint=digest.hexdigest(),
    )
    return frame, metadata
