"""Explicit opt-in local report persistence. Never save raw uploaded files."""
from __future__ import annotations
from datetime import datetime, timezone
import json
from pathlib import Path
from uuid import uuid4


def history_directory() -> Path:
    return Path.home() / ".datasage" / "investigations"


def save_investigation(report: dict, filename: str, target: str | None,
                       directory: Path | None = None) -> Path:
    folder = directory if directory is not None else history_directory()
    folder.mkdir(parents=True, exist_ok=True)
    record = {
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_filename": str(filename),
        "target": target,
        "report": report,
    }
    # UUID names prevent source filenames from being interpreted as file paths.
    destination = folder / (uuid4().hex + ".json")
    destination.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return destination


def list_investigations(directory: Path | None = None, limit: int = 25) -> list[dict]:
    folder = directory if directory is not None else history_directory()
    if not folder.exists():
        return []
    results = []
    for path in sorted(folder.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data.get("report"), dict):
                continue
            results.append({
                "id": path.stem,
                "saved_at_utc": str(data.get("saved_at_utc", "")),
                "source_filename": str(data.get("source_filename", "")),
                "target": data.get("target"),
                "report": data["report"],
            })
        except (ValueError, UnicodeError, OSError, KeyError):
            continue
    return results
