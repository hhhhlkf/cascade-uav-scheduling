from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, Mapping


def ensure_output_dir(path: str | Path) -> Path:
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(exist_ok=True)
    return output_dir


def write_json(path: str | Path, data: Mapping) -> None:
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def write_summary_csv(path: str | Path, summary: Mapping[str, Mapping[str, float]]) -> None:
    rows = []
    for method, metrics in summary.items():
        row = {"method": method}
        row.update(metrics)
        rows.append(row)
    _write_rows(path, rows)


def write_episode_csv(path: str | Path, episodes_by_method: Mapping[str, Iterable[Dict[str, float]]]) -> None:
    rows = []
    for method, episodes in episodes_by_method.items():
        for episode in episodes:
            row = {"method": method}
            row.update(episode)
            rows.append(row)
    _write_rows(path, rows)


def _write_rows(path: str | Path, rows: list[Dict[str, object]]) -> None:
    path = Path(path)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    if "method" in fieldnames:
        fieldnames.insert(0, fieldnames.pop(fieldnames.index("method")))
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

