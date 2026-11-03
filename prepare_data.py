"""Download small official UCI datasets; no archive files are retained."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

SOURCES = {
    "wine": {
        "url": "https://archive.ics.uci.edu/static/public/109/wine.zip", "member": "wine.data",
        "doi": "https://doi.org/10.24432/C5PC7J",
        "citation": "Aeberhard, S. & Forina, M. (1992). Wine. UCI Machine Learning Repository.",
        "columns": ["cultivar", "alcohol", "malic_acid", "ash", "alcalinity_of_ash", "magnesium",
                    "total_phenols", "flavanoids", "nonflavanoid_phenols", "proanthocyanins",
                    "color_intensity", "hue", "od280_od315", "proline"], "rows": 178,
    },
    "abalone": {
        "url": "https://archive.ics.uci.edu/static/public/1/abalone.zip", "member": "abalone.data",
        "doi": "https://doi.org/10.24432/C55C7W",
        "citation": "Nash, W., Sellers, T., Talbot, S., Cawthorn, A., & Ford, W. (1994). Abalone. UCI Machine Learning Repository.",
        "columns": ["sex", "length", "diameter", "height", "whole_weight", "shucked_weight",
                    "viscera_weight", "shell_weight", "rings"], "rows": 4177,
    },
}


def prepare(output: Path) -> None:
    if output.exists():
        raise ValueError("Use a new data directory; bundled data need no download.")
    prepared = []
    for name, source in SOURCES.items():
        request = urllib.request.Request(source["url"], headers={"User-Agent": "Rakitala-Brief-Practice/1.0"})
        with urllib.request.urlopen(request, timeout=45) as response:
            archive = response.read(2_000_001)
        if len(archive) > 2_000_000:
            raise ValueError("Unexpectedly large UCI archive.")
        with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
            raw = bundle.read(source["member"])
        # Preserve the original numeric strings. Add only headers and stable IDs.
        table = pd.read_csv(io.BytesIO(raw), header=None, names=source["columns"], dtype=str,
                            keep_default_na=False)
        if len(table) != source["rows"] or table.shape[1] != len(source["columns"]):
            raise ValueError(f"Unexpected shape for {name}: {table.shape}")
        table.insert(0, "row_id", [f"{name}-{i:05d}" for i in range(len(table))])
        csv_bytes = table.to_csv(index=False, lineterminator="\n").encode("utf-8")
        metadata = {**source, "license": "CC BY 4.0", "license_url": "https://creativecommons.org/licenses/by/4.0/",
                    "retrieved_utc": datetime.now(timezone.utc).isoformat(),
                    "archive_sha256": hashlib.sha256(archive).hexdigest(),
                    "original_data_sha256": hashlib.sha256(raw).hexdigest(),
                    "csv_sha256": hashlib.sha256(csv_bytes).hexdigest(),
                    "modifications": "Added descriptive column headers and stable zero-based row IDs; original values retained."}
        prepared.append((name, csv_bytes, metadata))
    output.mkdir(parents=True, exist_ok=False)
    for name, csv_bytes, metadata in prepared:
        (output / f"{name}.csv").write_bytes(csv_bytes)
    (output / "provenance.json").write_text(json.dumps({name: meta for name, _, meta in prepared}, indent=2) + "\n",
                                           encoding="utf-8")
    print("Prepared Wine (178 rows) and Abalone (4177 rows) from official UCI downloads.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data"))
    prepare(parser.parse_args().output)
