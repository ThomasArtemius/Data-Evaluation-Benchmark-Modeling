"""CSV validation, split construction, and a persisted feature schema."""
from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

MISSING = {"", "?", "NA", "N/A", "null", "None", "NaN", "nan"}


@dataclass
class Schema:
    features: list[str]
    numeric: list[str]
    categorical: list[str]
    target: str
    id_column: str | None

    def to_dict(self) -> dict:
        return asdict(self)


def read_table(path: str | Path) -> pd.DataFrame:
    """Preserve IDs/categories as text and reject ambiguous CSV headers."""
    path = Path(path)
    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle, strict=True)
            header = next(reader, [])
            if not header or any(not name or name != name.strip() for name in header):
                raise ValueError("CSV requires nonblank headers without surrounding spaces.")
            if len(header) != len(set(header)):
                raise ValueError("CSV contains duplicate column names.")
            for line_number, row in enumerate(reader, start=2):
                if len(row) != len(header):
                    raise ValueError(f"CSV row {line_number} has {len(row)} fields; expected {len(header)}.")
    except csv.Error as exc:
        raise ValueError(f"Malformed CSV quoting: {exc}") from exc
    table = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    if table.empty:
        raise ValueError("CSV has no data rows.")
    for column in table:
        table[column] = table[column].str.strip().replace(list(MISSING), np.nan)
    return table


def row_ids(table: pd.DataFrame, id_column: str | None) -> pd.Series:
    if id_column is None:
        return pd.Series([str(i) for i in range(len(table))], index=table.index, name="row_id")
    if id_column not in table:
        raise ValueError(f"Missing ID column: {id_column}")
    values = table[id_column]
    if values.isna().any() or values.duplicated().any():
        raise ValueError("IDs must be nonblank and unique.")
    return values.astype(str).rename("row_id")


def training_data(table: pd.DataFrame, target: str, task: str, id_column: str | None):
    if target not in table:
        raise ValueError(f"Missing target column: {target}")
    if target == id_column:
        raise ValueError("Target and ID columns must be different.")
    if table[target].isna().any():
        raise ValueError("Missing targets are not silently dropped or imputed.")
    if len(table) < 30:
        raise ValueError("At least 30 rows are required for the three-way split.")
    ids = row_ids(table, id_column)
    features = table.drop(columns=[target] + ([id_column] if id_column else []))
    if features.shape[1] == 0:
        raise ValueError("At least one predictor column is required.")
    if features.duplicated().any():
        raise ValueError("Duplicate predictor rows found; resolve duplicates or use a group-aware workflow.")
    if task == "regression":
        try:
            labels = pd.to_numeric(table[target], errors="raise").astype(float)
        except (TypeError, ValueError) as exc:
            raise ValueError("Regression targets must be numeric.") from exc
        if not np.isfinite(labels).all() or labels.nunique() < 2:
            raise ValueError("Regression targets must be finite and nonconstant.")
    elif task == "classification":
        labels = table[target].astype(str)
        counts = labels.value_counts()
        if len(counts) < 2 or counts.min() < 5:
            raise ValueError("Classification requires at least two classes and five rows per class.")
    else:
        raise ValueError("Task must be classification or regression.")
    return features, labels, ids


def split_rows(labels: pd.Series, task: str, seed: int) -> dict[str, np.ndarray]:
    """60/20/20 independent-row split; class-stratified for classification."""
    positions = np.arange(len(labels))
    stratify = labels if task == "classification" else None
    try:
        development, test = train_test_split(positions, test_size=0.2, random_state=seed,
                                             stratify=stratify)
        development_labels = labels.iloc[development] if task == "classification" else None
        train, validation = train_test_split(development, test_size=0.25, random_state=seed + 1,
                                             stratify=development_labels)
    except ValueError as exc:
        raise ValueError(f"Cannot construct the requested split: {exc}") from exc
    splits = {"train": train, "validation": validation, "test": test}
    if task == "classification":
        classes = set(labels)
        if any(set(labels.iloc[index]) != classes for index in splits.values()):
            raise ValueError("Each split must contain every class; supply more rows per class.")
    return splits


def infer_schema(train: pd.DataFrame, target: str, id_column: str | None) -> Schema:
    """Column roles are learned on training rows only, never validation/test."""
    numeric, categorical = [], []
    for column in train:
        observed = train[column].dropna()
        if observed.empty or pd.to_numeric(observed, errors="coerce").notna().all():
            numeric.append(column)
        else:
            categorical.append(column)
    return Schema(list(train.columns), numeric, categorical, target, id_column)


def cast_features(table: pd.DataFrame, schema: Schema) -> pd.DataFrame:
    missing = set(schema.features) - set(table.columns)
    permitted = set(schema.features) | {schema.target, schema.id_column}
    extra = set(table.columns) - permitted
    if missing or extra:
        raise ValueError(f"Feature schema mismatch. Missing={sorted(missing)}, unexpected={sorted(extra)}")
    frame = table.loc[:, schema.features].copy()
    for column in schema.numeric:
        original = frame[column]
        numeric = pd.to_numeric(original, errors="coerce").astype(float)
        if (original.notna() & numeric.isna()).any() or np.isinf(numeric).any():
            raise ValueError(f"Non-numeric or infinite value in numeric feature: {column}")
        frame[column] = numeric
    for column in schema.categorical:
        frame[column] = frame[column].map(lambda value: str(value) if pd.notna(value) else np.nan)
    return frame
