"""Run model comparisons and persist an auditable prediction artifact."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, log_loss, mean_absolute_error, mean_squared_error,
                             precision_score, r2_score, recall_score, roc_auc_score)
from threadpoolctl import threadpool_limits

from .data import Schema, cast_features, infer_schema, read_table, row_ids, split_rows, training_data
from .models import make_candidates


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def environment() -> dict:
    return {
        "python": platform.python_version(), "platform": platform.platform(),
        "processor": platform.processor(), "device": "CPU", "threads": 1,
        "packages": {name: importlib.metadata.version(name) for name in
                     ("numpy", "pandas", "scipy", "scikit-learn", "joblib", "threadpoolctl")},
    }


def dump_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")


def metrics(task: str, pipeline, features: pd.DataFrame, truth: pd.Series) -> dict:
    prediction = pipeline.predict(features)
    if task == "regression":
        return {"mae": float(mean_absolute_error(truth, prediction)),
                "rmse": float(np.sqrt(mean_squared_error(truth, prediction))),
                "r2": float(r2_score(truth, prediction))}
    classes = pipeline.named_steps["model"].classes_
    probabilities = pipeline.predict_proba(features)
    scores = {
        "accuracy": float(accuracy_score(truth, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
        "precision_macro": float(precision_score(truth, prediction, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(truth, prediction, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(truth, prediction, average="macro", zero_division=0)),
        "log_loss": float(log_loss(truth, probabilities, labels=classes)),
        "labels": [str(label) for label in classes],
        "confusion_matrix": confusion_matrix(truth, prediction, labels=classes).tolist(),
    }
    if len(classes) == 2:
        scores["roc_auc"] = float(roc_auc_score(truth == classes[1], probabilities[:, 1]))
    else:
        scores["roc_auc"] = float(roc_auc_score(truth, probabilities, labels=classes,
                                                multi_class="ovr", average="macro"))
    return scores


def batch_timing(pipeline, features: pd.DataFrame, batch_size: int, repeats: int) -> dict:
    batch = features.iloc[:min(batch_size, len(features))]
    for _ in range(3):
        pipeline.predict(batch)
    samples = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        pipeline.predict(batch)
        samples.append((time.perf_counter_ns() - start) / 1e6)
    return {"batch_rows": len(batch), "warmup_calls": 3, "repeats": repeats,
            "median_ms": float(np.median(samples)), "p95_ms": float(np.percentile(samples, 95)),
            "samples_ms": samples,
            "scope": "In-process Pipeline.predict, including fitted preprocessing; excludes CSV I/O, "
                     "schema casting, model loading and predict_proba. One CPU thread; sequential calls."}


def select_winner(records: dict, task: str) -> str:
    """Select only by validation quality. Registry order breaks exact ties."""
    if task == "classification":
        return max(records, key=lambda name: records[name]["validation"]["f1_macro"])
    return min(records, key=lambda name: records[name]["validation"]["rmse"])


def prediction_table(payload: dict, table: pd.DataFrame) -> pd.DataFrame:
    schema = Schema(**payload["schema"])
    features = cast_features(table, schema)
    ids = row_ids(table, schema.id_column)
    pipeline = payload["pipeline"]
    with threadpool_limits(limits=1):
        prediction = pipeline.predict(features)
        result = pd.DataFrame({"row_id": ids.to_numpy(), "prediction": prediction})
        if payload["task"] == "classification":
            probabilities = pipeline.predict_proba(features)
            for index in range(probabilities.shape[1]):
                result[f"probability_{index}"] = probabilities[:, index]
    return result


def load_artifact(path: str | Path) -> dict:
    # Joblib is pickle-based: callers must load only trusted model files.
    payload = joblib.load(path)
    if payload.get("artifact_version") != 1:
        raise ValueError("Unsupported model artifact version.")
    if payload["sklearn_version"] != sklearn.__version__:
        raise ValueError("scikit-learn version differs from training; use the pinned requirements.")
    return payload


def predict(model_path: str | Path, input_path: str | Path, output_path: str | Path) -> pd.DataFrame:
    output = Path(output_path)
    if output.exists():
        raise ValueError(f"Refusing to overwrite prediction output: {output}")
    payload = load_artifact(model_path)
    result = prediction_table(payload, read_table(input_path))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="") as handle:
        result.to_csv(handle, index=False)
    return result


def benchmark(input_path: str | Path, target: str, task: str, output: str | Path,
              id_column: str | None = None, seed: int = 42, trees: int = 150,
              batch_size: int = 32, repeats: int = 15) -> dict:
    output = Path(output)
    if output.exists():
        raise ValueError(f"Output must be a new directory to protect existing results: {output}")
    if batch_size < 1 or not 3 <= repeats <= 1000:
        raise ValueError("Batch size must be positive; repeats must be between 3 and 1000.")
    started = datetime.now(timezone.utc).isoformat()
    table = read_table(input_path)
    raw_features, labels, ids = training_data(table, target, task, id_column)
    splits = split_rows(labels, task, seed)
    schema = infer_schema(raw_features.iloc[splits["train"]], target, id_column)
    features = cast_features(raw_features, schema)
    if features.duplicated().any():
        raise ValueError("Duplicate predictor rows after numeric conversion; resolve duplicates first.")
    candidates = make_candidates(task, schema, seed, trees)
    records = {}
    with threadpool_limits(limits=1):
        # Test labels are deliberately not consulted in this selection loop.
        for name, pipeline in candidates.items():
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                start = time.perf_counter()
                pipeline.fit(features.iloc[splits["train"]], labels.iloc[splits["train"]])
                fit_seconds = time.perf_counter() - start
            records[name] = {
                "fit_seconds": fit_seconds,
                "estimator_parameters": pipeline.named_steps["model"].get_params(),
                "fit_warnings": [str(item.message) for item in caught],
                "validation": metrics(task, pipeline, features.iloc[splits["validation"]],
                                      labels.iloc[splits["validation"]]),
            }
        winner = select_winner(records, task)
        for name, pipeline in candidates.items():
            records[name]["test"] = metrics(task, pipeline, features.iloc[splits["test"]],
                                             labels.iloc[splits["test"]])
            records[name]["latency"] = batch_timing(pipeline, features.iloc[splits["test"]],
                                                     batch_size, repeats)

    classes = ([str(x) for x in candidates[winner].named_steps["model"].classes_]
               if task == "classification" else [])
    payload = {"artifact_version": 1, "task": task, "schema": schema.to_dict(),
               "selected_model": winner, "sklearn_version": sklearn.__version__,
               "classes": classes, "pipeline": candidates[winner]}
    result = {
        "started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(),
        "environment": environment(), "input_sha256": sha256(input_path),
        "input_filename": Path(input_path).name, "rows": len(table),
        "schema": schema.to_dict(), "task": task, "seed": seed,
        "split_counts": {name: len(index) for name, index in splits.items()},
        "selected_model": winner,
        "selection": {"split": "validation", "metric": "f1_macro" if task == "classification" else "rmse",
                      "direction": "maximize" if task == "classification" else "minimize",
                      "tie_break": "candidate registry order; simpler baseline first",
                      "refit_after_selection": False},
        "probability_columns": {f"probability_{index}": value for index, value in enumerate(classes)},
        "missing_feature_values": int(raw_features.isna().sum().sum()),
        "models": records,
    }
    # Only create deliverables after all fits, evaluations and timing runs succeed.
    output.mkdir(parents=True, exist_ok=False)
    joblib.dump(payload, output / "model.joblib", compress=3)
    result["model_sha256"] = sha256(output / "model.joblib")
    assignments = pd.DataFrame({"row_id": ids.to_numpy(), "split": "", "split_position": -1})
    for name, index in splits.items():
        assignments.loc[index, "split"] = name
        assignments.loc[index, "split_position"] = np.arange(len(index))
    assignments.to_csv(output / "splits.csv", index=False)
    test_table = table.iloc[splits["test"]].drop(columns=[target])
    test_table.to_csv(output / "holdout_input.csv", index=False)
    predictions = prediction_table(payload, test_table)
    predictions.insert(1, "actual", labels.iloc[splits["test"]].to_numpy())
    predictions.to_csv(output / "holdout_predictions.csv", index=False)
    dump_json(output / "metrics.json", result)
    comparison = []
    for name, record in records.items():
        row = {"model": name, "selected": name == winner, "fit_seconds": record["fit_seconds"],
               "batch_rows": record["latency"]["batch_rows"],
               "predict_batch_median_ms": record["latency"]["median_ms"],
               "predict_batch_p95_ms": record["latency"]["p95_ms"]}
        for split in ("validation", "test"):
            row.update({f"{split}_{key}": value for key, value in record[split].items()
                        if isinstance(value, (int, float))})
        comparison.append(row)
    pd.DataFrame(comparison).to_csv(output / "comparison.csv", index=False)
    write_summary(output, result)
    return result


def write_summary(output: Path, result: dict) -> None:
    key = result["selection"]["metric"]
    lines = [f"# {result['input_filename']} — {result['task']}", "",
             f"Selected **{result['selected_model']}** using validation {key}. "
             "Test results were computed after selection. The persisted model was not refitted.", "",
             "| Model | Validation score | Test score | Fit seconds | Batch predict median ms | p95 ms |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, record in result["models"].items():
        lines.append(f"| {name} | {record['validation'][key]:.6f} | {record['test'][key]:.6f} | "
                     f"{record['fit_seconds']:.4f} | {record['latency']['median_ms']:.4f} | "
                     f"{record['latency']['p95_ms']:.4f} |")
    lines += ["", f"Rows: {result['rows']}; split sizes: {result['split_counts']}; seed: {result['seed']}.",
              "Classification score is macro-F1 (higher is better); regression score is RMSE (lower is better).",
              "", "Timing is measured on this machine, one CPU thread, with three warmups and repeated "
              "predictions on a fixed holdout batch. It is not LLM inference, concurrent throughput or a production SLA.",
              "", "See metrics.json for all metrics, confusion matrices, probability labels, configuration, "
              "raw timing samples, environment and hashes; comparison.csv for the complete comparison table.", ""]
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
