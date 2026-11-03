"""Check delivered real-data runs, including prediction from a fresh process.

Creates only a temporary directory, removed when this verification exits.
Does not retrain models, invoke an LLM, or access the network.
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score

from benchflow.data import read_table
from benchflow.runner import sha256

ROOT = Path(__file__).resolve().parent


def verify(results: Path) -> dict:
    checks = []
    provenance = json.loads((ROOT / "data/provenance.json").read_text())
    with tempfile.TemporaryDirectory(prefix="benchflow-verify-") as temporary:
        for name in ["wine", "abalone"]:
            folder = results / name
            report = json.loads((folder / "metrics.json").read_text())
            assert sha256(ROOT / f"data/{name}.csv") == provenance[name]["csv_sha256"] == report["input_sha256"]
            assert sha256(folder / "model.joblib") == report["model_sha256"]
            assignment = pd.read_csv(folder / "splits.csv", dtype={"row_id": str})
            original = read_table(ROOT / f"data/{name}.csv")
            assert assignment.row_id.is_unique
            assert set(assignment.row_id) == set(original.row_id)
            assert set(assignment.split) == {"train", "validation", "test"}
            for split, count in report["split_counts"].items():
                subset = assignment.loc[assignment.split == split]
                assert len(subset) == count
                assert set(subset.split_position) == set(range(count))
            holdout = read_table(folder / "holdout_input.csv")
            test_ids = assignment.loc[assignment.split == "test"].sort_values("split_position").row_id.tolist()
            assert holdout.row_id.tolist() == test_ids
            # Confirm actual holdout feature values came from the stated source rows.
            source_test = original.set_index("row_id").loc[test_ids].drop(columns=report["schema"]["target"])
            pd.testing.assert_frame_equal(source_test.reset_index(), holdout, check_dtype=False)
            expected = pd.read_csv(folder / "holdout_predictions.csv", dtype={"row_id": str})
            expected_truth = original.set_index("row_id").loc[test_ids, report["schema"]["target"]].astype(float)
            np.testing.assert_allclose(expected.actual.astype(float), expected_truth)
            target = Path(temporary) / f"{name}.csv"
            process = subprocess.run([sys.executable, "-B", "-m", "benchflow", "predict", "--model",
                                      str(folder / "model.joblib"), "--data", str(folder / "holdout_input.csv"),
                                      "--output", str(target)], cwd=ROOT, capture_output=True, text=True)
            assert process.returncode == 0, process.stderr
            actual = pd.read_csv(target, dtype={"row_id": str})
            pd.testing.assert_frame_equal(expected.drop(columns="actual"), actual, check_exact=False,
                                          rtol=1e-12, atol=1e-12)
            chosen = report["models"][report["selected_model"]]["test"]
            if name == "wine":
                independently_computed = {"accuracy": accuracy_score(expected.actual, actual.prediction),
                                          "f1_macro": f1_score(expected.actual, actual.prediction, average="macro")}
                np.testing.assert_allclose(actual.filter(like="probability_").sum(axis=1), 1.0)
            else:
                independently_computed = {"mae": mean_absolute_error(expected.actual, actual.prediction),
                                          "rmse": np.sqrt(mean_squared_error(expected.actual, actual.prediction)),
                                          "r2": r2_score(expected.actual, actual.prediction)}
            for key, value in independently_computed.items():
                assert np.isclose(value, chosen[key], atol=1e-12), (name, key, value, chosen[key])
            assert report["selection"]["split"] == "validation"
            assert report["selection"]["refit_after_selection"] is False
            assert all(not model["fit_warnings"] for model in report["models"].values())
            checks.append({"dataset": name, "rows": len(original), "heldout_predictions_verified": len(actual),
                           "input_and_model_hashes_match": True, "split_and_source_rows_match": True,
                           "fresh_process_predictions_match": True, "metrics_recomputed": independently_computed})
    return {"status": "passed", "checks": checks,
            "scope": "Author-run reproducibility verification, not an independent AI review or reviewer inference benchmark."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    print(json.dumps(verify(args.results.resolve()), indent=2))
