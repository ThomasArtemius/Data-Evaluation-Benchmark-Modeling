"""Behavior checks using explicitly synthetic fixtures, not reported benchmark data."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from benchflow.data import (cast_features, infer_schema, read_table, split_rows,
                            training_data)
from benchflow.models import make_candidates
from benchflow.runner import benchmark, load_artifact, predict, prediction_table, select_winner

ROOT = Path(__file__).resolve().parents[1]


def synthetic_table(task="classification", rows=120):
    rng = np.random.default_rng(731)
    x = rng.normal(size=rows)
    y = rng.normal(size=rows)
    frame = pd.DataFrame({"id": [f"{i:04d}" for i in range(rows)], "x": x, "y": y,
                          "kind": np.where(y > 0, "blue", "red")})
    frame["target"] = np.where(x + y > 0, "yes", "no") if task == "classification" else 3 * x + y
    frame.loc[::13, "y"] = np.nan
    frame.loc[::17, "kind"] = np.nan
    return frame


@pytest.fixture(scope="module", params=["classification", "regression"])
def fitted(request, tmp_path_factory):
    root = tmp_path_factory.mktemp(request.param)
    data = root / "input.csv"
    synthetic_table(request.param).to_csv(data, index=False)
    output = root / "run"
    report = benchmark(data, "target", request.param, output, "id", trees=12, repeats=3)
    return root, output, report, data


def test_fresh_process_prediction_reproduces_holdout(fitted):
    root, output, report, _ = fitted
    destination = root / "fresh_predictions.csv"
    process = subprocess.run([sys.executable, "-B", "-m", "benchflow", "predict", "--model",
                              str(output / "model.joblib"), "--data", str(output / "holdout_input.csv"),
                              "--output", str(destination)], cwd=ROOT, capture_output=True, text=True)
    assert process.returncode == 0, process.stderr
    expected = pd.read_csv(output / "holdout_predictions.csv", dtype={"row_id": str}).drop(columns="actual")
    actual = pd.read_csv(destination, dtype={"row_id": str})
    pd.testing.assert_frame_equal(expected, actual, check_exact=False, rtol=1e-12, atol=1e-12)
    assert actual["row_id"].str.len().eq(4).all()  # Leading zeroes survive the CLI.
    if report["task"] == "classification":
        np.testing.assert_allclose(actual.filter(like="probability_").sum(axis=1), 1.0)


def test_split_membership_and_order_are_auditable(fitted):
    _, output, report, data = fitted
    original = read_table(data)
    assignment = pd.read_csv(output / "splits.csv", dtype={"row_id": str})
    assert len(assignment) == len(original)
    assert assignment["row_id"].is_unique
    assert set(assignment.row_id) == set(original.id)
    for split, count in report["split_counts"].items():
        group = assignment.loc[assignment.split == split]
        assert len(group) == count
        assert set(group.split_position) == set(range(count))
    expected_ids = assignment.loc[assignment.split == "test"].sort_values("split_position").row_id.tolist()
    assert read_table(output / "holdout_input.csv").id.tolist() == expected_ids


def test_missing_values_and_unseen_categories(fitted):
    _, output, _, _ = fitted
    payload = load_artifact(output / "model.joblib")
    fresh = read_table(output / "holdout_input.csv").iloc[:3].copy()
    fresh.loc[:, "kind"] = "previously-unseen"
    fresh.loc[:, "y"] = np.nan
    result = prediction_table(payload, fresh)
    assert len(result) == 3
    assert result.prediction.notna().all()


def test_invalid_prediction_schema_and_numeric_values(fitted):
    _, output, _, _ = fitted
    payload = load_artifact(output / "model.joblib")
    table = read_table(output / "holdout_input.csv")
    for malformed in [table.drop(columns="x"), table.assign(extra="bad")]:
        with pytest.raises(ValueError, match="schema mismatch"):
            prediction_table(payload, malformed)
    table.loc[table.index[0], "x"] = "not-a-number"
    with pytest.raises(ValueError, match="Non-numeric"):
        prediction_table(payload, table)


def test_existing_outputs_are_preserved(fitted):
    root, output, report, data = fitted
    before = (output / "metrics.json").read_bytes()
    with pytest.raises(ValueError, match="new directory"):
        benchmark(data, "target", report["task"], output, "id", trees=12, repeats=3)
    destination = root / "existing.csv"
    destination.write_text("KEEP", encoding="utf-8")
    with pytest.raises(ValueError, match="overwrite"):
        predict(output / "model.joblib", output / "holdout_input.csv", destination)
    assert destination.read_text() == "KEEP"
    assert (output / "metrics.json").read_bytes() == before


def test_selection_uses_validation_even_when_test_ranking_disagrees():
    records = {
        "a": {"validation": {"f1_macro": .9, "rmse": 1.0}, "test": {"f1_macro": .1, "rmse": 20}},
        "b": {"validation": {"f1_macro": .8, "rmse": 2.0}, "test": {"f1_macro": .99, "rmse": .1}},
    }
    assert select_winner(records, "classification") == "a"
    assert select_winner(records, "regression") == "a"


def test_imputation_and_scaling_are_learned_only_from_training():
    train = pd.DataFrame({"x": [0.0, 2.0, np.nan, 4.0], "category": ["a", "b", "a", "b"]})
    schema = infer_schema(train, "target", None)
    model = make_candidates("classification", schema, 42, trees=2)["logistic_regression"]
    model.fit(cast_features(train, schema), ["n", "y", "n", "y"])
    numeric = model.named_steps["preprocess"].named_transformers_["numeric"]
    assert numeric.named_steps["impute"].statistics_[0] == 2.0
    assert numeric.named_steps["scale"].mean_[0] == 2.0
    model.predict(pd.DataFrame({"x": [1e9, np.nan], "category": ["new", "b"]}))
    assert numeric.named_steps["impute"].statistics_[0] == 2.0
    assert numeric.named_steps["scale"].mean_[0] == 2.0


@pytest.mark.parametrize("contents", ["a,a\n1,2\n", "a,b\n1,2,3\n", "a,b\n1\n", "a,b\n", " a,b\n1,2\n", 'a,b\n1,"unfinished\n'])
def test_rejects_ambiguous_csv(tmp_path, contents):
    path = tmp_path / "bad.csv"
    path.write_text(contents)
    with pytest.raises(ValueError):
        read_table(path)


def test_valid_quoted_csv_is_preserved(tmp_path):
    path = tmp_path / "quoted.csv"
    path.write_bytes('id,note\n001,"two, parts"\n002,"two\nlines"\n'.encode("utf-8"))
    table = read_table(path)
    assert table.id.tolist() == ["001", "002"]
    assert table.note.tolist() == ["two, parts", "two\nlines"]


@pytest.mark.parametrize("fault", ["missing_target", "duplicate_id", "duplicate_features", "nonfinite_target", "target_is_id"])
def test_training_rejects_invalid_data(tmp_path, fault):
    frame = synthetic_table("regression")
    if fault == "missing_target":
        frame.loc[0, "target"] = np.nan
    elif fault == "duplicate_id":
        frame.loc[0, "id"] = frame.loc[1, "id"]
    elif fault == "duplicate_features":
        frame.loc[0, ["x", "y", "kind"]] = frame.loc[1, ["x", "y", "kind"]]
    elif fault == "nonfinite_target":
        frame.loc[0, "target"] = np.inf
    data = tmp_path / "bad.csv"
    frame.to_csv(data, index=False)
    output = tmp_path / "result"
    with pytest.raises(ValueError):
        benchmark(data, "target", "regression", output, "target" if fault == "target_is_id" else "id",
                  trees=2, repeats=3)
    assert not output.exists()


def test_repeated_seed_reproduces_splits_scores_and_predictions(fitted):
    root, output, original, data = fitted
    repeat_output = root / "repeat"
    repeated = benchmark(data, "target", original["task"], repeat_output, "id", trees=12, repeats=3)
    assert original["selected_model"] == repeated["selected_model"]
    for name in original["models"]:
        for partition in ["validation", "test"]:
            assert original["models"][name][partition] == repeated["models"][name][partition]
    for filename in ["splits.csv", "holdout_predictions.csv"]:
        assert (output / filename).read_bytes() == (repeat_output / filename).read_bytes()


def test_without_id_column_has_documented_positional_ids(tmp_path):
    data = tmp_path / "without_id.csv"
    synthetic_table().drop(columns="id").to_csv(data, index=False)
    output = tmp_path / "run"
    benchmark(data, "target", "classification", output, trees=2, repeats=3)
    prediction = pd.read_csv(output / "holdout_predictions.csv", dtype={"row_id": str})
    assert prediction.row_id.tolist() == [str(i) for i in range(len(prediction))]
    assignment = pd.read_csv(output / "splits.csv")
    assert assignment.query("split == 'test'").split_position.is_unique


def test_small_or_rare_class_is_rejected():
    table = synthetic_table().astype(str)
    table["target"] = "common"
    table.loc[:2, "target"] = "rare"
    with pytest.raises(ValueError, match="five rows"):
        training_data(table, "target", "classification", "id")


def test_empty_numeric_column_and_infinite_feature():
    train = pd.DataFrame({"empty": [np.nan] * 4, "x": [1., 2., 3., 4.]})
    schema = infer_schema(train, "target", None)
    model = make_candidates("regression", schema, 1, trees=2)["ridge"]
    model.fit(cast_features(train, schema), [1., 2., 3., 4.])
    assert np.isfinite(model.predict(cast_features(train, schema))).all()
    train.loc[0, "x"] = np.inf
    with pytest.raises(ValueError, match="infinite"):
        cast_features(train, schema)


def test_cli_errors_are_nonzero_and_do_not_create_output(tmp_path):
    process = subprocess.run([sys.executable, "-B", "-m", "benchflow", "benchmark", "--data",
                              str(tmp_path / "missing.csv"), "--target", "label", "--task", "classification",
                              "--output", str(tmp_path / "out")], cwd=ROOT, capture_output=True, text=True)
    assert process.returncode == 2
    assert "Error:" in process.stderr
    assert not (tmp_path / "out").exists()
