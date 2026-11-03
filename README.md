# Data evaluation and benchmark framework

A completed practice submission for Rakitala's **Data Evaluation & Benchmark Modeling** brief. It compares three models for classification and three for regression, saves the selected preprocessing/model pipeline, and predicts from a new CSV. Read [BRIEF.md](BRIEF.md) for the original task and [REVIEW_HANDOFF.md](REVIEW_HANDOFF.md) for independent review.

## Start here

The bundled `data/` and `results/` are real completed runs. You do not need to download data or retrain to inspect them.

- [Results and interpretation](REPORT.md)
- [Wine comparison](results/wine/summary.md) and [Abalone comparison](results/abalone/summary.md)
- Each result directory includes `comparison.csv`, detailed `metrics.json`, `splits.csv`, the selected `model.joblib`, `holdout_input.csv` and `holdout_predictions.csv`.
- [Dataset provenance and hashes](data/provenance.json)
- [Author verification evidence](AUTHOR_VERIFICATION.json)

Tested with Python **3.11.9** on Windows. All computation is CPU-based. Package versions are pinned in `requirements.txt`. No LLM, API key, paid endpoint or server is needed.

## Setup and verification

Clone the submission, then enter its directory:

```powershell
git clone https://github.com/ThomasArtemius/Data-Evaluation-Benchmark-Modeling.git
cd Data-Evaluation-Benchmark-Modeling
```

From this directory in PowerShell, create an isolated environment if the pinned packages are not already installed:

```powershell
py -3.11 -m venv .venv
$benchPython = '.\.venv\Scripts\python.exe'
& $benchPython -m pip install -r requirements.txt
& $benchPython -B verify_submission.py
& $benchPython -B -m pytest -q -p no:cacheprovider
```

If using the already verified environment on this computer, the equivalent commands are `python -B verify_submission.py` and `python -B -m pytest -q -p no:cacheprovider`.

The verification script checks source/model hashes, split membership and ordering, holdout values, a fresh-process prediction for every holdout row, probability sums and independently recomputed selected-model metrics. It uses an automatically removed temporary directory and makes no network request. The unit/integration tests use explicitly synthetic fixtures for error cases and repeatability; they are separate from the real-data results.

## Reproduce the benchmarks

Use new output directories; existing results are intentionally never overwritten. These commands use the isolated environment above:

```powershell
& $benchPython -B -m benchflow benchmark --data data/wine.csv --target cultivar --task classification --id-column row_id --output runs/wine
& $benchPython -B -m benchflow benchmark --data data/abalone.csv --target rings --task regression --id-column row_id --output runs/abalone
```

The default seed is 42, random forests have 150 trees, and timing uses three warmups followed by 15 calls on a batch of up to 32 holdout rows. Override `--seed`, `--trees`, `--batch-size` or `--repeats` as needed. Compare repeated predictions and metrics, not wall-clock timings or artifact hashes, which need not be identical across environments. After changing settings, treat the new run as a separate experiment.

To verify the reproduction through the same checks: `& $benchPython -B verify_submission.py --results runs`.

## Predict from the saved model

```powershell
& $benchPython -B -m benchflow predict --model results/wine/model.joblib --data results/wine/holdout_input.csv --output runs/wine_predictions.csv
& $benchPython -B -m benchflow predict --model results/abalone/model.joblib --data results/abalone/holdout_input.csv --output runs/abalone_predictions.csv
```

The CSV has one output per input row, in input order:

- `row_id`: supplied unique ID, preserved as text; otherwise the zero-based row position within the input CSV.
- `prediction`: the predicted original class label or numeric regression value.
- `probability_0`, `probability_1`, etc.: classification only. `metrics.json -> probability_columns` maps these columns to original labels. Probabilities are not calibrated guarantees.

The delivered `holdout_predictions.csv` additionally contains `actual` for auditing. Its other columns must match fresh prediction output. If no ID column is used, `splits.csv -> split_position` maps the local holdout row order back to source-file row positions. Always use a real ID column when joining predictions to another system.

Model files are Joblib/pickle artifacts: load only models from a trusted source. The predictor checks the artifact format and exact scikit-learn version. Recreate the pinned environment rather than forcing incompatible model loading.

## Use a different dataset

Supply UTF-8 CSV with a header, one target column, and optionally a unique ID column. Choose `classification` or `regression` explicitly. No code changes are needed for ordinary numeric/categorical tabular datasets; change the input path, target and ID flags.

The loader trims surrounding whitespace. Blank strings, `?`, `NA`, `N/A`, `null`, `None`, `NaN` and `nan` are treated as missing; do not use these as meaningful labels/categories without changing the loader. Missing targets, duplicate IDs, duplicate predictor rows, malformed CSV rows, ambiguous headers and infinite numeric values are rejected. Classification needs at least two classes, five rows per class, every class in each split, and at least 30 total rows. Regression requires finite, nonconstant targets and at least 30 rows.

Feature types are inferred from training rows only. An all-missing training feature is treated as numeric. Numeric missing values receive training medians (zero for wholly missing training columns); numeric features are standardized using training statistics. Categorical missing values use `__MISSING__`, so reserve that token for missingness. Categories are one-hot encoded; unseen categories map to all-zero indicators. Numeric-looking identifiers/categories must be removed, named as the ID, or explicitly represented as text categories if numeric treatment is inappropriate. Invalid numeric tokens in later data fail validation.

Prediction requires exactly the trained predictor names and the configured ID column. The target column may be present but is ignored. Unexpected columns fail validation. Feature order is restored to training order automatically. Exclude post-outcome or target-proxy columns yourself: software cannot infer whether a business feature was available at prediction time.

## Evaluation method

1. Fixed 60/20/20 train/validation/test split; classification is stratified. Actual rounded sizes are recorded.
2. Fit preprocessing and each candidate only on training data.
3. Choose maximum validation macro-F1 for classification or minimum validation RMSE for regression. Exact ties follow registry order (simpler baseline first). Speed is reported alongside quality, not silently included in the selection score.
4. Compute held-out test metrics for all candidates after that choice is frozen. Do not use these test results for another tuning round; use a new outer holdout if iterating.
5. Save the exact selected model as tested, without refitting on validation or test rows. This makes the delivered predictions reproducible.

Classification candidates: prior-frequency dummy, logistic regression and random forest. Regression: mean dummy, Ridge and random forest. Hyperparameters are recorded in `metrics.json`; the readable registry is `benchflow/models.py`. Add models there to extend the comparison. All shipped candidates support the chosen metrics.

Classification metrics: accuracy, balanced accuracy, macro precision/recall/F1, log loss, ROC-AUC and labeled confusion matrix. Binary ROC-AUC treats the second sorted class as positive. Multiclass ROC-AUC uses macro one-versus-rest. Regression metrics: MAE, RMSE and R². Selection and metric definitions are explicit, so comparing the two tasks does not imply that their numeric scores share units.

Latency is sequential, in-process `Pipeline.predict` wall time, including fitted preprocessing. It excludes CSV parsing, schema casting, model loading and probability generation. Fit time includes preprocessing. Thread pools and forests use one CPU thread. Each JSON retains every timing observation plus median/p95. These are measured batch times on this machine, not a concurrent service benchmark.

## Limits and source attribution

This compact framework assumes independent tabular rows and loads data into memory. It does not implement grouped/time-series splits, automated hyperparameter search, uncertainty intervals, calibrated classification probabilities, drift monitoring, streaming, hosting or LLM-based review. Dense one-hot encoding can be unsuitable for high-cardinality or huge data. No universal accuracy or production reliability claim is made.

Public datasets are bundled under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); attribution, URLs, download hashes and modifications are recorded in `data/provenance.json`:

- Aeberhard, S. & Forina, M. (1992). [Wine](https://doi.org/10.24432/C5PC7J), UCI Machine Learning Repository.
- Nash, W., Sellers, T., Talbot, S., Cawthorn, A. & Ford, W. (1994). [Abalone](https://doi.org/10.24432/C55C7W), UCI Machine Learning Repository. The target is ring count; it is not labeled as age in years.

`prepare_data.py --output data-redownloaded` reproduces CSV preparation from the official archives if needed. It retains no ZIP and refuses existing directories. Normal runs use bundled data offline. Numeric strings are retained; only headers and stable IDs are added. See scikit-learn's [leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html) for the training-only preprocessing rationale.
