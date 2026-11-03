# Results and interpretation

The delivered framework completes the brief's classification/regression evaluation, baseline comparison and saved-model prediction workflow. Both datasets were processed in full. These are author-run results; independent AI review remains pending.

## Classification: Wine

178 real UCI observations, 13 numeric predictors, three cultivar labels. Seed 42 produced 106 training, 36 validation and 36 held-out test rows. Selection used validation macro-F1.

| Candidate | Validation macro-F1 | Test accuracy | Test macro-F1 | Fit seconds | Median batch prediction, ms |
|---|---:|---:|---:|---:|---:|
| Prior-frequency baseline | 0.186667 | 0.388889 | 0.186667 | 0.0049 | 1.4778 |
| **Logistic regression — selected** | **0.974321** | **1.000000** | **1.000000** | **0.0093** | **1.4920** |
| Random forest | 0.971781 | 1.000000 | 1.000000 | 0.1117 | 7.0637 |

The selected model correctly classified all 36 test rows: 12 class-1, 14 class-2 and 10 class-3 observations. This small, well-separated benchmark does not establish perfect future accuracy. Both learned models obtained the same test accuracy; logistic regression was selected beforehand by validation macro-F1 and also had lower measured prediction latency.

## Regression: Abalone ring count

4,177 real UCI observations, seven numeric predictors and one categorical predictor. Seed 42 produced 2,505 training, 836 validation and 836 held-out test rows. The target is **number of rings**, not age in years. Selection minimized validation RMSE.

| Candidate | Validation RMSE | Test RMSE | Test MAE | Test R² | Fit seconds | Median batch prediction, ms |
|---|---:|---:|---:|---:|---:|---:|
| Mean baseline | 3.350128 | 3.290307 | 2.384594 | -0.000084 | 0.0557 | 2.5881 |
| Ridge | 2.272215 | 2.217827 | 1.599219 | 0.545620 | 0.0071 | 2.6525 |
| **Random forest — selected** | **2.246569** | **2.209199** | **1.554475** | **0.549149** | **0.7869** | **10.7164** |

Random forest had the lowest validation RMSE among these candidates. Ridge was close in error and substantially faster in this run, so a future latency requirement could justify a different explicit selection policy. No significance test or claim of a universally superior model is made. Neither model captures all variation in ring count.

## What was measured

Each table's latency is for **32 rows**, one CPU thread, three warmups and 15 sequential in-process `Pipeline.predict` calls, including fitted preprocessing. It excludes disk I/O, schema casting, model loading and probability generation. Raw observations and p95 values are in each `metrics.json`; they describe this run on this Windows machine, not a service guarantee. Fit times include preprocessing and can contain normal startup/runtime effects.

All candidates shared the same splits. Type inference, imputation, scaling and category fitting used training rows only. No test result was used to select a model. The exact train-fitted winner is persisted without a later refit, so its predictions match the reported held-out evaluation.

The data files retain the source values, with headers and row IDs added. Both original datasets have no missing values; missing-value handling is exercised separately by synthetic verification fixtures. Sources, transformations and SHA-256 hashes are in [data/provenance.json](data/provenance.json).

## Evidence and limits

- `verify_submission.py` verifies all **872** held-out predictions in fresh Python processes, source/model hashes, split/source correspondence, probability sums and independently recomputed selected-model metrics.
- The test suite covers train-only preprocessing, validation-only selection, both prediction tasks, repeatability, unknown categories, missing data, CSV quoting, malformed rows, schema mismatches, ID preservation and refusal to overwrite prior work. See [AUTHOR_VERIFICATION.json](AUTHOR_VERIFICATION.json) for the final command and observed result.
- Rerunning both full benchmarks with the same seed reproduced every model's validation/test metrics and selected model. Runtime measurements may vary.
- No application deployment, time-series/grouped-data validation, large-scale memory test, calibration study or independent reviewer judgment is claimed.

The software's CPU inference timings are not the cost of asking an LLM to review this submission. The later reviewer must measure its own model calls, actual token usage, tool time and memory where available. [REVIEW_HANDOFF.md](REVIEW_HANDOFF.md) explains that separate measurement.
