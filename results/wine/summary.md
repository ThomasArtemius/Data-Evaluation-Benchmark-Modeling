# wine.csv — classification

Selected **logistic_regression** using validation f1_macro. Test results were computed after selection. The persisted model was not refitted.

| Model | Validation score | Test score | Fit seconds | Batch predict median ms | p95 ms |
|---|---:|---:|---:|---:|---:|
| dummy_prior | 0.186667 | 0.186667 | 0.0049 | 1.4778 | 1.7174 |
| logistic_regression | 0.974321 | 1.000000 | 0.0093 | 1.4920 | 1.8055 |
| random_forest | 0.971781 | 1.000000 | 0.1117 | 7.0637 | 7.3694 |

Rows: 178; split sizes: {'train': 106, 'validation': 36, 'test': 36}; seed: 42.
Classification score is macro-F1 (higher is better); regression score is RMSE (lower is better).

Timing is measured on this machine, one CPU thread, with three warmups and repeated predictions on a fixed holdout batch. It is not LLM inference, concurrent throughput or a production SLA.

See metrics.json for all metrics, confusion matrices, probability labels, configuration, raw timing samples, environment and hashes; comparison.csv for the complete comparison table.
