# abalone.csv — regression

Selected **random_forest** using validation rmse. Test results were computed after selection. The persisted model was not refitted.

| Model | Validation score | Test score | Fit seconds | Batch predict median ms | p95 ms |
|---|---:|---:|---:|---:|---:|
| dummy_mean | 3.350128 | 3.290307 | 0.0557 | 2.5881 | 3.1654 |
| ridge | 2.272215 | 2.217827 | 0.0071 | 2.6525 | 3.1637 |
| random_forest | 2.246569 | 2.209199 | 0.7869 | 10.7164 | 12.4235 |

Rows: 4177; split sizes: {'train': 2505, 'validation': 836, 'test': 836}; seed: 42.
Classification score is macro-F1 (higher is better); regression score is RMSE (lower is better).

Timing is measured on this machine, one CPU thread, with three warmups and repeated predictions on a fixed holdout batch. It is not LLM inference, concurrent throughput or a production SLA.

See metrics.json for all metrics, confusion matrices, probability labels, configuration, raw timing samples, environment and hashes; comparison.csv for the complete comparison table.
