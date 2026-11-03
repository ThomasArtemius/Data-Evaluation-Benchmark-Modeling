# Independent review handoff

This folder is the submitted work for **Data Evaluation & Benchmark Modeling**. The author has run its tests, but no independent AI review has been performed. Review the work against the original [BRIEF.md](BRIEF.md); author-written tests, explanations and verification are evidence to inspect, not an authoritative verdict.

## Requested review

1. Read the original brief and determine whether the submission meets its three deliverables and acceptance criteria.
2. Inspect relevant source, dataset provenance, split handling, preprocessing, model selection, metrics, persistence and prediction behavior. Check claims against executable behavior; add independent probes where warranted.
3. Reproduce predictions and relevant tests using README commands. Distinguish code inspection, executed checks, unexplored cases and subjective judgments.
4. Return findings with severity, affected file/line, a reproducible input or observation, and impact. Do not silently fix the submission during the review. Preserve a clear record of the artifact version actually assessed.
5. State accepted, partially met or unmet for each original acceptance criterion, plus the limitations of this one review.

The submission runs locally with no LLM/API credentials. `verify_submission.py` verifies both delivered real-data runs in fresh subprocesses. The source is under `benchflow/`; tests are under `tests/`. The original dataset CSVs and model binaries need not be pasted into an LLM context; inspect them with tools as required. `MANIFEST.json` identifies content by path, byte size and SHA-256, excluding itself.

## Capture the review's inference demand

The purpose of the later measurement is to size a local AI reviewer for Rakitala. The tabular-model timings in `results/` are **not** the inference demand of the reviewer LLM. Record the review separately, without inventing missing telemetry:

- Reviewer model/provider/version, serving engine, quantization, context limit, decoding settings and whether tools or vision were used.
- UTC start/end and total review wall time; separate LLM generation time from file reading, test execution, network work and waiting wherever observable.
- Number of LLM calls, each call's actual input/output tokens, cached-input tokens, and reasoning-token counts only if the provider exposes usage metadata. Record missing values as `null`; do not infer them from file bytes or expose private chain-of-thought.
- Peak per-call input/context size, total uncached/cached tokens, number of tool calls and retries. Count repeated context across calls rather than assuming each file is read once.
- On an instrumented local model: exact GPU/CPU/RAM, peak VRAM and system RAM, idle baseline, tokens/second and request latency. Hosted reviewers often do not expose physical memory or GPU usage; leave those unknown.
- Review quality: criteria checked, defects found, verified true/false positives and any instructor judgment available. A low token count is not sufficient if the review misses correctness issues.

Save the review and its usage record outside this immutable submission, for example under `work/company-gpu/reviews/brief-benchmark/<run-id>/`. A simple `review_usage.json` can use this structure (nulls are intentional placeholders, not measurements):

```json
{
  "submission_manifest_sha256": null,
  "reviewer_model": null,
  "engine_and_quantization": null,
  "started_utc": null,
  "finished_utc": null,
  "wall_seconds": null,
  "llm_seconds": null,
  "tool_seconds": null,
  "calls": [],
  "total_input_tokens": null,
  "total_cached_input_tokens": null,
  "total_output_tokens": null,
  "peak_context_tokens": null,
  "peak_vram_mib": null,
  "peak_system_ram_mib": null,
  "tool_calls": null,
  "retries": null,
  "telemetry_source": null,
  "quality_and_limitations": null
}
```

Each entry in `calls` should contain a call ID, elapsed generation time and provider-reported token usage when available. Include which fields were measured, estimated or unavailable; do not substitute a provider model's nominal context limit for observed context use.

This is one tabular-code submission. Its review cannot establish the demand of visual design, video, large repositories or multi-step application testing. Hardware planning will also need representative additional tasks and measured student/internal concurrency. No such broader benchmark is claimed here.
