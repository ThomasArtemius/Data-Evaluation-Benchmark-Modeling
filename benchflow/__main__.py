"""python -m benchflow benchmark|predict"""
import argparse
import sys

from .runner import benchmark, predict


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Reproducible tabular model evaluation and prediction")
    commands = parser.add_subparsers(dest="command", required=True)
    evaluate = commands.add_parser("benchmark", help="Compare models on fixed train/validation/test splits")
    evaluate.add_argument("--data", required=True)
    evaluate.add_argument("--target", required=True)
    evaluate.add_argument("--task", choices=["classification", "regression"], required=True)
    evaluate.add_argument("--output", required=True, help="New, nonexistent directory")
    evaluate.add_argument("--id-column")
    evaluate.add_argument("--seed", type=int, default=42)
    evaluate.add_argument("--trees", type=int, default=150)
    evaluate.add_argument("--batch-size", type=int, default=32)
    evaluate.add_argument("--repeats", type=int, default=15)
    infer = commands.add_parser("predict", help="Predict with a trusted saved artifact")
    infer.add_argument("--model", required=True, help="Only load trusted joblib files")
    infer.add_argument("--data", required=True)
    infer.add_argument("--output", required=True, help="New CSV file; never overwrite")
    args = parser.parse_args(argv)
    try:
        if args.command == "benchmark":
            result = benchmark(args.data, args.target, args.task, args.output, args.id_column,
                               args.seed, args.trees, args.batch_size, args.repeats)
            print(f"Selected {result['selected_model']} using validation {result['selection']['metric']}.")
            print(f"Results: {args.output}")
        else:
            rows = predict(args.model, args.data, args.output)
            print(f"Wrote {len(rows)} predictions to {args.output}")
    except (ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
