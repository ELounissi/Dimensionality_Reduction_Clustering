"""Run the benchmark or rebuild the paper outputs from the bundled results."""

import argparse
import os
from pathlib import Path

for variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[variable] = "1"

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--from-frozen",
        action="store_true",
        help="Rebuild tables and figures without fitting models.",
    )
    mode.add_argument(
        "--verify",
        action="store_true",
        help="Check archive integrity and every reported fit against its candidate scores.",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument(
        "--output",
        type=Path,
        help="Default: results for frozen outputs, runs/full for a new benchmark.",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.verify:
        from benchmark.validation import verify

        verify(ROOT)
        return
    from benchmark.report import build_report

    if args.from_frozen:
        from benchmark.validation import verify

        verify(ROOT)
        build_report(
            ROOT / "frozen/raw_results.csv.gz", args.output or ROOT / "results"
        )
    else:
        from benchmark.runner import run

        output = (args.output or ROOT / "runs/full").resolve()
        if (
            output == ROOT / "results"
            or output.is_relative_to(ROOT / "frozen")
            or output == ROOT
        ):
            parser.error("A new benchmark needs its own output directory.")
        raw = run(args.data_dir.resolve(), output, args.workers)
        build_report(raw, output / "results")


if __name__ == "__main__":
    main()
