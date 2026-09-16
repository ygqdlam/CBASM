#!/usr/bin/env python3
import argparse
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_ROOT = os.path.dirname(CURRENT_DIR)
if CODE_ROOT not in sys.path:
    sys.path.insert(0, CODE_ROOT)

from revision_utils.stats import paired_significance_rows, write_significance_csv


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run paired significance tests on two per-case metric CSV files."
    )
    parser.add_argument("--baseline_csv", required=True, help="baseline per-case metric CSV")
    parser.add_argument("--candidate_csv", required=True, help="candidate per-case metric CSV")
    parser.add_argument("--output_csv", required=True, help="output significance CSV")
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["dice", "hd95"],
        help="metrics to compare, e.g. dice hd95 asd",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    rows = paired_significance_rows(args.baseline_csv, args.candidate_csv, metrics=args.metrics)
    write_significance_csv(args.output_csv, rows)
    print("wrote significance results to {}".format(args.output_csv))
    for row in rows:
        print(
            "class={class_id} metric={metric} n={n_pairs} diff={mean_diff_candidate_minus_baseline} p={p_value} method={method}".format(
                **row
            )
        )


if __name__ == "__main__":
    main()
