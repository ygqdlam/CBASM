#!/usr/bin/env python3
import argparse
import csv
import glob
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_ROOT = os.path.dirname(CURRENT_DIR)
if CODE_ROOT not in sys.path:
    sys.path.insert(0, CODE_ROOT)

DEFAULT_RESULTS_ROOT = os.path.join(CODE_ROOT, "results_revised")


def read_last_csv_row(path):
    with open(path, "r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return rows[-1] if rows else {}


def collect_csv_last_rows(root, pattern):
    records = []
    for path in sorted(glob.glob(os.path.join(root, pattern), recursive=True)):
        row = read_last_csv_row(path)
        row["source_csv"] = path
        records.append(row)
    return records


def write_csv(path, rows):
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    fieldnames = sorted({field for row in rows for field in row.keys()})
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_markdown(path, sections):
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("# Revision Result Summary\n\n")
        for title, rows in sections:
            handle.write("## {}\n\n".format(title))
            if not rows:
                handle.write("No matching CSV files found.\n\n")
                continue
            fields = sorted({field for row in rows for field in row.keys()})
            handle.write("| {} |\n".format(" | ".join(fields)))
            handle.write("| {} |\n".format(" | ".join(["---"] * len(fields))))
            for row in rows:
                handle.write("| {} |\n".format(" | ".join(str(row.get(field, "")) for field in fields)))
            handle.write("\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect the latest train/profile/evaluation CSV rows under a revision result root."
    )
    parser.add_argument("--root", default=DEFAULT_RESULTS_ROOT, help="result root to scan")
    parser.add_argument(
        "--output_dir",
        default=os.path.join(DEFAULT_RESULTS_ROOT, "revision_summary"),
        help="summary output directory",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    sections = [
        ("Training Last Rows", collect_csv_last_rows(args.root, "**/log/seg_*_train_iter.csv")),
        ("Validation Last Rows", collect_csv_last_rows(args.root, "**/log/seg_*_validate_ep.csv")),
        ("Profiling Last Rows", collect_csv_last_rows(args.root, "**/profiling/iteration_profile.csv")),
        ("Metric Summary Rows", collect_csv_last_rows(args.root, "**/summary_metrics_*.csv")),
    ]
    os.makedirs(args.output_dir, exist_ok=True)
    for title, rows in sections:
        filename = title.lower().replace(" ", "_") + ".csv"
        write_csv(os.path.join(args.output_dir, filename), rows)
    markdown_path = os.path.join(args.output_dir, "revision_result_summary.md")
    write_markdown(markdown_path, sections)
    print("wrote summary markdown to {}".format(markdown_path))


if __name__ == "__main__":
    main()
