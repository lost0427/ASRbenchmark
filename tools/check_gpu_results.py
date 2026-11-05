"""Validate an exported Android report before publishing GPU numbers.

Hard failures:
  * a GPU row without backend evidence
  * a GPU row that decodes text but has WER above 15%

Warnings:
  * a GPU row without a CPU control on the same model folder
"""
import argparse
import json
from pathlib import Path
import sys


def is_gpu(row):
    return row.get("accelerator", "CPU") not in ("CPU", "")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--max-wer", type=float, default=0.15)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    rows = report.get("results", [])
    if not rows:
        print("No results in report", file=sys.stderr)
        return 2
    failures, warnings = [], []
    folders_with_cpu = {row.get("modelFolder") for row in rows if not is_gpu(row)}
    for row in rows:
        name = row.get("id", "?")
        if is_gpu(row) and not row.get("backendInfo"):
            failures.append(f"{name}: GPU row has no backend evidence (backendInfo)")
        if is_gpu(row) and row.get("status") == "ok" and row.get("wer", 0) > args.max_wer:
            failures.append(f"{name}: WER {row['wer']:.2%} exceeds {args.max_wer:.0%}")
        if is_gpu(row) and row.get("modelFolder") not in folders_with_cpu:
            warnings.append(f"{name}: no CPU control for {row.get('modelFolder')}")

    rows = sorted(rows, key=lambda r: float("inf") if "meanMs" not in r else r["meanMs"])
    print(f"{'id':22s} {'accelerator':14s} {'compute':26s} {'mean ms':>9s} {'WER':>7s} {'status':16s}")
    for row in rows:
        mean = f"{row['meanMs']:.1f}" if "meanMs" in row else "-"
        wer = f"{row['wer']:.2%}" if "wer" in row else "-"
        print(f"{row.get('id', '?'):22s} {row.get('accelerator', 'CPU'):14s} {row.get('computePrecision', ''):26s} "
              f"{mean:>9s} {wer:>7s} {row.get('status', ''):16s}")
    for message in warnings:
        print("warning:", message)
    for message in failures:
        print("failure:", message)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
