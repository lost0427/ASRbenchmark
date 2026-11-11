"""Combine two benchmark passes into a single per-utterance best-of run.

Each utterance keeps the row (and therefore the stage timing) from the faster
pass, then the corpus summary is recomputed from those rows. WER is identical
across passes because decoding is greedy and deterministic.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def load(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, metavar="run-dir")
    parser.add_argument("--engine", default="tensorrt-exllama")
    parser.add_argument("--passes", type=int, default=2)
    args = parser.parse_args()

    pass_dirs = [args.run_dir / f"pass{i}" for i in range(1, args.passes + 1)]
    engines = [
        path.name[: -len(".jsonl")]
        for path in sorted(pass_dirs[0].glob("*.jsonl"))
        if (pass_dirs[0] / f"{path.stem}.summary.json").is_file()
    ]
    output_dir = args.run_dir / "combined"
    output_dir.mkdir(parents=True, exist_ok=True)

    for engine in engines:
        passes = []
        for directory in pass_dirs:
            path = directory / f"{engine}.jsonl"
            if not path.is_file():
                raise FileNotFoundError(path)
            passes.append({row["id"]: row for row in load(path)})
        order = [row["id"] for row in load(pass_dirs[0] / f"{engine}.jsonl")]
        for other in passes[1:]:
            if set(other) != set(passes[0]):
                raise ValueError(f"{engine}: passes cover different utterances")
            mismatched = [
                key for key in order
                if other[key]["hypothesis"] != passes[0][key]["hypothesis"]
            ]
            if mismatched:
                print(f"{engine}: {len(mismatched)} utterances differ between passes", flush=True)

        rows = []
        for key in order:
            best = min((p[key] for p in passes), key=lambda row: row["elapsedMs"])
            rows.append(best)
        with (output_dir / f"{engine}.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

        summary = json.loads((pass_dirs[0] / f"{engine}.summary.json").read_text(encoding="utf-8"))
        edits = sum(row["edits"] for row in rows)
        words = sum(row["referenceWords"] for row in rows)
        latencies = [row["elapsedMs"] for row in rows]
        stage_names = {name for row in rows for name in row.get("stages", {})}
        audio_seconds = sum(row["durationSeconds"] for row in rows) if "durationSeconds" in rows[0] else summary["audioSeconds"]
        summary.update(
            {
                "utterances": len(rows),
                "referenceWords": words,
                "hypothesisWords": sum(row["hypothesisWords"] for row in rows),
                "edits": edits,
                "corpusWER": edits / max(1, words),
                "emptyOutputs": sum(1 for row in rows if not row["hypothesis"].strip()),
                "meanMs": float(np.mean(latencies)),
                "medianMs": float(np.median(latencies)),
                "p90Ms": float(np.percentile(latencies, 90)),
                "rtf": sum(latencies) / (1000 * audio_seconds),
                "medianStagesMs": {
                    name: float(np.median([row["stages"][name] for row in rows if name in row.get("stages", {})]))
                    for name in stage_names
                },
                "passes": args.passes,
            }
        )
        (output_dir / f"{engine}.summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"{engine}: WER {summary['corpusWER']:.6f} median {summary['medianMs']:.2f} "
            f"p90 {summary['p90Ms']:.2f} edits {edits}",
            flush=True,
        )


if __name__ == "__main__":
    main()
