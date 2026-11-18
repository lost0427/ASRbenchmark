"""Interleaved A/B of the Python and native C++ runtimes on the same utterances.

Runs both backends on the same files, alternating, so GPU clock drift affects
them equally. Writes one JSON per model with median stage times and WER.
"""
import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

from windows_benchmark import normalize

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent / "Qwen3-ASR-TensorRT"
DATA = Path(r"C:\Users\z1\Documents\ASR-data\ls")
MANIFEST = ROOT / "data/manifest.jsonl"
sys.path.insert(0, str(PROJECT / "src"))

CONFIGS = {
    "0.6b": {
        "model": PROJECT / "models/Qwen3-ASR-0.6B-hf",
        "packed": PROJECT / "artifacts/qwen3-asr-0.6b/decoder-int4-g32",
        "encoder": PROJECT / "artifacts/qwen3-asr-0.6b/encoder-window-win.bf16.engine",
        "prefill": PROJECT / "artifacts/qwen3-asr-0.6b/decoder-trt-awq32-tc100/asr-prefill.fp16.engine",
    },
    "1.7b": {
        "model": PROJECT / "models/Qwen3-ASR-1.7B-hf",
        "packed": PROJECT / "artifacts/qwen3-asr-1.7b/decoder-int4-g32",
        "encoder": PROJECT / "artifacts/qwen3-asr-1.7b/encoder-window-win.bf16.engine",
        "prefill": PROJECT / "artifacts/qwen3-asr-1.7b/decoder-prefill-2048/asr-prefill.fp16.engine",
    },
}
TIME_RE = re.compile(
    r"mel ([\d.]+) encoder ([\d.]+) prefill ([\d.]+) decode ([\d.]+) total ([\d.]+)"
)


def edit_distance(reference, hypothesis):
    left, right = normalize(reference).split(), normalize(hypothesis).split()
    row = list(range(len(right) + 1))
    for i, token in enumerate(left):
        current = [i + 1]
        for j, other in enumerate(right):
            current.append(min(current[-1] + 1, row[j + 1] + 1, row[j] + (token != other)))
        row = current
    return row[-1], len(left)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", choices=("0.6b", "1.7b"), required=True)
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/runs/qwen3-asr-py-vs-cpp-ab-20261118")
    args = parser.parse_args()

    config = CONFIGS[args.size]
    rows = [json.loads(line) for line in MANIFEST.read_text(encoding="utf-8").splitlines()]
    rows = [row for row in rows if (DATA / row["flac"]).is_file()][: args.limit]
    paths = [DATA / row["flac"] for row in rows]

    from qwen3_asr_trt.runtime import TensorRTW4A16Transcriber

    transcriber = TensorRTW4A16Transcriber(
        str(config["model"]), config["prefill"], config["encoder"],
        max_new_tokens=256, max_cache=512, decoder_dir=config["packed"],
    )

    def run_python():
        stages, edits = [], 0
        started = time.perf_counter()
        for row in rows:
            text = transcriber.transcribe(DATA / row["flac"], "English")
            stages.append(dict(transcriber.last_timings))
            edits += edit_distance(row["reference"], text)[0]
        words = sum(len(normalize(row["reference"]).split()) for row in rows)
        return {
            "medianStagesMs": {
                "preprocessingMs": statistics.median(s["preprocessingMs"] for s in stages),
                "encoderMs": statistics.median(s["encoderMs"] for s in stages),
                "decoderPrefillMs": statistics.median(s["decoderPrefillMs"] for s in stages),
                "decoderGenerationMs": statistics.median(s["decoderGenerationMs"] for s in stages),
            },
            "medianMs": statistics.median(s["totalMs"] for s in stages),
            "elapsedSeconds": time.perf_counter() - started,
            "edits": edits,
            "referenceWords": words,
            "corpusWER": edits / max(1, words),
        }

    def run_cpp():
        started = time.perf_counter()
        run = subprocess.run(
            [str(PROJECT / "native/build/qwen3-asr-native.exe"),
             "--model", str(config["packed"]), "--encoder", str(config["encoder"]),
             "--decoder", str(config["prefill"]), "--assets", str(PROJECT / "native/assets"),
             "--language", "English", *map(str, paths)],
            capture_output=True, text=True,
            env={**os.environ, "QASR_TIMING": "1"},
        )
        elapsed = time.perf_counter() - started
        texts = [json.loads(line)["text"] for line in run.stdout.splitlines() if line.startswith("{")]
        stages = []
        for line in run.stderr.splitlines():
            match = TIME_RE.search(line) if line.startswith("[time]") else None
            if match is None:
                continue
            mel, encoder, prefill, decode, total = (float(match.group(i)) for i in range(1, 6))
            stages.append({"preprocessingMs": mel, "encoderMs": encoder, "decoderPrefillMs": prefill,
                           "decoderGenerationMs": decode, "totalMs": total})
        if len(texts) != len(rows) or len(stages) != len(rows):
            raise SystemExit(f"cpp mismatch: {len(texts)} texts, {len(stages)} timings")
        edits = sum(edit_distance(row["reference"], text)[0] for row, text in zip(rows, texts))
        words = sum(len(normalize(row["reference"]).split()) for row in rows)
        return {
            "medianStagesMs": {key: statistics.median(s[key] for s in stages)
                               for key in ("preprocessingMs", "encoderMs", "decoderPrefillMs", "decoderGenerationMs")},
            "medianMs": statistics.median(s["totalMs"] for s in stages),
            "elapsedSeconds": elapsed,
            "edits": edits,
            "referenceWords": words,
            "corpusWER": edits / max(1, words),
        }

    results = {}
    for index in range(2):
        results.setdefault("python", []).append(run_python())
        results.setdefault("cpp", []).append(run_cpp())
        print(f"pass {index + 1} done", flush=True)

    summary = {"size": args.size, "utterances": len(rows), "passes": results}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / f"{args.size}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for name in ("python", "cpp"):
        stage = statistics.median((p["medianStagesMs"]["decoderGenerationMs"] for p in results[name]))
        total = statistics.median((p.get("medianMs", sum(p["medianStagesMs"].values())) for p in results[name]))
        print(f"{args.size} {name}: gen median {stage:.1f} ms, total {total:.1f} ms")


if __name__ == "__main__":
    main()
