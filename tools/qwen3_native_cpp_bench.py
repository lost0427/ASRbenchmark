"""Benchmark the native C++ runtime over the full LibriSpeech test-clean manifest.

Writes the same ``<engine>.jsonl`` / ``<engine>.summary.json`` pair the other
Qwen3-ASR runs use, so the native numbers can be plotted next to the Python ones.
"""
import argparse
import json
import os
import re
import statistics
import subprocess
import time
from pathlib import Path

from windows_benchmark import normalize

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent / "Qwen3-ASR-TensorRT"
DATA = Path(r"C:\Users\z1\Documents\ASR-data\ls")
EXE = PROJECT / "native/build/qwen3-asr-native.exe"
MANIFEST = ROOT / "data/manifest.jsonl"

ENGINES = {
    "0.6b": {
        "model": PROJECT / "artifacts/qwen3-asr-0.6b/decoder-int4-g32",
        "encoder": PROJECT / "artifacts/qwen3-asr-0.6b/encoder-window-win.bf16.engine",
        "decoder": PROJECT / "artifacts/qwen3-asr-0.6b/decoder-trt-awq32-tc100/asr-prefill.fp16.engine",
    },
    "1.7b": {
        "model": PROJECT / "artifacts/qwen3-asr-1.7b/decoder-int4-g32",
        "encoder": PROJECT / "artifacts/qwen3-asr-1.7b/encoder-window-win.bf16.engine",
        "decoder": PROJECT / "artifacts/qwen3-asr-1.7b/decoder-prefill-2048/asr-prefill.fp16.engine",
    },
}

TIME_RE = re.compile(
    r"mel ([\d.]+) encoder ([\d.]+) prefill ([\d.]+) decode ([\d.]+) total ([\d.]+) "
    r"\| frames (\d+) prompt (\d+) tokens (\d+)"
)


def edit_distance(reference, hypothesis):
    left, right = normalize(reference).split(), normalize(hypothesis).split()
    row = list(range(len(right) + 1))
    for i, token in enumerate(left):
        current = [i + 1]
        for j, other in enumerate(right):
            current.append(min(current[-1] + 1, row[j + 1] + 1, row[j] + (token != other)))
        row = current
    return row[-1], len(left), len(right)


def run_batch(paths, engines, language):
    command = [
        str(EXE),
        "--model", str(engines["model"]),
        "--encoder", str(engines["encoder"]),
        "--decoder", str(engines["decoder"]),
        "--assets", str(PROJECT / "native/assets"),
        "--language", language,
        *paths,
    ]
    env = dict(os.environ)
    env["QASR_TIMING"] = "1"
    result = subprocess.run(command, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        raise SystemExit(f"native run failed:\n{result.stderr[-800:]}")
    texts = [json.loads(line)["text"] for line in result.stdout.splitlines() if line.startswith("{")]
    timings = [TIME_RE.search(line) for line in result.stderr.splitlines() if line.startswith("[time]")]
    if len(texts) != len(paths) or len(timings) != len(paths) or any(t is None for t in timings):
        raise SystemExit(f"batch mismatch: {len(paths)} paths, {len(texts)} texts, {len(timings)} timings")
    return texts, timings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", choices=("0.6b", "1.7b"), required=True)
    parser.add_argument("--engine-name", default=None)
    parser.add_argument("--language", default="English")
    parser.add_argument("--batch", type=int, default=200)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    name = args.engine_name or f"native-cpp"
    output = args.output or ROOT / "data/runs" / f"qwen3-asr-{args.size}-native-cpp-full-test-clean"
    output.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(line) for line in MANIFEST.read_text(encoding="utf-8").splitlines()]
    rows = [row for row in rows if (DATA / row["flac"]).is_file()]
    if args.limit:
        rows = rows[: args.limit]

    engines = ENGINES[args.size]
    records = []
    wall_start = time.perf_counter()
    for start in range(0, len(rows), args.batch):
        chunk = rows[start : start + args.batch]
        paths = [str(DATA / row["flac"]) for row in chunk]
        texts, timings = run_batch(paths, engines, args.language)
        for row, text, timing in zip(chunk, texts, timings):
            mel = float(timing.group(1))
            encoder = float(timing.group(2))
            prefill = float(timing.group(3))
            decode = float(timing.group(4))
            total = float(timing.group(5))
            edits, reference_words, hypothesis_words = edit_distance(row["reference"], text)
            records.append(
                {
                    "id": row["id"],
                    "reference": row["reference"],
                    "hypothesis": text,
                    "edits": edits,
                    "referenceWords": reference_words,
                    "hypothesisWords": hypothesis_words,
                    "wer": edits / max(1, reference_words),
                    "elapsedMs": total,
                    "stages": {
                        "preprocessingMs": mel,
                        "encoderMs": encoder,
                        "decoderMs": prefill + decode,
                        "decoderPrefillMs": prefill,
                        "decoderGenerationMs": decode,
                        "otherMs": max(0.0, total - mel - encoder - prefill - decode),
                        "decodeTokens": int(timing.group(8)),
                    },
                }
            )
        print(f"{start + len(chunk):5d}/{len(rows)} utterances", flush=True)

    elapsed = time.perf_counter() - wall_start
    samples = [record["elapsedMs"] for record in records]
    stage_keys = ("preprocessingMs", "encoderMs", "decoderMs", "decoderPrefillMs", "decoderGenerationMs", "otherMs")
    stage_medians = {key: statistics.median(record["stages"][key] for record in records) for key in stage_keys}
    stage_medians["totalMs"] = statistics.median(samples)
    stage_medians["decodeTokens"] = statistics.median(record["stages"]["decodeTokens"] for record in records)
    reference_words = sum(record["referenceWords"] for record in records)
    edits = sum(record["edits"] for record in records)
    summary = {
        "engine": name,
        "provider": "CUDA (TensorRT encoder + prefill, AWQ g32 INT4 decode kernels, CUDA graph) - native C++",
        "utterances": len(records),
        "referenceWords": reference_words,
        "hypothesisWords": sum(record["hypothesisWords"] for record in records),
        "edits": edits,
        "corpusWER": edits / max(1, reference_words),
        "emptyOutputs": sum(1 for record in records if not record["hypothesis"].strip()),
        "elapsedSeconds": elapsed,
        "meanMs": statistics.fmean(samples),
        "medianMs": statistics.median(samples),
        "p90Ms": float(sorted(samples)[int(0.9 * (len(samples) - 1))]),
        "audioSeconds": sum(row["durationSeconds"] for row in rows),
        "rtf": statistics.median(samples) / 1000.0 / (sum(row["durationSeconds"] for row in rows) / len(rows)),
        "medianStagesMs": stage_medians,
        "warmups": 0,
        "sampleSize": len(records),
        "seed": 1337,
    }

    (output / f"{name}.jsonl").write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n", encoding="utf-8"
    )
    (output / f"{name}.summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("medianMs", "p90Ms", "corpusWER", "utterances")}, indent=2))


if __name__ == "__main__":
    main()
