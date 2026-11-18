"""Benchmark the Python TensorRT W4A16 runtime over the full test-clean manifest.

Same output schema as qwen3_native_cpp_bench.py so the two can be plotted
side by side from the same session (GPU clocks drift between days).
"""
import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

from windows_benchmark import normalize

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent / "Qwen3-ASR-TensorRT"
DATA = Path(r"C:\Users\z1\Documents\ASR-data\ls")
MANIFEST = ROOT / "data/manifest.jsonl"
sys.path.insert(0, str(PROJECT / "src"))

ENGINES = {
    "0.6b": {
        "model": PROJECT / "models/Qwen3-ASR-0.6B-hf",
        "encoder": PROJECT / "artifacts/qwen3-asr-0.6b/encoder-window-win.bf16.engine",
        "decoder": PROJECT / "artifacts/qwen3-asr-0.6b/decoder-trt-awq32-tc100/asr-prefill.fp16.engine",
        "packed": PROJECT / "artifacts/qwen3-asr-0.6b/decoder-int4-g32",
    },
    "1.7b": {
        "model": PROJECT / "models/Qwen3-ASR-1.7B-hf",
        "encoder": PROJECT / "artifacts/qwen3-asr-1.7b/encoder-window-win.bf16.engine",
        "decoder": PROJECT / "artifacts/qwen3-asr-1.7b/decoder-prefill-2048/asr-prefill.fp16.engine",
        "packed": PROJECT / "artifacts/qwen3-asr-1.7b/decoder-int4-g32",
    },
}


def edit_distance(reference, hypothesis):
    left, right = normalize(reference).split(), normalize(hypothesis).split()
    row = list(range(len(right) + 1))
    for i, token in enumerate(left):
        current = [i + 1]
        for j, other in enumerate(right):
            current.append(min(current[-1] + 1, row[j + 1] + 1, row[j] + (token != other)))
        row = current
    return row[-1], len(left), len(right)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", choices=("0.6b", "1.7b"), required=True)
    parser.add_argument("--engine-name", default=None)
    parser.add_argument("--language", default="English")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    from qwen3_asr_trt.runtime import TensorRTW4A16Transcriber

    name = args.engine_name or "w4a16"
    output = args.output or ROOT / "data/runs" / f"qwen3-asr-{args.size}-w4a16-20261118"
    output.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(line) for line in MANIFEST.read_text(encoding="utf-8").splitlines()]
    rows = [row for row in rows if (DATA / row["flac"]).is_file()]
    if args.limit:
        rows = rows[: args.limit]

    engines = ENGINES[args.size]
    transcriber = TensorRTW4A16Transcriber(
        str(engines["model"]),
        engines["decoder"],
        engines["encoder"],
        max_new_tokens=256,
        max_cache=512,
        decoder_dir=engines["packed"],
    )

    records = []
    wall_start = time.perf_counter()
    for index, row in enumerate(rows, 1):
        text = transcriber.transcribe(DATA / row["flac"], args.language)
        timings = dict(transcriber.last_timings)
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
                "elapsedMs": timings["totalMs"],
                "stages": {
                    "preprocessingMs": timings["preprocessingMs"],
                    "encoderMs": timings["encoderMs"],
                    "decoderMs": timings["decoderPrefillMs"] + timings["decoderGenerationMs"],
                    "decoderPrefillMs": timings["decoderPrefillMs"],
                    "decoderGenerationMs": timings["decoderGenerationMs"],
                    "otherMs": timings.get("otherMs", 0.0),
                    "decodeTokens": timings.get("decodeTokens", 0),
                },
            }
        )
        if index % 200 == 0:
            print(f"{index:5d}/{len(rows)} utterances", flush=True)

    elapsed = time.perf_counter() - wall_start
    samples = [record["elapsedMs"] for record in records]
    stage_keys = ("preprocessingMs", "encoderMs", "decoderMs", "decoderPrefillMs", "decoderGenerationMs",
                  "otherMs", "decodeTokens")
    stage_medians = {key: statistics.median(record["stages"][key] for record in records) for key in stage_keys}
    stage_medians["totalMs"] = statistics.median(samples)
    reference_words = sum(record["referenceWords"] for record in records)
    edits = sum(record["edits"] for record in records)
    sorted_samples = sorted(samples)
    summary = {
        "engine": name,
        "provider": "CUDA (TensorRT encoder + fp16 prefill + AWQ g32 W4A16 decode kernels, per-request KV cache) - Python",
        "utterances": len(records),
        "referenceWords": reference_words,
        "hypothesisWords": sum(record["hypothesisWords"] for record in records),
        "edits": edits,
        "corpusWER": edits / max(1, reference_words),
        "emptyOutputs": sum(1 for record in records if not record["hypothesis"].strip()),
        "elapsedSeconds": elapsed,
        "meanMs": statistics.fmean(samples),
        "medianMs": statistics.median(samples),
        "p90Ms": float(sorted_samples[int(0.9 * (len(sorted_samples) - 1))]),
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
    print(json.dumps({key: summary[key] for key in ("medianMs", "p90Ms", "corpusWER", "utterances")}, indent=2))


if __name__ == "__main__":
    main()
