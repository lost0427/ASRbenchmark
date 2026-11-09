"""Evaluate the official qwen-asr Transformers backend on LibriSpeech test-clean."""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from windows_benchmark import normalize

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / ".native/librispeech"
DEFAULT_PACKAGES = ROOT / ".native/qwen-asr-pkgs"
DEFAULT_MODEL_06 = ROOT.parent / "Qwen3-ASR-TensorRT/models/Qwen3-ASR-0.6B"
DEFAULT_MODEL_17 = ROOT.parent / "Qwen3-ASR-TensorRT/models/Qwen3-ASR-1.7B"


def edit_distance(reference, hypothesis):
    left, right = normalize(reference).split(), normalize(hypothesis).split()
    row = list(range(len(right) + 1))
    for i, token in enumerate(left, 1):
        current = [i]
        for j, other in enumerate(right, 1):
            current.append(min(current[-1] + 1, row[j] + 1, row[j - 1] + (token != other)))
        row = current
    return row[-1], len(left), len(right)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_DATA / "manifest.jsonl")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=2620)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines()]
    sample_path = args.output_dir / "sample.jsonl"
    if sample_path.is_file():
        sample = [json.loads(line) for line in sample_path.read_text(encoding="utf-8").splitlines()]
    else:
        rng = np.random.default_rng(args.seed)
        indices = np.sort(rng.choice(len(rows), size=min(args.sample_size, len(rows)), replace=False))
        sample = [rows[index] for index in indices]
        sample_path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in sample),
            encoding="utf-8",
        )
    if args.limit:
        sample = sample[: args.limit]
    print(f"sample={len(sample)} seed={args.seed}", flush=True)

    import torch
    from qwen_asr import Qwen3ASRModel

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    torch.cuda.empty_cache()
    load_started = time.perf_counter()
    model = Qwen3ASRModel.from_pretrained(
        str(args.model_path),
        device_map="cuda:0",
        dtype=dtype,
        max_inference_batch_size=1,
        max_new_tokens=args.max_new_tokens,
    )
    load_seconds = time.perf_counter() - load_started
    print(f"load={load_seconds:.2f}s", flush=True)

    output = args.output_dir / "official-transformers.jsonl"
    done = set()
    if args.resume and output.exists():
        done = {json.loads(line)["id"] for line in output.read_text(encoding="utf-8").splitlines()}
    selected = [row for row in sample if row["id"] not in done]
    warmups = min(2, len(selected))
    for row in selected[:warmups]:
        model.transcribe(audio=str(args.data / row["flac"]), language="English")
    print(f"warmups={warmups}", flush=True)

    started = time.perf_counter()
    with output.open("a" if args.resume else "w", encoding="utf-8", newline="\n") as handle:
        for index, row in enumerate(selected, 1):
            audio = args.data / row["flac"]
            inference_start = time.perf_counter()
            result = model.transcribe(audio=str(audio), language="English")
            elapsed_ms = (time.perf_counter() - inference_start) * 1000
            hypothesis = result[0].text if result else ""
            distance, ref_words, hyp_words = edit_distance(row["reference"], hypothesis)
            handle.write(json.dumps({
                "id": row["id"],
                "reference": row["reference"],
                "hypothesis": hypothesis,
                "edits": distance,
                "referenceWords": ref_words,
                "hypothesisWords": hyp_words,
                "wer": distance / max(1, ref_words),
                "elapsedMs": elapsed_ms,
            }, ensure_ascii=False) + "\n")
            if index % 25 == 0:
                handle.flush()
                print(f"{index}/{len(selected)} ({time.perf_counter() - started:.1f}s)", flush=True)

    all_rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    edits = sum(item["edits"] for item in all_rows)
    words = sum(item["referenceWords"] for item in all_rows)
    latencies = [item["elapsedMs"] for item in all_rows]
    summary = {
        "engine": "official-transformers",
        "provider": f"qwen-asr 0.0.6 · {args.dtype}",
        "modelPath": str(args.model_path),
        "utterances": len(all_rows),
        "referenceWords": words,
        "hypothesisWords": sum(item["hypothesisWords"] for item in all_rows),
        "edits": edits,
        "corpusWER": edits / max(1, words),
        "emptyOutputs": sum(1 for item in all_rows if not normalize(item["hypothesis"])),
        "elapsedSeconds": time.perf_counter() - started,
        "meanMs": float(np.mean(latencies)),
        "medianMs": float(np.median(latencies)),
        "p90Ms": float(np.percentile(latencies, 90)),
        "audioSeconds": sum(row["durationSeconds"] for row in sample),
        "rtf": sum(latencies) / (1000 * sum(row["durationSeconds"] for row in sample)),
        "loadSeconds": load_seconds,
        "warmups": warmups,
        "sampleSize": len(sample),
        "seed": args.seed,
    }
    (args.output_dir / "official-transformers.summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
