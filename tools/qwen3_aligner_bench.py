"""Benchmark Qwen3-ForcedAligner with the official or TensorRT+EXL3 backend."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / ".native/librispeech"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("official", "tensorrt", "gguf"), required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_DATA / "manifest.jsonl")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--language", default="English")
    parser.add_argument("--aligner-model", default="Qwen/Qwen3-ForcedAligner-0.6B")
    parser.add_argument("--trt-repo", type=Path, default=ROOT.parent / "Qwen3-ASR-TensorRT")
    parser.add_argument("--engine", type=Path)
    parser.add_argument("--decoder-dir", type=Path)
    parser.add_argument("--gguf-repo", type=Path, default=ROOT / ".native/qwen3-asr-gguf")
    parser.add_argument("--gguf-model-dir", type=Path)
    parser.add_argument("--gguf-llm-fn", default="qwen3_aligner_llm.q4_k.gguf")
    parser.add_argument("--gguf-provider", default="DML")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines()]
    rng = np.random.default_rng(args.seed)
    indices = np.sort(rng.choice(len(rows), size=min(args.limit, len(rows)), replace=False))
    sample = [rows[index] for index in indices]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "sample.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in sample),
        encoding="utf-8",
    )

    load_started = time.perf_counter()
    if args.backend == "official":
        import torch
        from qwen_asr import Qwen3ForcedAligner

        runner = Qwen3ForcedAligner.from_pretrained(
            args.aligner_model, device_map="cuda:0", dtype=torch.bfloat16
        )

        def align(audio, text):
            result = runner.align(audio=str(audio), text=text, language=args.language)[0]
            return [
                {"text": item.text, "start_time": item.start_time, "end_time": item.end_time}
                for item in result
            ], None
    elif args.backend == "gguf":
        import soundfile as sf

        sys.path.insert(0, str(args.gguf_repo))
        from qwen_asr_gguf.inference import AlignerConfig, QwenForcedAligner

        model_dir = args.gguf_model_dir or args.gguf_repo / "aligner-model"
        runner = QwenForcedAligner(
            AlignerConfig(
                model_dir=str(model_dir),
                llm_fn=args.gguf_llm_fn,
                onnx_provider=args.gguf_provider,
                llm_use_gpu=True,
            )
        )

        def align(audio, text):
            pcm, sample_rate = sf.read(str(audio), dtype="float32", always_2d=False)
            if sample_rate != 16000:
                raise ValueError(f"expected 16 kHz audio, got {sample_rate}")
            result = runner.align(pcm, text, args.language)
            items = [
                {"text": item.text, "start_time": float(item.start_time),
                 "end_time": float(item.end_time)}
                for item in result
                if item.text.strip()
            ]
            performance = result.performance
            total_ms = performance["total_time"] * 1000
            encoder_ms = performance["encoder_time"] * 1000
            decoder_ms = performance["decoder_time"] * 1000
            return items, {
                "preprocessingMs": max(0.0, total_ms - encoder_ms - decoder_ms),
                "encoderMs": encoder_ms,
                "decoderMs": decoder_ms,
                "totalMs": total_ms,
            }
    else:
        import torch

        sys.path.insert(0, str(args.trt_repo / "src"))
        from qwen3_asr_trt.aligner import TensorRTExLlamaAligner

        aligner_dir = args.trt_repo / "artifacts/qwen3-forcedaligner-0.6b"
        runner = TensorRTExLlamaAligner(
            args.aligner_model,
            args.decoder_dir or aligner_dir / "decoder-exl3-4bpw",
            args.engine or aligner_dir / "aligner-encoder-window.bf16.engine",
        )

        def align(audio, text):
            items = runner.align(Path(audio), text, args.language)
            return items, dict(runner.last_timings)

    load_seconds = time.perf_counter() - load_started
    output = args.output_dir / f"{args.backend}.jsonl"
    done = set()
    if args.resume and output.exists():
        done = {json.loads(line)["id"] for line in output.read_text(encoding="utf-8").splitlines()}
    selected = [row for row in sample if row["id"] not in done]
    for row in selected[: min(2, len(selected))]:
        align(args.data / row["flac"], row["reference"])

    started = time.perf_counter()
    with output.open("a" if args.resume else "w", encoding="utf-8", newline="\n") as handle:
        for index, row in enumerate(selected, 1):
            audio = args.data / row["flac"]
            inference_started = time.perf_counter()
            items, stages = align(audio, row["reference"])
            elapsed_ms = (time.perf_counter() - inference_started) * 1000
            record = {
                "id": row["id"],
                "durationSeconds": row["durationSeconds"],
                "items": items,
                "elapsedMs": elapsed_ms,
            }
            if stages:
                record["stages"] = stages
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            if index % 25 == 0:
                handle.flush()
                print(f"{index}/{len(selected)} ({time.perf_counter() - started:.1f}s)", flush=True)

    all_rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    latencies = [row["elapsedMs"] for row in all_rows]
    stage_names = {name for row in all_rows for name in row.get("stages", {})}
    summary = {
        "engine": f"aligner-{args.backend}",
        "utterances": len(all_rows),
        "items": sum(len(row["items"]) for row in all_rows),
        "elapsedSeconds": time.perf_counter() - started,
        "meanMs": float(np.mean(latencies)),
        "medianMs": float(np.median(latencies)),
        "p90Ms": float(np.percentile(latencies, 90)),
        "audioSeconds": sum(row["durationSeconds"] for row in all_rows),
        "rtf": sum(latencies) / (1000 * sum(row["durationSeconds"] for row in all_rows)),
        "loadSeconds": load_seconds,
        "medianStagesMs": {
            name: float(np.median([row["stages"][name] for row in all_rows if name in row.get("stages", {})]))
            for name in stage_names
        },
        "sampleSize": len(sample),
        "seed": args.seed,
    }
    (args.output_dir / f"{args.backend}.summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
