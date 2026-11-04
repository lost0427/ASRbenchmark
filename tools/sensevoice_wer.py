"""Evaluate SenseVoice WER over a LibriSpeech manifest."""
import argparse
import gc
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from windows_benchmark import make_runner, normalize

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / ".native/librispeech"
DEFAULT_MODELS = ROOT / "models/low-precision-models"
DEFAULT_TRANSCRIBE = ROOT / ".native/models/transcribe/SenseVoiceSmall-Q8_0.gguf"
DEFAULT_TRANSCRIBE_LIBRARY = (
    ROOT / ".native/transcribe-bin/cpu/transcribe-native-windows-x86_64-cpu-vulkan/transcribe.dll")


def edit_distance(reference, hypothesis):
    left, right = normalize(reference).split(), normalize(hypothesis).split()
    row = list(range(len(right) + 1))
    for i, token in enumerate(left, 1):
        current = [i]
        for j, other in enumerate(right, 1):
            current.append(min(current[-1] + 1, row[j] + 1, row[j - 1] + (token != other)))
        row = current
    return row[-1], len(left), len(right)


def load_pcm(path):
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sample_rate != 16000:
        raise ValueError(f"Expected 16 kHz audio: {path} ({sample_rate})")
    return np.asarray(audio, dtype=np.float32)


def make_transcribe_runner(model_path, library, threads, device_name):
    import os
    import transcribe_cpp
    if library:
        os.environ["TRANSCRIBE_LIBRARY"] = str(library)
    devices = transcribe_cpp.backends()
    if device_name == "cuda":
        device = next((item for item in devices
                       if item.device_type == "gpu" and "CUDA" in item.name), None)
    else:
        device = next((item for item in devices if item.device_type == "cpu"), None)
    if device is None:
        raise RuntimeError(f"No transcribe.cpp {device_name} device")
    model = transcribe_cpp.Model(str(model_path), device=device)
    session = model.session(n_threads=threads)
    kwargs = {"timestamps": "none"}
    if model.supports("itn"):
        kwargs["itn"] = "off"
    return lambda pcm: session.run(pcm, **kwargs).text, device_name


def build_runner(engine, models, threads, transcribe_model, transcribe_library,
                 transcribe_device):
    if engine == "transcribe":
        return make_transcribe_runner(transcribe_model, transcribe_library, threads,
                                      transcribe_device)
    folders = {"sherpa": "onnx", "ort": "onnx", "mnn": "mnn",
               "ncnn": "ncnn", "litert": "litert"}
    if engine not in folders:
        raise ValueError(f"Unknown engine: {engine}")
    return make_runner(engine, models / folders[engine], threads, "cpu")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_DATA / "manifest.jsonl")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--models", type=Path, default=DEFAULT_MODELS)
    parser.add_argument("--engines", nargs="+",
                        default=["sherpa", "ort", "mnn", "ncnn", "litert", "transcribe"])
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DATA / "sensevoice-wer")
    parser.add_argument("--transcribe-model", type=Path, default=DEFAULT_TRANSCRIBE)
    parser.add_argument("--transcribe-library", type=Path, default=DEFAULT_TRANSCRIBE_LIBRARY)
    parser.add_argument("--transcribe-device", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines()]
    if args.offset:
        rows = rows[args.offset:]
    if args.limit:
        rows = rows[:args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for engine in args.engines:
        output = args.output_dir / f"{engine}.jsonl"
        done = set()
        if args.resume and output.exists():
            done = {json.loads(line)["id"] for line in output.read_text(encoding="utf-8").splitlines()}
        selected = [row for row in rows if row["id"] not in done]
        print(f"{engine}: {len(selected)} pending of {len(rows)}", flush=True)
        if not selected:
            continue
        transcribe_library = (args.transcribe_library
                              if args.transcribe_device == "cpu" else None)
        run, provider = build_runner(engine, args.models, args.threads,
                                     args.transcribe_model, transcribe_library,
                                     args.transcribe_device)
        if engine == "litert":
            base_run = run
            def run(pcm, base_run=base_run):
                if len(pcm) / 16000 <= 29.0:
                    return base_run(pcm)
                middle = len(pcm) // 2
                return f"{base_run(pcm[:middle]).strip()} {base_run(pcm[middle:]).strip()}".strip()
        started = time.perf_counter()
        errors = words = hypotheses = empty = 0
        with output.open("a" if args.resume else "w", encoding="utf-8", newline="\n") as handle:
            for index, row in enumerate(selected, 1):
                pcm = load_pcm(args.data / row["flac"])
                hypothesis = run(pcm)
                distance, ref_words, hyp_words = edit_distance(row["reference"], hypothesis)
                errors += distance
                words += ref_words
                hypotheses += hyp_words
                empty += int(not normalize(hypothesis))
                result = {
                    "id": row["id"],
                    "reference": row["reference"],
                    "hypothesis": hypothesis,
                    "edits": distance,
                    "referenceWords": ref_words,
                    "hypothesisWords": hyp_words,
                    "wer": distance / max(1, ref_words),
                }
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                if index % 100 == 0:
                    handle.flush()
                    elapsed = time.perf_counter() - started
                    print(f"{engine}: {index}/{len(selected)} "
                          f"({elapsed:.1f}s, corpus WER {errors / max(1, words):.4%})",
                          flush=True)
        all_rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
        total_errors = sum(item["edits"] for item in all_rows)
        total_words = sum(item["referenceWords"] for item in all_rows)
        total_hyp_words = sum(item["hypothesisWords"] for item in all_rows)
        summary = {
            "engine": engine,
            "provider": provider,
            "utterances": len(all_rows),
            "referenceWords": total_words,
            "hypothesisWords": total_hyp_words,
            "edits": total_errors,
            "corpusWER": total_errors / max(1, total_words),
            "emptyOutputs": sum(1 for item in all_rows if not normalize(item["hypothesis"])),
            "elapsedSeconds": time.perf_counter() - started,
            "threads": args.threads,
            "manifest": str(args.manifest),
        }
        (args.output_dir / f"{engine}.summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary, sort_keys=True), flush=True)
        del run
        gc.collect()


if __name__ == "__main__":
    main()
