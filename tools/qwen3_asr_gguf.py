"""Benchmark Qwen3-ASR-GGUF (ONNX int4 encoder + GGUF q4_k decoder)."""
import argparse
import contextlib
import io
import json
import sys
import time
from pathlib import Path

import numpy as np

from windows_benchmark import wer

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO = ROOT / ".native/qwen3-asr-gguf"
DEFAULT_AUDIO = ROOT / "app/src/main/assets/audio/sample.wav"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--provider", choices=["dml", "cpu"], default="dml")
    parser.add_argument("--chunk-size", type=float, default=40.0)
    parser.add_argument("--n-ctx", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.warmups < 0 or args.repeats < 1:
        parser.error("warmups must be nonnegative; repeats must be positive")

    repo = args.repo.resolve()
    if not (repo / "qwen_asr_gguf").is_dir():
        raise SystemExit(f"Qwen3-ASR-GGUF repo not found: {repo}")
    sys.path.insert(0, str(repo))
    from qwen_asr_gguf.inference import ASREngineConfig, QwenASREngine
    from qwen_asr_gguf.inference.audio import load_audio

    model_dir = (args.model_dir or repo / "model").resolve()
    config = ASREngineConfig(
        model_dir=str(model_dir),
        encoder_frontend_fn="qwen3_asr_encoder_frontend.int4.onnx",
        encoder_backend_fn="qwen3_asr_encoder_backend.int4.onnx",
        llm_fn="qwen3_asr_llm.q4_k.gguf",
        onnx_provider="DML" if args.provider == "dml" else "CPU",
        llm_use_gpu=args.provider == "dml",
        dml_pad_to=args.chunk_size,
        n_ctx=args.n_ctx,
        chunk_size=args.chunk_size,
        memory_num=1,
        enable_aligner=False,
        verbose=False,
    )
    reference = None
    reference_path = args.reference or args.audio.with_suffix(".json")
    if reference_path.is_file():
        reference = json.loads(reference_path.read_text(encoding="utf-8")).get("reference")
    started = time.perf_counter()
    engine = QwenASREngine(config)
    load_ms = (time.perf_counter() - started) * 1000
    audio = load_audio(str(args.audio))
    seconds = len(audio) / 16000

    def run():
        start = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            result = engine.asr(audio, context="", language="English",
                                chunk_size_sec=args.chunk_size, memory_chunks=1,
                                temperature=args.temperature, rollback_num=5)
        return (time.perf_counter() - start) * 1000, result

    for _ in range(args.warmups):
        run()
    samples, decode_tokens, prefill_tokens = [], [], []
    result = None
    for _ in range(args.repeats):
        elapsed, result = run()
        samples.append(elapsed)
        decode_tokens.append(result.performance["decode_tokens"])
        prefill_tokens.append(result.performance["prefill_tokens"])
    text = result.text.strip()
    report = {
        "engine": "qwen3-asr-gguf",
        "provider": "DML+Vulkan" if args.provider == "dml" else "CPU",
        "model": str(model_dir),
        "audioSeconds": seconds,
        "loadMs": load_ms,
        "meanMs": float(np.mean(samples)),
        "medianMs": float(np.median(samples)),
        "p90Ms": float(np.percentile(samples, 90)),
        "rtf": float(np.median(samples)) / (seconds * 1000),
        "prefillTokens": int(np.median(prefill_tokens)),
        "decodeTokens": int(np.median(decode_tokens)),
        "text": text,
    }
    if reference is not None:
        report["wer"] = wer(reference, text)
    print(json.dumps({key: value for key, value in report.items() if key != "text"},
                     sort_keys=True), flush=True)
    print(f"TEXT: {text}", flush=True)
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
