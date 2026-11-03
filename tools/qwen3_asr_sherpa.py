"""Benchmark Qwen3-ASR through sherpa-onnx."""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from windows_benchmark import load_audio, wer

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUDIO = ROOT / "app/src/main/assets/audio/sample.wav"


def clean_text(text):
    import re
    if "<asr_text>" in text:
        text = text.split("<asr_text>", 1)[1]
    return re.sub(r"^\s*language\s+[A-Za-z]+\s*", "", text, flags=re.IGNORECASE).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--provider", default="cpu")
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.warmups < 0 or args.repeats < 1:
        parser.error("warmups must be nonnegative; repeats must be positive")
    try:
        import sherpa_onnx
    except ImportError as error:
        raise SystemExit("Install tools/qwen3-asr-requirements.txt") from error

    folder = args.model_dir.resolve()
    reference = None
    reference_path = args.reference or args.audio.with_suffix(".json")
    if reference_path.is_file():
        reference = json.loads(reference_path.read_text(encoding="utf-8")).get("reference")
    pcm = load_audio(args.audio)
    recognizer = sherpa_onnx.OfflineRecognizer.from_qwen3_asr(
        conv_frontend=str(folder / "conv_frontend.onnx"),
        encoder=str(folder / "encoder.int8.onnx"),
        decoder=str(folder / "decoder.int8.onnx"),
        tokenizer=str(folder / "tokenizer"),
        num_threads=args.threads,
        provider=args.provider,
        max_new_tokens=args.max_new_tokens,
    )
    audio = np.asarray(pcm, dtype=np.float32)

    def run():
        start = time.perf_counter()
        stream = recognizer.create_stream()
        stream.accept_waveform(16000, audio)
        recognizer.decode_stream(stream)
        return (time.perf_counter() - start) * 1000, stream.result.text

    for _ in range(args.warmups):
        run()
    samples = []
    text = ""
    for _ in range(args.repeats):
        elapsed, text = run()
        samples.append(elapsed)
    text = clean_text(text)
    seconds = len(pcm) / 16000
    mean_ms = float(np.median(samples))
    report = {
        "engine": "sherpa-onnx",
        "provider": args.provider,
        "model": str(folder),
        "audioSeconds": seconds,
        "meanMs": mean_ms,
        "rtf": mean_ms / (seconds * 1000),
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
