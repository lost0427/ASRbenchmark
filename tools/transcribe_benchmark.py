"""Benchmark a SenseVoice GGUF with the transcribe.cpp Python bindings."""
import argparse
import json
import os
import statistics
import time
from pathlib import Path

from windows_benchmark import load_audio, wer

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUDIO = ROOT / "app/src/main/assets/audio/sample.wav"


def select_device(transcribe_cpp, name):
    devices = transcribe_cpp.backends()
    if name == "cpu":
        matches = [device for device in devices if device.device_type == "cpu"]
    elif name == "igpu":
        matches = [device for device in devices if device.device_type == "igpu"]
    elif name == "cuda":
        matches = [device for device in devices
                   if device.device_type == "gpu" and "CUDA" in device.name]
    else:
        matches = [device for device in devices if "Vulkan" in device.name]
    if not matches:
        available = ", ".join(f"{device.name} ({device.device_type})" for device in devices)
        raise SystemExit(f"{name} device not found; available: {available}")
    return matches[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda", "vulkan", "igpu"], default="cpu")
    parser.add_argument("--library", type=Path,
                        help="transcribe.dll to load (use the CPU+Vulkan package for fast CPU)")
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--itn", choices=["default", "off", "on"], default="off")
    parser.add_argument("--timestamps", choices=["none", "auto", "segment", "word", "token"],
                        default="none")
    parser.add_argument("--language")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.warmups < 0 or args.repeats < 1:
        parser.error("warmups must be nonnegative; repeats must be positive")
    if args.library:
        library = args.library.resolve()
        if not library.is_file():
            raise SystemExit(f"transcribe library not found: {library}")
        os.environ["TRANSCRIBE_LIBRARY"] = str(library)
    try:
        import transcribe_cpp
    except ImportError as error:
        raise SystemExit(
            "Install the transcribe.cpp release wheels first; see "
            "docs/WINDOWS_BENCHMARK_RESULTS.md") from error
    if not args.model.is_file():
        raise SystemExit(f"model not found: {args.model}")

    pcm = load_audio(args.audio)
    reference = None
    reference_path = args.reference or args.audio.with_suffix(".json")
    if reference_path.is_file():
        reference = json.loads(reference_path.read_text(encoding="utf-8")).get("reference")
    device = select_device(transcribe_cpp, args.device)
    kwargs = {"timestamps": args.timestamps, "itn": args.itn}
    if args.language:
        kwargs["language"] = args.language

    started = time.perf_counter()
    with transcribe_cpp.Model(str(args.model), device=device) as model:
        load_ms = (time.perf_counter() - started) * 1000
        backend = model.backend
        runtime_device = model.device
        with model.session(n_threads=args.threads) as session:
            for _ in range(args.warmups):
                session.run(pcm, **kwargs)
            samples = []
            result = None
            for _ in range(args.repeats):
                started = time.perf_counter()
                result = session.run(pcm, **kwargs)
                samples.append((time.perf_counter() - started) * 1000)

    seconds = len(pcm) / 16000
    report = {
        "engine": "transcribe.cpp",
        "model": str(args.model),
        "device": args.device,
        "backend": backend,
        "runtimeDevice": runtime_device.name,
        "threads": args.threads,
        "loadMs": load_ms,
        "meanMs": statistics.mean(samples),
        "medianMs": statistics.median(samples),
        "p90Ms": sorted(samples)[min(len(samples) - 1, int(len(samples) * 0.9))],
        "rtf": statistics.mean(samples) / (seconds * 1000),
        "libraryMs": {
            "mel": result.timings.mel_ms,
            "encode": result.timings.encode_ms,
            "decode": result.timings.decode_ms,
        },
        "text": result.text,
    }
    if reference is not None:
        report["wer"] = wer(reference, result.text)
    print(json.dumps({key: value for key, value in report.items() if key != "text"},
                     sort_keys=True), flush=True)
    print(f"TEXT: {result.text}", flush=True)
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
