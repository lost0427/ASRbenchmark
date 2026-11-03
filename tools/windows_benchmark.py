"""Run the local SenseVoice models with Windows Python runtime wheels."""
import argparse
import gc
import json
import os
import platform
from pathlib import Path
import re
import subprocess
import time
import wave

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS = ROOT / "models/low-precision-models"
DEFAULT_AUDIO = ROOT / "app/src/main/assets/audio/sample.wav"
DEFAULT_GGUF = ROOT / ".native/gguf-host-build/bin" / (
    "benchmark_gguf_smoke.exe" if os.name == "nt" else "benchmark_gguf_smoke")
DEFAULT_GGUF_CUDA = ROOT / ".native/gguf-host-cuda-build/bin" / (
    "benchmark_gguf_smoke.exe" if os.name == "nt" else "benchmark_gguf_smoke")
ENGINES = ["sherpa", "ort", "mnn", "ncnn", "litert", "gguf"]
GPU_ENGINES = {"ort", "gguf"}


def normalize(text):
    return re.sub(r"\s+", " ", re.sub(r"[^A-Z0-9 ]", " ", text.upper())).strip()


def wer(reference, hypothesis):
    left, right = normalize(reference).split(), normalize(hypothesis).split()
    row = list(range(len(right) + 1))
    for i, token in enumerate(left):
        current = [i + 1]
        for j, other in enumerate(right):
            current.append(min(current[-1] + 1, row[j + 1] + 1,
                               row[j] + (token != other)))
        row = current
    return row[-1] / max(1, len(left))


def load_audio(path):
    with wave.open(str(path), "rb") as wav:
        if wav.getnchannels() != 1 or wav.getsampwidth() != 2 or wav.getframerate() != 16000:
            raise ValueError("Audio must be mono 16 kHz PCM16 WAV")
        pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.float32)
    return pcm / 32768.0


def feature_frames(pcm, meta):
    import kaldi_native_fbank as knf
    opts = knf.FbankOptions()
    opts.frame_opts.samp_freq = 16000
    opts.frame_opts.frame_length_ms = 25
    opts.frame_opts.frame_shift_ms = 10
    opts.frame_opts.dither = 0
    opts.frame_opts.snip_edges = True
    opts.frame_opts.window_type = "hamming"
    opts.mel_opts.num_bins = 80
    opts.mel_opts.low_freq = 20
    opts.mel_opts.high_freq = 0
    bank = knf.OnlineFbank(opts)
    bank.accept_waveform(16000, (pcm * 32768.0).tolist())
    bank.input_finished()
    values = np.stack([bank.get_frame(i) for i in range(bank.num_frames_ready)])
    count = (len(values) + 5) // 6
    indices = np.clip(np.arange(count)[:, None] * 6 + np.arange(7)[None, :] - 3,
                      0, len(values) - 1)
    frames = values[indices].reshape(count, 560)
    shift = np.asarray([float(x) for x in meta["neg_mean"].split(",")], dtype=np.float32)
    scale = np.asarray([float(x) for x in meta["inv_stddev"].split(",")], dtype=np.float32)
    return ((frames + shift) * scale)[None].astype(np.float32)


def decode(logits, tokens, frame_count):
    previous, result = -1, []
    for best in logits[0, 4:frame_count + 4].argmax(axis=-1):
        best = int(best)
        if best and best != previous:
            token = tokens.get(best)
            if token is None:
                raise ValueError(f"Missing token {best}")
            if not token.startswith("<|"):
                result.append(token)
        previous = best
    return "".join(result).replace("▁", " ").strip()


def tokens_for(folder):
    return {int(line.rsplit(" ", 1)[1]): line.rsplit(" ", 1)[0]
            for line in (folder / "tokens.txt").read_text(encoding="utf-8").splitlines()}


def source_metadata(models):
    import onnxruntime as ort
    source = models / "onnx/model.int8.onnx"
    session = ort.InferenceSession(str(source), providers=["CPUExecutionProvider"])
    return session.get_modelmeta().custom_metadata_map


def enable_cuda_dlls():
    import site
    directories = []
    for root in (Path(path) for path in site.getsitepackages()):
        directories += [root / "nvidia/cudnn/bin", root / "nvidia/cublas/bin",
                        root / "nvidia/cufft/bin", root / "nvidia/cuda_runtime/bin"]
    if os.environ.get("CUDA_PATH"):
        directories.append(Path(os.environ["CUDA_PATH"]) / "bin")
    directories = [path for path in directories if path.is_dir()]
    os.environ["PATH"] = os.pathsep.join([str(path) for path in directories] + [os.environ["PATH"]])
    for path in directories:
        os.add_dll_directory(str(path))


def make_runner(engine, folder, threads, provider):
    if engine in ("ort", "sherpa"):
        model = folder / ("model.int8.onnx" if folder.name == "onnx" else "model.onnx")
        if engine == "sherpa":
            if provider != "cpu":
                print("note: sherpa-onnx wheels are CPU-only", flush=True)
            import sherpa_onnx
            if folder.name != "onnx":
                raise ValueError("sherpa-onnx currently runs the local Q8 model only")
            recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
                model=str(model), tokens=str(folder / "tokens.txt"),
                use_itn=False, num_threads=threads, debug=False)
            def run(pcm):
                stream = recognizer.create_stream()
                stream.accept_waveform(16000, pcm)
                recognizer.decode_stream(stream)
                return stream.result.text
            return run, "cpu"
        if provider == "cuda":
            enable_cuda_dlls()
        import onnxruntime as ort
        providers = ["CPUExecutionProvider"]
        if provider == "cuda":
            if "CUDAExecutionProvider" not in ort.get_available_providers():
                raise RuntimeError(
                    "CUDAExecutionProvider is unavailable; install "
                    "tools/windows-benchmark-gpu-requirements.txt and cuDNN 9")
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        options = ort.SessionOptions()
        options.intra_op_num_threads = 0 if provider == "cuda" else threads
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(str(model), options, providers=providers)
        actual = "cuda" if "CUDAExecutionProvider" in session.get_providers() else "cpu"
        if provider == "cuda" and actual != "cuda":
            raise RuntimeError(
                "CUDAExecutionProvider failed to initialize; check the CUDA 12 / cuDNN 9 "
                "runtime and PATH")
        metadata = session.get_modelmeta().custom_metadata_map
        tokens = tokens_for(folder)
        def run(pcm):
            features = feature_frames(pcm, metadata)
            count = features.shape[1]
            inputs = {"x": features, "x_length": np.array([count], dtype=np.int32),
                      "language": np.array([int(metadata["lang_en"])], dtype=np.int32),
                      "text_norm": np.array([int(metadata["without_itn"])], dtype=np.int32)}
            return decode(session.run(None, inputs)[0], tokens, count)
        return run, actual
    if engine == "mnn":
        import MNN.expr as F
        from MNN import nn
        meta = source_metadata(folder.parent)
        tokens = tokens_for(folder)
        module = nn.load_module_from_file(str(folder / "model.mnn"),
                    ["x", "x_length", "language", "text_norm"], ["logits"],
                    thread_num=threads, memory_mode=F.MemoryMode.Low)
        def run(pcm):
            features = feature_frames(pcm, meta)
            count = features.shape[1]
            values = [features, np.array([count], dtype=np.int32),
                      np.array([int(meta["lang_en"])], dtype=np.int32),
                      np.array([int(meta["without_itn"])], dtype=np.int32)]
            tensors = [F.const(v.flatten().tolist(), list(v.shape), F.NCHW,
                               F.float if v.dtype == np.float32 else F.int) for v in values]
            return decode(module.onForward(tensors)[0].read(), tokens, count)
        return run, "cpu"
    if engine == "litert":
        from ai_edge_litert.interpreter import Interpreter
        model = folder / ("sensevoice_small_q8.tflite" if folder.name == "litert" else "model.tflite")
        interpreter = Interpreter(model_path=str(model), num_threads=threads)
        meta = source_metadata(folder.parent)
        tokens = tokens_for(folder)
        runner = interpreter.get_signature_runner("sv_500")
        def run(pcm):
            features = feature_frames(pcm, meta)
            count = features.shape[1]
            padded = np.zeros((1, 500, 560), dtype=np.float32)
            padded[:, :count] = features
            output = next(iter(runner(args_0=padded, args_1=np.array([count], np.int32),
                                      args_2=np.array([int(meta["lang_en"])], np.int32),
                                      args_3=np.array([int(meta["without_itn"])], np.int32)).values()))
            return decode(output, tokens, count)
        return run, "cpu"
    if engine == "ncnn":
        try:
            import sherpa_ncnn
        except ImportError as error:
            raise RuntimeError(
                "Install sherpa-ncnn==2.1.15 from tools/windows-benchmark-requirements.txt") from error
        config = sherpa_ncnn.OfflineRecognizerConfig()
        config.model_config.sense_voice.model_dir = str(folder)
        config.model_config.sense_voice.language = "en"
        config.model_config.sense_voice.use_itn = False
        config.model_config.tokens = str(folder / "tokens.txt")
        config.model_config.num_threads = threads
        recognizer = sherpa_ncnn.OfflineRecognizer(config)
        def run(pcm):
            stream = recognizer.create_stream()
            stream.accept_waveform(16000, pcm)
            recognizer.decode_stream(stream)
            return stream.result.text
        return run, "cpu"
    raise ValueError(f"Unsupported engine: {engine}")


def summarize(engine, model, provider, load_ms, first_ms, samples, text, reference, seconds):
    return {"engine": engine, "model": model, "provider": provider, "loadMs": load_ms,
            "firstInferenceMs": first_ms, "meanMs": float(np.mean(samples)),
            "medianMs": float(np.median(samples)), "p90Ms": float(np.percentile(samples, 90)),
            "rtf": float(np.mean(samples) / seconds),
            "wer": wer(reference, text), "text": text, "samplesMs": samples}


def run_inprocess(engine, folder, pcm, args, provider, reference):
    print(f"Loading {engine} / {folder.name} [{provider}]...", flush=True)
    start = time.perf_counter()
    run, actual = make_runner(engine, folder, args.threads, provider)
    load_ms = (time.perf_counter() - start) * 1000
    start = time.perf_counter()
    text = run(pcm)
    first_ms = (time.perf_counter() - start) * 1000
    for _ in range(args.warmups):
        run(pcm)
    samples = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        text = run(pcm)
        samples.append((time.perf_counter() - start) * 1000)
    del run
    gc.collect()
    return summarize(engine, folder.name, actual, load_ms, first_ms, samples, text,
                     reference, len(pcm) / 16000)


def run_gguf(exe, folder, args, reference, seconds, gpu):
    model = next(folder.glob("*.gguf"))
    provider = "cuda" if gpu else "cpu"
    print(f"Running gguf / {folder.name} [{provider}]...", flush=True)
    command = [str(exe), str(model), str(args.audio), str(args.threads),
               str(args.warmups), str(args.repeats), "gpu" if gpu else "cpu"]
    if args.gguf_flash:
        command.append("fa")
    finished = subprocess.run(command, capture_output=True, text=True,
                              encoding="utf-8", errors="replace")
    if finished.returncode:
        raise RuntimeError(f"GGUF runner failed for {folder.name}: {finished.stderr.strip()}")
    if gpu and "using CUDA" not in finished.stderr:
        raise RuntimeError(f"GGUF runner did not use a CUDA backend for {folder.name}")
    line = next(line for line in reversed(finished.stdout.splitlines())
                if line.startswith("RESULT: "))
    result = json.loads(line[len("RESULT: "):])
    return summarize("gguf", folder.name, provider, result["loadMs"], result["firstInferenceMs"],
                     result["samplesMs"], result["text"], reference, seconds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, default=DEFAULT_MODELS)
    parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO)
    parser.add_argument("--engines", nargs="+", choices=ENGINES)
    parser.add_argument("--folders", nargs="+", help="Model folders from manifest.json")
    parser.add_argument("--provider", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--gguf-exe", type=Path, default=DEFAULT_GGUF)
    parser.add_argument("--gguf-cuda-exe", type=Path, default=DEFAULT_GGUF_CUDA)
    parser.add_argument("--gguf-flash", action="store_true",
                        help="Enable GGUF flash attention")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("windows-benchmark.json"))
    args = parser.parse_args()
    if args.threads < 1 or args.warmups < 0 or args.repeats < 1:
        parser.error("threads and repeats must be positive; warmups must be nonnegative")

    engines = args.engines or ENGINES
    if "gguf" in engines:
        gguf_exe = args.gguf_cuda_exe if args.provider == "cuda" else args.gguf_exe
        if not gguf_exe.is_file():
            hint = f"build it with python tools/build_gguf.py --host" + (" --cuda" if args.provider == "cuda" else "")
            if args.engines:
                parser.error(f"GGUF runner not found: {gguf_exe}; {hint}")
            print(f"Skipping gguf: {gguf_exe} not found ({hint})", flush=True)
            engines = [engine for engine in engines if engine != "gguf"]

    reference = json.loads(args.audio.with_suffix(".json").read_text(encoding="utf-8"))["reference"]
    pcm = load_audio(args.audio)
    manifest = json.loads((args.models / "manifest.json").read_text(encoding="utf-8"))
    entries = manifest["engines"]
    selected = args.folders or list(entries)
    report = {"platform": platform.platform(), "processor": platform.processor(),
              "python": platform.python_version(), "backends": "CPU",
              "requestedProvider": args.provider,
              "audio": str(args.audio),
              "audioSeconds": len(pcm) / 16000, "threads": args.threads,
              "warmups": args.warmups, "repeats": args.repeats, "results": []}
    for folder_name in selected:
        if folder_name not in entries:
            raise ValueError(f"Unknown model folder: {folder_name}")
        if folder_name.startswith("onnx"):
            applicable = [e for e in engines
                          if e == "ort" or (e == "sherpa" and folder_name == "onnx")]
        elif folder_name.startswith("mnn"):
            applicable = [e for e in engines if e == "mnn"]
        elif folder_name.startswith("ncnn"):
            applicable = [e for e in engines if e == "ncnn"]
        elif folder_name.startswith("litert"):
            applicable = [e for e in engines if e == "litert"]
        elif folder_name.startswith("gguf"):
            applicable = [e for e in engines if e == "gguf"]
        else:
            continue
        for engine in applicable:
            provider = "cuda" if args.provider == "cuda" and engine in GPU_ENGINES else "cpu"
            if args.provider == "cuda" and provider == "cpu":
                print(f"note: {engine} has no GPU runtime, running CPU", flush=True)
            if engine == "gguf":
                gpu = provider == "cuda"
                exe = args.gguf_cuda_exe if gpu else args.gguf_exe
                result = run_gguf(exe, args.models / folder_name, args,
                                  reference, len(pcm) / 16000, gpu)
            else:
                result = run_inprocess(engine, args.models / folder_name, pcm, args,
                                       provider, reference)
            report["results"].append(result)
            report["backends"] = ("CUDA+CPU" if any(item["provider"] == "cuda"
                                  for item in report["results"]) else "CPU")
            args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                                   encoding="utf-8")
            print(f"{engine}/{folder_name} [{result['provider']}]: "
                  f"{result['meanMs']:.1f} ms, WER {result['wer']:.2%}", flush=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    print(f"Saved {args.output.resolve()}")


if __name__ == "__main__":
    main()
