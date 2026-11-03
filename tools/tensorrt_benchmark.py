"""Build and benchmark a fixed-shape TensorRT engine for SenseVoice."""
import argparse
import json
import os
import statistics
import time
from pathlib import Path

import numpy as np

from windows_benchmark import decode, enable_cuda_dlls, feature_frames, load_audio, wer

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "models/low-precision-models/onnx-fp32"
DEFAULT_AUDIO = ROOT / "app/src/main/assets/audio/sample.wav"
DEFAULT_ENGINE_DIR = ROOT / ".native/trt"


def read_metadata(path):
    try:
        import onnx
    except ImportError as error:
        raise SystemExit("Install tools/tensorrt-requirements.txt first") from error
    model = onnx.load(str(path), load_external_data=False)
    return {item.key: item.value for item in model.metadata_props}


def read_tokens(path):
    return {int(line.rsplit(" ", 1)[1]): line.rsplit(" ", 1)[0]
            for line in path.read_text(encoding="utf-8").splitlines()}


def prepare_dlls(tensorrt_lib):
    if tensorrt_lib:
        path = Path(tensorrt_lib).resolve()
        if not path.is_dir():
            raise SystemExit(f"TensorRT lib directory not found: {path}")
        os.environ["PATH"] = str(path) + os.pathsep + os.environ.get("PATH", "")
        os.add_dll_directory(str(path))
    enable_cuda_dlls()


def build_engine(onnx_path, engine_path, frames, precision, workspace_gb, verbose=False):
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise SystemExit(f"TensorRT could not parse {onnx_path}:\n{errors}")

    names = {network.get_input(i).name for i in range(network.num_inputs)}
    expected = {"x", "x_length", "language", "text_norm"}
    if names != expected:
        raise SystemExit(f"Expected inputs {sorted(expected)}, got {sorted(names)}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1 << 30)))
    if precision == "fp16":
        if not builder.platform_has_fast_fp16:
            raise SystemExit("This GPU does not report fast FP16 support")
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == "bf16":
        if hasattr(builder, "platform_has_fast_bf16") and not builder.platform_has_fast_bf16:
            raise SystemExit("This GPU does not report fast BF16 support")
        config.set_flag(trt.BuilderFlag.BF16)

    profile = builder.create_optimization_profile()
    profile.set_shape("x", (1, frames, 560), (1, frames, 560), (1, frames, 560))
    for name in ("x_length", "language", "text_norm"):
        profile.set_shape(name, (1,), (1,), (1,))
    config.add_optimization_profile(profile)

    started = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise SystemExit("TensorRT engine build failed")
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = engine_path.with_name(engine_path.name + ".tmp")
    temporary.write_bytes(serialized)
    os.replace(temporary, engine_path)
    return {"engine": str(engine_path), "engineBytes": engine_path.stat().st_size,
            "buildSeconds": round(time.perf_counter() - started, 1),
            "tensorrt": trt.__version__, "precision": precision, "frames": frames}


def run_engine(engine_path, pcm, meta, tokens, frames, warmups, repeats):
    from cuda import cudart
    import tensorrt as trt

    success = cudart.cudaError_t.cudaSuccess

    def ok(result):
        error = result[0] if isinstance(result, tuple) else result
        if error != success:
            raise SystemExit(f"CUDA error: {result}")

    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    started = time.perf_counter()
    engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    load_ms = (time.perf_counter() - started) * 1000
    context = engine.create_execution_context()

    features = feature_frames(pcm, meta)
    count = features.shape[1]
    if count > frames:
        features = features[:, :frames]
        count = frames
    elif count < frames:
        features = np.pad(features, ((0, 0), (0, frames - count), (0, 0)))
    features = np.ascontiguousarray(features, dtype=np.float32)

    error, stream = cudart.cudaStreamCreate()
    ok(error)
    host = {
        "x": features,
        "x_length": np.array([count], dtype=np.int32),
        "language": np.array([int(meta["lang_en"])], dtype=np.int32),
        "text_norm": np.array([int(meta["without_itn"])], dtype=np.int32),
    }
    context.set_input_shape("x", (1, frames, 560))
    for name in ("x_length", "language", "text_norm"):
        context.set_input_shape(name, (1,))
    out_shape = tuple(context.get_tensor_shape("logits"))
    devices = {}
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        dtype = trt.nptype(engine.get_tensor_dtype(name))
        shape = tuple(context.get_tensor_shape(name))
        nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
        error, pointer = cudart.cudaMalloc(nbytes)
        ok(error)
        devices[name] = (pointer, nbytes)
        context.set_tensor_address(name, int(pointer))
    output = np.empty(out_shape, dtype=trt.nptype(engine.get_tensor_dtype("logits")))

    def infer():
        for name, value in host.items():
            ok(cudart.cudaMemcpyAsync(devices[name][0], value.ctypes.data, value.nbytes,
                                      cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream))
        context.execute_async_v3(int(stream))
        ok(cudart.cudaMemcpyAsync(output.ctypes.data, devices["logits"][0], output.nbytes,
                                  cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream))
        ok(cudart.cudaStreamSynchronize(stream))

    def fit(values):
        actual = values.shape[1]
        if actual > frames:
            return values[:, :frames], frames
        if actual < frames:
            return np.pad(values, ((0, 0), (0, frames - actual), (0, 0))), actual
        return values, actual

    for _ in range(warmups):
        fitted, _ = fit(feature_frames(pcm, meta))
        host["x"] = np.ascontiguousarray(fitted, dtype=np.float32)
        infer()
    samples = []
    feature_ms = []
    engine_ms = []
    decode_ms = []
    text = ""
    for _ in range(repeats):
        start = time.perf_counter()
        fitted, actual = fit(feature_frames(pcm, meta))
        middle = time.perf_counter()
        host["x"] = np.ascontiguousarray(fitted, dtype=np.float32)
        infer()
        end = time.perf_counter()
        text = decode(output, tokens, actual)
        finished = time.perf_counter()
        feature_ms.append((middle - start) * 1000)
        engine_ms.append((end - middle) * 1000)
        decode_ms.append((finished - end) * 1000)
        samples.append((finished - start) * 1000)

    return {
        "engine": str(engine_path),
        "loadMs": load_ms,
        "frames": frames,
        "featureMs": statistics.mean(feature_ms),
        "engineMs": statistics.mean(engine_ms),
        "decodeMs": statistics.mean(decode_ms),
        "meanMs": statistics.mean(samples),
        "medianMs": statistics.median(samples),
        "p90Ms": float(np.percentile(samples, 90)),
        "text": text,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, default=DEFAULT_MODEL / "model.onnx")
    parser.add_argument("--tokens", type=Path, default=DEFAULT_MODEL / "tokens.txt")
    parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--engine", type=Path)
    parser.add_argument("--frames", type=int, default=0)
    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--workspace-gb", type=float, default=4.0)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--tensorrt-lib", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if args.warmups < 0 or args.repeats < 1 or args.workspace_gb <= 0:
        parser.error("warmups must be nonnegative; repeats and workspace-gb must be positive")
    if not args.onnx.is_file():
        raise SystemExit("ONNX model not found; run python tools/prepare_low_precision.py --fp32")
    if not args.tokens.is_file():
        raise SystemExit(f"tokens.txt not found: {args.tokens}")

    prepare_dlls(args.tensorrt_lib)
    meta = read_metadata(args.onnx)
    pcm = load_audio(args.audio)
    features = feature_frames(pcm, meta)
    frames = args.frames or features.shape[1]
    if frames < 1:
        raise SystemExit("frames must be positive")
    engine_path = args.engine or (DEFAULT_ENGINE_DIR /
                                  f"sensevoice-{args.precision}-{frames}.engine")
    build = None
    if args.build or not engine_path.is_file():
        build = build_engine(args.onnx, engine_path, frames, args.precision,
                             args.workspace_gb, args.verbose)
        print(json.dumps(build, sort_keys=True), flush=True)
    if args.build_only:
        result = {"build": build, "precision": args.precision, "frames": frames}
        if args.output:
            args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return

    tokens = read_tokens(args.tokens)
    reference = None
    reference_path = args.reference or args.audio.with_suffix(".json")
    if reference_path.is_file():
        reference = json.loads(reference_path.read_text(encoding="utf-8")).get("reference")
    result = run_engine(engine_path, pcm, meta, tokens, frames,
                        args.warmups, args.repeats)
    result["build"] = build
    result["precision"] = args.precision
    result["provider"] = "tensorrt"
    if reference is not None:
        result["wer"] = wer(reference, result["text"])
        result["rtf"] = result["meanMs"] / ((len(pcm) / 16000) * 1000)
    print(json.dumps({key: value for key, value in result.items() if key != "text"},
                     sort_keys=True), flush=True)
    print(f"TEXT: {result['text']}", flush=True)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
