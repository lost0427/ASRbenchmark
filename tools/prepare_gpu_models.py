"""Prepare GPU-ready models from the existing low-precision bundle.

Creates two extra folders in the same SAF-importable bundle:

  mnn-fp32     FP32 MNN model converted from onnx-fp32 (no quantization)
  litert-fp32  FP32 LiteRT model dequantized from the dynamic-range W8 model

The LiteRT conversion is a lossless inverse of W8 weight quantization: the graph,
signatures and activations stay identical, so GPU delegate coverage is directly
comparable with the original CPU q8 row.
"""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

from build_native import fetch

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / ".native" / "lowprecision-source"
LITERT_REV = "8d1dfd033bafc5d40e03cb1dc714134386eb6895"


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def copy_file(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def record(root, manifest, folder, source, quantization, extra):
    path = root / folder
    (path / "provenance.json").write_text(json.dumps(extra, indent=2) + "\n", encoding="utf-8")
    files = sorted(item for item in path.iterdir() if item.is_file())
    manifest.setdefault("engines", {})[folder] = {
        "source": source,
        "quantization": quantization,
        "files": [
            {"path": f"{folder}/{item.name}", "sizeBytes": item.stat().st_size, "sha256": sha(item)}
            for item in files
        ],
    }


def schema_modules():
    if not (WORK / "schema-python/tflite/Model.py").exists():
        fetch("https://raw.githubusercontent.com/google-ai-edge/LiteRT/v2.1.6/tflite/converter/schema/schema.fbs", WORK / "schema.fbs")
        compiler = shutil.which("flatc")
        if compiler is None and sys.platform == "win32":
            archive = WORK / "flatc.zip"
            fetch("https://github.com/google/flatbuffers/releases/download/v25.12.19-2026-02-06-03fffb2/Windows.flatc.binary.zip", archive)
            with zipfile.ZipFile(archive) as zip:
                zip.extractall(WORK / "flatc")
            compiler = str(WORK / "flatc/flatc.exe")
        if compiler is None:
            raise RuntimeError("Install flatc to generate the pinned LiteRT schema")
        subprocess.run([compiler, "--python", "--gen-object-api", "-o", str(WORK / "schema-python"), str(WORK / "schema.fbs")], check=True)
    sys.path.insert(0, str(WORK / "schema-python"))
    return importlib.import_module("tflite.Model"), importlib.import_module("tflite.TensorType").TensorType


def prepare_litert_fp32(baseline, root, manifest):
    import numpy as np

    schema, tensor_types = schema_modules()
    buffer_module = importlib.import_module("tflite.Buffer")
    original = baseline / "litert/sensevoice_small_q8.tflite"
    if not original.is_file():
        raise FileNotFoundError(f"Missing {original}")
    data = original.read_bytes()
    model = schema.ModelT.InitFromObj(schema.Model.GetRootAsModel(data, 0))
    converted = 0
    for graph in model.subgraphs:
        for tensor in graph.tensors:
            if tensor.type != tensor_types.INT8:
                continue
            buffer = model.buffers[tensor.buffer]
            raw = bytes(buffer.data) if buffer.data is not None else b""
            if not raw:
                raise RuntimeError("Found a non-constant INT8 tensor; refusing to dequantize")
            q = tensor.quantization
            weights = np.frombuffer(raw, dtype=np.int8)
            scales = np.asarray(q.scale, dtype=np.float32)
            zero_points = q.zeroPoint
            if zero_points is None or len(zero_points) == 0:
                zero_points = [0] * len(scales)
            zeros = np.asarray(zero_points, dtype=np.float32)
            if len(scales) == 1:
                scale = np.full(weights.size, scales[0], dtype=np.float32)
                zero = np.full(weights.size, zeros[0], dtype=np.float32)
            else:
                shape = [int(dim) for dim in tensor.shape]
                axis = int(q.quantizedDimension)
                assert 0 <= axis < len(shape), "Invalid quantization axis"
                assert shape[axis] == len(scales), "Scale count does not match quantized dimension"
                inner = int(np.prod(shape[axis + 1:])) if axis + 1 < len(shape) else 1
                outer = int(np.prod(shape[:axis])) if axis else 1
                scale = np.tile(np.repeat(scales, inner), outer)
                zero = np.tile(np.repeat(zeros, inner), outer)
            assert scale.size == weights.size, "Quantization pattern does not match tensor size"
            real = (weights.astype(np.float32) - zero) * scale
            new_buffer = buffer_module.BufferT()
            new_buffer.data = real.astype(np.float32).tobytes()
            model.buffers.append(new_buffer)
            tensor.buffer = len(model.buffers) - 1
            tensor.type = tensor_types.FLOAT32
            tensor.quantization = None
            converted += 1
    model.metadata = []
    import flatbuffers
    builder = flatbuffers.Builder(max(len(data) * 4, 1 << 20))
    builder.Finish(model.Pack(builder), file_identifier=b"TFL3")
    folder = "litert-fp32"
    path = root / folder
    path.mkdir(parents=True, exist_ok=True)
    (path / "model.tflite").write_bytes(builder.Output())
    for name in ["tokens.txt", "cmvn.json", "LICENSE.txt"]:
        copy_file(baseline / "litert" / name, path / name)
    record(root, manifest, folder,
           f"https://huggingface.co/Luigi/sensevoice-litert/tree/{LITERT_REV}",
           "FP32 weights dequantized from dynamic-range W8; FP32 activations; lossless inverse of weight quantization", {
               "sourceSha256": sha(original), "converter": "tools/prepare_gpu_models.py",
               "schema": "LiteRT v2.1.6", "dequantizedWeightTensors": converted,
               "method": "W = (int8 - zero_point) * per-channel scale, per output channel",
               "limitation": "Weights carry the original W8 quantization error; this is not the FP32 checkpoint.",
           })
    print(f"litert-fp32: {converted} weight tensors dequantized")


def prepare_mnn_fp32(root, manifest):
    source = WORK / "model.onnx"
    if not source.is_file():
        raise FileNotFoundError(
            f"Missing {source}. Run python tools/prepare_low_precision.py --fp32 first.")
    folder = "mnn-fp32"
    path = root / folder
    path.mkdir(parents=True, exist_ok=True)
    model = path / "model.mnn"
    if not model.exists():
        args = ["MNNConvert", "-f", "ONNX", "--modelFile", str(source),
                "--MNNModel", str(model), "--bizCode", "MNN"]
        subprocess.run([sys.executable, "-c", "import _tools,sys; _tools.mnnconvert(sys.argv[1:])", *args], check=True)
    if not model.is_file() or model.stat().st_size < 1_000_000:
        raise RuntimeError("MNN FP32 conversion failed")
    baseline = root / "mnn"
    for name in ["tokens.txt", "LICENSE.txt"]:
        copy_file(baseline / name, path / name)
    record(root, manifest, folder,
           "https://huggingface.co/csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17",
           "FP32 weights, no quantization, MNN 3.6.1", {
               "sourceSha256": sha(source), "converter": "MNN 3.6.1",
               "arguments": {"weightQuantBits": 0},
               "activationPrecision": "floating",
           })
    print(f"mnn-fp32: {model.stat().st_size / 1e6:.1f} MB")


def prepare_gpu(root, baseline, force=False):
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {"schemaVersion": 2, "engines": {}}
    if force or not (root / "litert-fp32/model.tflite").exists():
        prepare_litert_fp32(baseline, root, manifest)
    if force or not (root / "mnn-fp32/model.mnn").exists():
        prepare_mnn_fp32(root, manifest)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("GPU-ready bundle:", root.resolve())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "models/low-precision-models")
    parser.add_argument("--baseline", type=Path, default=ROOT / "models/cpu-models")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    prepare_gpu(args.root.resolve(), args.baseline.resolve(), args.force)


if __name__ == "__main__":
    main()
