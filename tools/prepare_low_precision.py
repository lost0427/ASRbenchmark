"""Prepare auditable multi-precision SenseVoice models for the Android benchmark.

Use an isolated environment with tools/model-requirements.txt. Original Q8 models
must first be prepared with prepare_models.py. Weights never belong in Git.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import zipfile

import requests

from build_native import fetch
from prepare_models import ONNX, ONNX_REV, sha

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / ".native" / "lowprecision-source"
SOURCE_HASH = "977016bd9c79f9eb343430b5cc305e07ab64d5212dff41b0dcfa1694bee9a8cb"
GGUF_REPO = "lovemefan/sense-voice-gguf"
GGUF_REV = "c3dc3564b1795b03f4c2cfa8549af5797faca4ad"
GGUF_VARIANTS = ["q8_0", "q6_k", "q5_0", "q5_k", "q4_0", "q4_1", "q4_k", "q3_k", "fp16"]


def copy_file(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        shutil.copy2(source, destination)
    if sha(source) != sha(destination):
        raise ValueError(f"Existing file differs from source: {destination}")


def record(root, manifest, folder, source, quantization, provenance):
    path = root / folder
    (path / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    files = [{"path": f"{folder}/{file.name}", "sizeBytes": file.stat().st_size, "sha256": sha(file)}
             for file in sorted(path.iterdir()) if file.is_file() and file.suffix != ".part"]
    manifest["engines"][folder] = {"source": source, "quantization": quantization, "files": files}
    # Save complete entries as each conversion finishes, so interrupted jobs resume.
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Ready: {folder} ({sum(f['sizeBytes'] for f in files) / 1e6:.1f} MB)", flush=True)


def supporting_files(baseline, root, engine, folder):
    for name in ["tokens.txt", "LICENSE.txt"]:
        copy_file(baseline / engine / name, root / folder / name)


def prepare_onnx(source, baseline, root, manifest, bits):
    import onnx
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer

    folder = f"onnx-q{bits}"
    path = root / folder
    path.mkdir(parents=True, exist_ok=True)
    model_path = path / "model.onnx"
    if not model_path.exists():
        model = onnx.load(str(source))
        quantizer = MatMulNBitsQuantizer(model, bits=bits, block_size=64, is_symmetric=True)
        quantizer.process()
        quantizer.model.save_model_to_file(str(model_path), use_external_data_format=False)
    model = onnx.load(str(model_path))
    nodes = [n for n in model.graph.node if n.op_type == "MatMulNBits"]
    assert nodes, "No MatMul weights were quantized"
    assert all(next(a.i for a in n.attribute if a.name == "bits") == bits for n in nodes)
    assert {x.key: x.value for x in model.metadata_props}["model_type"] == "sense_voice_ctc"
    onnx.checker.check_model(str(model_path))
    supporting_files(baseline, root, "onnx", folder)
    record(root, manifest, folder, f"https://huggingface.co/{ONNX}/tree/{ONNX_REV}",
           f"W{bits} MatMulNBits, block 64; other operators retain source precision", {
               "sourceSha256": SOURCE_HASH, "converter": "onnxruntime 1.30.0 MatMulNBitsQuantizer",
               "bits": bits, "blockSize": 64, "symmetric": True, "quantizedMatMulCount": len(nodes),
               "activationPrecision": "floating", "calibration": "none; weight-only quantization",
           })


def prepare_mnn(source, baseline, root, manifest, bits):
    folder = f"mnn-q{bits}"
    path = root / folder
    path.mkdir(parents=True, exist_ok=True)
    model = path / "model.mnn"
    if not model.exists():
        # Call the converter extension directly; its CLI wrapper sends telemetry.
        args = ["MNNConvert", "-f", "ONNX", "--modelFile", str(source), "--MNNModel", str(model),
                "--bizCode", "MNN", "--weightQuantBits", str(bits), "--weightQuantBlock", "64"]
        subprocess.run([sys.executable, "-c", "import _tools,sys; _tools.mnnconvert(sys.argv[1:])", *args], check=True)
    assert model.is_file() and model.stat().st_size > 1_000_000
    supporting_files(baseline, root, "mnn", folder)
    record(root, manifest, folder, f"https://huggingface.co/{ONNX}/tree/{ONNX_REV}",
           f"Q{bits} weight-only, block 64, MNN 3.6.1", {
               "sourceSha256": SOURCE_HASH, "converter": "MNN 3.6.1",
               "arguments": {"weightQuantBits": bits, "weightQuantBlock": 64, "bizCode": "MNN"},
               "activationPrecision": "floating", "calibration": "none; weight-only quantization",
               "note": "Requested weight bit width does not imply native CPU arithmetic at that width.",
           })


def prepare_litert(baseline, root, manifest, bits):
    import flatbuffers
    import numpy as np

    if not (WORK / "schema-python/tflite/Model.py").exists():
        fetch("https://raw.githubusercontent.com/google-ai-edge/LiteRT/v2.1.6/tflite/converter/schema/schema.fbs", WORK / "schema.fbs")
        compiler = shutil.which("flatc")
        if compiler is None and sys.platform == "win32":
            archive = WORK / "flatc.zip"
            fetch("https://github.com/google/flatbuffers/releases/download/v25.12.19-2026-02-06-03fffb2/Windows.flatc.binary.zip", archive)
            with zipfile.ZipFile(archive) as zip: zip.extractall(WORK / "flatc")
            compiler = str(WORK / "flatc/flatc.exe")
        if compiler is None: raise RuntimeError("Install flatc to generate the pinned LiteRT schema")
        subprocess.run([compiler, "--python", "--gen-object-api", "-o", str(WORK / "schema-python"), str(WORK / "schema.fbs")], check=True)
    sys.path.insert(0, str(WORK / "schema-python"))
    schema = importlib.import_module("tflite.Model")
    tensor_types = importlib.import_module("tflite.TensorType").TensorType
    original = baseline / "litert/sensevoice_small_q8.tflite"
    data = original.read_bytes()
    model = schema.ModelT.InitFromObj(schema.Model.GetRootAsModel(data, 0))
    # Requantize constant fully-connected weights only. Depthwise weights stay W8.
    candidates = {}
    consumers = {}
    for graph in model.subgraphs:
        for op in graph.operators:
            builtin = model.operatorCodes[op.opcodeIndex].builtinCode
            for index in op.inputs:
                if index >= 0:
                    tensor = graph.tensors[index]
                    consumers.setdefault(tensor.buffer, set()).add(builtin)
            if builtin == 9:  # FULLY_CONNECTED
                tensor = graph.tensors[op.inputs[1]]
                if tensor.type == tensor_types.INT8 and len(tensor.shape) == 2:
                    candidates.setdefault(tensor.buffer, tensor)
    chosen = {buffer: tensor for buffer, tensor in candidates.items() if consumers[buffer] == {9}}
    assert chosen, "No exclusively fully-connected INT8 weights found"
    updated = {}
    minimum, maximum = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    for buffer, tensor in chosen.items():
        q = tensor.quantization
        assert q.quantizedDimension == 0 and np.all(q.zeroPoint == 0)
        weights = np.frombuffer(model.buffers[buffer].data, dtype=np.int8).reshape(tensor.shape)
        scales = np.asarray(q.scale, dtype=np.float32).reshape(-1, 1)
        real = weights.astype(np.float32) * scales
        new_scales = np.maximum(real.max(axis=1) / maximum, real.min(axis=1) / minimum)
        new_scales = np.maximum(new_scales, np.finfo(np.float32).tiny)
        codes = np.clip(np.rint(real / new_scales[:, None]), minimum, maximum).astype(np.int8)
        flat = codes.reshape(-1).astype(np.uint8) & ((1 << bits) - 1)
        per_byte = 8 // bits
        if flat.size % per_byte:
            flat = np.pad(flat, (0, per_byte - flat.size % per_byte))
        packed = np.zeros(flat.size // per_byte, dtype=np.uint8)
        for offset in range(per_byte):
            packed |= flat[offset::per_byte] << (offset * bits)
        model.buffers[buffer].data = packed
        updated[buffer] = new_scales
    for graph in model.subgraphs:
        for tensor in graph.tensors:
            if tensor.buffer in updated:
                assert tensor.type == tensor_types.INT8
                tensor.type = tensor_types.INT4 if bits == 4 else tensor_types.INT2
                tensor.quantization.scale = updated[tensor.buffer]
                tensor.quantization.zeroPoint = np.zeros(len(updated[tensor.buffer]), dtype=np.int64)
    for graph in model.subgraphs:
        for op in graph.operators:
            if model.operatorCodes[op.opcodeIndex].builtinCode == 9:
                tensor = graph.tensors[op.inputs[1]]
                if tensor.buffer in updated:
                    model.operatorCodes[op.opcodeIndex].version = max(model.operatorCodes[op.opcodeIndex].version, 12)
    # Original conversion metadata no longer describes the transformed weights.
    model.metadata = []
    builder = flatbuffers.Builder(len(data))
    builder.Finish(model.Pack(builder), file_identifier=b"TFL3")
    folder = f"litert-q{bits}"
    path = root / folder
    path.mkdir(parents=True, exist_ok=True)
    (path / "model.tflite").write_bytes(builder.Output())
    supporting_files(baseline, root, "litert", folder)
    copy_file(baseline / "litert/cmvn.json", path / "cmvn.json")
    record(root, manifest, folder, "https://huggingface.co/Luigi/sensevoice-litert/tree/8d1dfd033bafc5d40e03cb1dc714134386eb6895",
           f"W{bits} fully-connected weights; W8 depthwise; FP32 activations; requantized from W8", {
               "sourceSha256": sha(original), "converter": "tools/prepare_low_precision.py",
               "schema": "LiteRT v2.1.6", "bits": bits, "quantizedWeightBuffers": len(updated),
               "method": "Dequantize original W8 weights, per-channel zero-point-0 requantization, signed packed weights",
               "limitation": "Not quantized from the original floating checkpoint; extra quantization error is possible.",
           })


def prepare_ncnn(baseline, root, manifest):
    url = "https://github.com/k2-fsa/sherpa-ncnn/releases/download/asr-models/sherpa-ncnn-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2"
    archive = WORK / "ncnn-fp16.tar.bz2"
    fetch(url, archive)
    folder = "ncnn-fp16"
    path = root / folder
    path.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:bz2") as tar:
        for name in ["model.ncnn.param", "model.ncnn.bin"]:
            matches = [m for m in tar.getmembers() if m.isfile() and Path(m.name).name == name]
            assert len(matches) == 1
            with tar.extractfile(matches[0]) as src, (path / name).open("wb") as dst:
                shutil.copyfileobj(src, dst)
    supporting_files(baseline, root, "ncnn", folder)
    record(root, manifest, folder, url, "FP16 weight storage; CPU, no Vulkan", {
        "archiveSha256": sha(archive), "export": "official sherpa-ncnn SenseVoice 2024-07-17 FP16 export",
        "note": "Storage precision does not establish arithmetic precision.",
    })


def prepare_gguf(baseline, root, manifest):
    import gguf

    response = requests.get(f"https://huggingface.co/api/models/{GGUF_REPO}/revision/{GGUF_REV}", params={"blobs": "true"}, timeout=45)
    response.raise_for_status()
    entries = {item["rfilename"]: item for item in response.json()["siblings"]}
    def download(variant):
        name = f"sense-voice-small-{variant}.gguf"
        fetch(f"https://huggingface.co/{GGUF_REPO}/resolve/{GGUF_REV}/{name}", root / f"gguf-{variant}/model.gguf")
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(download, GGUF_VARIANTS))
    variants = list(GGUF_VARIANTS)
    for variant in variants:
        folder = f"gguf-{variant}"
        path = root / folder
        path.mkdir(parents=True, exist_ok=True)
        model = path / "model.gguf"
        name = f"sense-voice-small-{variant}.gguf"
        fetch(f"https://huggingface.co/{GGUF_REPO}/resolve/{GGUF_REV}/{name}", model)
        assert sha(model) == entries[name]["lfs"]["sha256"], f"GGUF checksum mismatch: {variant}"
        provenance = {"repository": GGUF_REPO, "revision": GGUF_REV, "upstreamFile": name}
        reader = gguf.GGUFReader(str(model))
        types = dict(Counter(t.tensor_type.name for t in reader.tensors))
        provenance["actualTensorTypes"] = types
        provenance["note"] = "Variant name is upstream labeling; inspect actualTensorTypes for effective precision."
        del reader
        copy_file(baseline / "onnx/LICENSE.txt", path / "LICENSE.txt")
        record(root, manifest, folder, f"https://huggingface.co/{GGUF_REPO}/tree/{GGUF_REV}", f"{variant.upper()} mixed GGUF; floating activations", provenance)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=ROOT / "models/cpu-models")
    parser.add_argument("--output", type=Path, default=ROOT / "models/low-precision-models")
    parser.add_argument("--engines", nargs="+", choices=["onnx", "mnn", "ncnn", "litert", "gguf"], default=["onnx", "mnn", "ncnn", "litert", "gguf"])
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    baseline = args.baseline.resolve()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"schemaVersion": 2, "engines": {}}
    manifest["schemaVersion"] = 2
    original = json.loads((baseline / "manifest.json").read_text())
    for folder, entry in original["engines"].items():
        for file in entry["files"]:
            copy_file(baseline / file["path"], root / file["path"])
        manifest["engines"][folder] = entry
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    source = WORK / "model.onnx"
    if any(e in args.engines for e in ["onnx", "mnn"]):
        fetch(f"https://huggingface.co/{ONNX}/resolve/{ONNX_REV}/model.onnx", source)
        assert sha(source) == SOURCE_HASH, "Original floating ONNX checksum mismatch"
    for engine in args.engines:
        if engine == "onnx":
            for bits in [4, 2]: prepare_onnx(source, baseline, root, manifest, bits)
        elif engine == "mnn":
            for bits in range(7, 1, -1): prepare_mnn(source, baseline, root, manifest, bits)
        elif engine == "litert":
            for bits in [4, 2]: prepare_litert(baseline, root, manifest, bits)
        elif engine == "ncnn": prepare_ncnn(baseline, root, manifest)
        else: prepare_gguf(baseline, root, manifest)
    print("Import this folder on the phone:", root, flush=True)


if __name__ == "__main__":
    main()
