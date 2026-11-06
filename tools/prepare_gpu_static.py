"""Prepare a static-shape SenseVoice model for GPU backends.

Bakes the fixed-clip frame count and the three scalar inputs into the graph,
then constant-folds the dynamic Shape/Gather/Reshape chain with onnxsim. The
graph keeps its four input declarations so the existing sherpa wrappers keep
working; the scalar inputs simply have no consumers.

Outputs:
  onnx-fp32-static/model.onnx
  mnn-fp32-static/model.mnn  (+ tokens/LICENSE)
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import wave

import numpy as np
import onnx
from onnx import numpy_helper

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / ".native" / "lowprecision-source"
SOURCE = WORK / "model.onnx"
AUDIO = ROOT / "app/src/main/assets/audio/sample.wav"


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def fixed_frames():
    with wave.open(str(AUDIO)) as wav:
        samples = wav.getnframes()
    fbank = 1 + (samples - 400) // 160
    return (fbank + 5) // 6


def input_type(model, name):
    for item in model.graph.input:
        if item.name == name:
            return item.type.tensor_type.elem_type
    raise KeyError(name)


def bake(model, name, value):
    dtype = np.int64 if input_type(model, name) == onnx.TensorProto.INT64 else np.int32
    constant = f"__{name}_const"
    model.graph.initializer.append(numpy_helper.from_array(np.array([value], dtype=dtype), constant))
    for node in model.graph.node:
        for index, item in enumerate(node.input):
            if item == name:
                node.input[index] = constant
    return constant


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--root", type=Path, default=ROOT / "models/low-precision-models")
    parser.add_argument("--frames", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    frames = args.frames or fixed_frames()
    root = args.root.resolve()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source = args.source.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)

    static_dir = WORK / "onnx-fp32-static"
    static_model = static_dir / "model.onnx"
    if args.force or not static_model.exists():
        model = onnx.load(str(source))
        meta = {item.key: item.value for item in model.metadata_props}
        bake(model, "x_length", frames)
        bake(model, "language", int(meta["lang_en"]))
        bake(model, "text_norm", int(meta["without_itn"]))
        for item in model.graph.input:
            if item.name == "x":
                dims = item.type.tensor_type.shape.dim
                dims[0].dim_value = 1
                dims[1].dim_value = frames
                dims[2].dim_value = 560
        onnx.checker.check_model(model)
        print(f"baked frames={frames}; running onnxsim", flush=True)
        import onnxsim
        simplified, ok = onnxsim.simplify(
            model,
            overwrite_input_shapes={"x": [1, frames, 560], "x_length": [1],
                                    "language": [1], "text_norm": [1]},
            dynamic_input_shape=False,
        )
        if not ok:
            raise RuntimeError("onnxsim did not converge")
        static_dir.mkdir(parents=True, exist_ok=True)
        onnx.save(simplified, str(static_model))
        print("static ONNX inputs:", [i.name for i in simplified.graph.input], flush=True)
        print("static ONNX outputs:", [o.name for o in simplified.graph.output], flush=True)
        print("nodes:", len(simplified.graph.node), flush=True)

    for name in ["tokens.txt", "LICENSE.txt"]:
        shutil.copyfile(root / "mnn" / name, static_dir / name)

    mnn_dir = root / "mnn-fp32-static"
    mnn_dir.mkdir(parents=True, exist_ok=True)
    mnn_model = mnn_dir / "model.mnn"
    if args.force or not mnn_model.exists():
        subprocess.run([sys.executable, "-c", "import _tools,sys; _tools.mnnconvert(sys.argv[1:])",
                        "MNNConvert", "-f", "ONNX", "--modelFile", str(static_model),
                        "--MNNModel", str(mnn_model), "--bizCode", "MNN"], check=True)
    for name in ["tokens.txt", "LICENSE.txt"]:
        shutil.copyfile(root / "mnn" / name, mnn_dir / name)

    provenance = {
        "sourceSha256": sha(source),
        "converter": "onnxsim 0.8.1 + MNN 3.6.1",
        "frames": frames,
        "method": "bake x_length/language/text_norm as constants, fold dynamic shape chain, static x",
        "limitation": "static shape is tied to the fixed 23.86 s clip; variable-length audio needs bucketing",
    }
    (mnn_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    files = sorted(item for item in mnn_dir.iterdir() if item.is_file())
    manifest["engines"]["mnn-fp32-static"] = {
        "source": "https://huggingface.co/csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17",
        "quantization": "FP32 weights, static shapes, MNN 3.6.1",
        "files": [{"path": f"mnn-fp32-static/{item.name}", "sizeBytes": item.stat().st_size,
                   "sha256": sha(item)} for item in files],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"mnn-fp32-static: {mnn_model.stat().st_size / 1e6:.1f} MB, frames={frames}")


if __name__ == "__main__":
    main()
