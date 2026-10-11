"""Run fixed-audio desktop correctness checks, without claiming phone speed."""
import argparse
import gc
import json
from pathlib import Path
import re
import subprocess
import wave

import numpy as np
from prepare_models import sha

ROOT = Path(__file__).resolve().parents[1]


def normalize(text):
    return re.sub(r"\s+", " ", re.sub(r"[^A-Z0-9 ]", " ", text.upper())).strip()


def wer(reference, text):
    a, b = normalize(reference).split(), normalize(text).split()
    previous = list(range(len(b) + 1))
    for i, left in enumerate(a):
        current = [i + 1]
        for j, right in enumerate(b):
            current.append(min(current[-1] + 1, previous[j + 1] + 1, previous[j] + (left != right)))
        previous = current
    return previous[-1] / max(len(a), 1)


def decode(logits, tokens, valid_frames):
    previous = -1
    text = []
    for best in logits[0, 4:valid_frames + 4].argmax(axis=-1):
        if best != 0 and best != previous:
            token = tokens[int(best)]
            if not token.startswith("<|"): text.append(token)
        previous = best
    return "".join(text).replace("▁", " ").strip()


def feature_frames(pcm, meta):
    import kaldi_native_fbank as knf
    options = knf.FbankOptions()
    options.frame_opts.samp_freq = 16000
    options.frame_opts.frame_length_ms = 25
    options.frame_opts.frame_shift_ms = 10
    options.frame_opts.dither = 0
    options.frame_opts.snip_edges = True
    options.frame_opts.window_type = "hamming"
    options.mel_opts.num_bins = 80
    options.mel_opts.low_freq = 20
    options.mel_opts.high_freq = 0
    bank = knf.OnlineFbank(options)
    bank.accept_waveform(16000, pcm.tolist())
    bank.input_finished()
    values = np.stack([bank.get_frame(i) for i in range(bank.num_frames_ready)])
    rows = (len(values) + 5) // 6
    indices = np.clip(np.arange(rows)[:, None] * 6 + np.arange(7)[None, :] - 3, 0, len(values) - 1)
    frames = values[indices].reshape(rows, 560)
    shift = np.array([float(x) for x in meta["neg_mean"].split(',')], dtype=np.float32)
    scale = np.array([float(x) for x in meta["inv_stddev"].split(',')], dtype=np.float32)
    return ((frames + shift) * scale)[None].astype(np.float32)


def check_model(folder, root, features, meta):
    path = root / folder
    tokens = {}
    token_file = path / "tokens.txt"
    if token_file.exists():
        tokens = {int(line.rsplit(' ', 1)[1]): line.rsplit(' ', 1)[0] for line in token_file.read_text(encoding='utf-8').splitlines()}
    frame_count = features.shape[1]
    inputs = {"x": features, "x_length": np.array([frame_count], dtype=np.int32),
              "language": np.array([int(meta["lang_en"])], dtype=np.int32),
              "text_norm": np.array([int(meta["without_itn"])], dtype=np.int32)}
    if folder.startswith("onnx"):
        import onnxruntime as ort
        filename = "model.int8.onnx" if folder == "onnx" else "model.onnx"
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(str(path / filename), options, providers=["CPUExecutionProvider"])
        actual_meta = session.get_modelmeta().custom_metadata_map
        assert all(actual_meta.get(key) == value for key, value in meta.items()), "Source metadata changed"
        first = decode(session.run(None, inputs)[0], tokens, frame_count)
        second = decode(session.run(None, inputs)[0], tokens, frame_count)
    elif folder.startswith("mnn"):
        import MNN.expr as F
        from MNN import nn
        module = nn.load_module_from_file(str(path / "model.mnn"), list(inputs), ["logits"], thread_num=4, memory_mode=F.MemoryMode.Low)
        tensors = [F.const(value.flatten().tolist(), list(value.shape), F.NCHW, F.float if value.dtype == np.float32 else F.int) for value in inputs.values()]
        first = decode(module.onForward(tensors)[0].read(), tokens, frame_count)
        second = decode(module.onForward(tensors)[0].read(), tokens, frame_count)
    elif folder.startswith("litert"):
        from ai_edge_litert.interpreter import Interpreter
        filename = "sensevoice_small_q8.tflite" if folder == "litert" else "model.tflite"
        interpreter = Interpreter(model_path=str(path / filename), num_threads=4)
        runner = interpreter.get_signature_runner("sv_500")
        padded = np.zeros((1, 500, 560), dtype=np.float32)
        padded[:, :frame_count] = features
        def run():
            return next(iter(runner(args_0=padded, args_1=np.array([frame_count], dtype=np.int32),
                                    args_2=inputs['language'], args_3=inputs['text_norm']).values()))
        first = decode(run(), tokens, frame_count)
        second = decode(run(), tokens, frame_count)
    elif folder.startswith("gguf"):
        executable = ROOT / ".native/gguf-host-build/bin/benchmark_gguf_smoke.exe"
        if not executable.exists(): executable = executable.with_suffix("")
        run = subprocess.run([str(executable), str(path / "model.gguf"), str(ROOT / "app/src/main/assets/audio/sample.wav")], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
        if run.returncode: raise RuntimeError(f"GGUF exited {run.returncode}: {run.stderr[-1000:]}")
        first = second = next(line[6:].strip() for line in run.stdout.splitlines() if line.startswith("TEXT: "))
    else:
        return {"status": "device_check_required", "note": "Official ncnn archive and hashes checked; Android inference remains to be run."}
    assert first == second, "Inconsistent repeated recognition"
    return {"status": "ok" if first else "empty_output", "text": first, "repeatedInferenceConsistent": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, default=ROOT / "models/low-precision-models")
    parser.add_argument("--output", type=Path, default=ROOT / ".native/lowprecision-host-checks.json")
    parser.add_argument("--folders", nargs="+")
    args = parser.parse_args()
    manifest = json.loads((args.models / "manifest.json").read_text())
    import onnx
    source = onnx.load(str(ROOT / ".native/lowprecision-source/model.onnx"), load_external_data=False)
    meta = {item.key: item.value for item in source.metadata_props}
    del source
    gc.collect()
    audio = ROOT / "app/src/main/assets/audio/sample.wav"
    audio_meta = json.loads(audio.with_suffix('.json').read_text())
    with wave.open(str(audio)) as wav:
        assert wav.getframerate() == 16000 and wav.getnchannels() == 1
        pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype='<i2').astype(np.float32)
    features = feature_frames(pcm, meta)
    report = {"purpose": "Desktop model loading and repeated fixed-audio recognition, not Android speed", "audioSha256": sha(audio), "results": []}
    for folder, entry in manifest['engines'].items():
        if args.folders and folder not in args.folders: continue
        print("Checking", folder, flush=True)
        try:
            for file in entry['files']:
                path = args.models / file['path']
                assert path.stat().st_size == file['sizeBytes'] and sha(path) == file['sha256'], f"Checksum mismatch: {path}"
            result = check_model(folder, args.models, features, meta)
            if 'text' in result: result['wer'] = wer(audio_meta['reference'], result['text'])
        except Exception as error:
            result = {"status": "error", "error": f"{type(error).__name__}: {error}"}
        result['folder'] = folder
        report['results'].append(result)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        print(folder, result, flush=True)
        gc.collect()
    if any(r['status'] == 'error' for r in report['results']): raise SystemExit(1)


if __name__ == '__main__': main()
