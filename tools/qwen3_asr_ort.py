"""Benchmark Qwen3-ASR split ONNX exports with ONNX Runtime."""
import argparse
import gc
import json
import re
import time
from pathlib import Path

import numpy as np

from windows_benchmark import enable_cuda_dlls, load_audio, wer

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUDIO = ROOT / "app/src/main/assets/audio/sample.wav"


def find_files(folder):
    for name in ("encoder.int4.onnx", "encoder.onnx"):
        if (folder / name).is_file():
            encoder = folder / name
            break
    else:
        raise SystemExit(f"No encoder ONNX in {folder}")
    init = folder / ("decoder_init.int4.onnx" if "int4" in encoder.name else "decoder_init.onnx")
    step = folder / ("decoder_step.int4.onnx" if "int4" in encoder.name else "decoder_step.onnx")
    for path in (init, step, folder / "embed_tokens.bin", folder / "tokenizer.json"):
        if not path.is_file():
            raise SystemExit(f"Missing model file: {path}")
    return encoder, init, step


def tensor_dtype(session, name):
    for value in session.get_inputs():
        if value.name == name:
            return value.type
    raise KeyError(name)


def clean_text(text):
    if "<asr_text>" in text:
        text = text.split("<asr_text>", 1)[1]
    return re.sub(r"^\s*language\s+[A-Za-z]+\s*", "", text, flags=re.IGNORECASE).strip()


def build_prompt(tokenizer, audio_tokens):
    prefix = "<|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|>"
    suffix = "<|audio_end|><|im_end|>\n<|im_start|>assistant\n"
    ids = tokenizer.encode(prefix + "<|audio_pad|>" * audio_tokens + suffix,
                           add_special_tokens=False)
    return np.array([ids], dtype=np.int64), ids.index(151676), len(ids)


def make_session(path, provider, threads):
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads if provider == "cpu" else 1
    options.inter_op_num_threads = 1
    providers = ["CPUExecutionProvider"]
    if provider == "cuda":
        enable_cuda_dlls()
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(str(path), options, providers=providers)
    actual = session.get_providers()[0]
    if provider == "cuda" and actual != "CUDAExecutionProvider":
        raise SystemExit("CUDAExecutionProvider failed to initialize")
    return session


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--provider", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--features", type=Path,
                        help="Precomputed encoder output (.npy) from a separate process")
    parser.add_argument("--features-out", type=Path)
    parser.add_argument("--encoder-ms", type=float, default=0.0,
                        help="Encoder latency measured by a separate --encoder-only run")
    parser.add_argument("--encoder-only", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.warmups < 0 or args.repeats < 1:
        parser.error("warmups must be nonnegative; repeats must be positive")
    try:
        from transformers import AutoTokenizer, WhisperFeatureExtractor
    except ImportError as error:
        raise SystemExit("Install tools/qwen3-asr-requirements.txt") from error

    folder = args.model_dir.resolve()
    encoder_path, init_path, step_path = find_files(folder)
    pcm = load_audio(args.audio)
    reference = None
    reference_path = args.reference or args.audio.with_suffix(".json")
    if reference_path.is_file():
        reference = json.loads(reference_path.read_text(encoding="utf-8")).get("reference")
    feature_extractor = WhisperFeatureExtractor.from_pretrained(folder)
    tokenizer = AutoTokenizer.from_pretrained(folder, fix_mistral_regex=True)
    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    eos = set(config["special_tokens"]["eos_token_ids"])
    embed_dtype = np.float16 if config.get("embed_tokens_dtype") == "float16" else np.float32
    embed = np.memmap(folder / "embed_tokens.bin", dtype=embed_dtype, mode="r").reshape(
        config["decoder"]["vocab_size"], config["decoder"]["hidden_size"])

    def mel():
        return feature_extractor(pcm, sampling_rate=16000, return_tensors="np",
                                 padding="longest", truncation=False)["input_features"].astype(np.float32)

    mel_samples = []
    for _ in range(max(5, args.warmups + 1)):
        start = time.perf_counter()
        features = mel()
        mel_samples.append((time.perf_counter() - start) * 1000)
    mel_ms = float(np.median(mel_samples))

    if args.features:
        audio = np.load(args.features)
        encoder_ms = args.encoder_ms
    else:
        encoder = make_session(encoder_path, args.provider, args.threads)
        encoder_samples = []
        for _ in range(max(5, args.warmups + 1)):
            start = time.perf_counter()
            audio = encoder.run(None, {"mel": features})[0]
            encoder_samples.append((time.perf_counter() - start) * 1000)
        encoder_ms = float(np.median(encoder_samples))
        if args.features_out:
            args.features_out.parent.mkdir(parents=True, exist_ok=True)
            np.save(args.features_out, audio)
        if args.encoder_only:
            report = {"engine": "onnxruntime", "stage": "encoder", "provider": args.provider,
                      "model": str(folder), "melMs": mel_ms, "encoderMs": encoder_ms}
            print(json.dumps(report, sort_keys=True), flush=True)
            return
        del encoder
        gc.collect()

    init = make_session(init_path, args.provider, args.threads)
    audio = audio.astype(np.float16 if "float16" in tensor_dtype(init, "audio_features") else np.float32)
    step = make_session(step_path, args.provider, args.threads)
    input_ids, offset, prompt_len = build_prompt(tokenizer, audio.shape[1])
    init_inputs = {"input_ids": input_ids,
                   "position_ids": np.arange(prompt_len, dtype=np.int64)[None],
                   "audio_features": audio,
                   "audio_offset": np.array([offset], dtype=np.int64)}
    init_outputs = [value.name for value in init.get_outputs()]
    step_outputs = [value.name for value in step.get_outputs()]
    decoder_dtype = np.float16 if "float16" in tensor_dtype(step, "input_embeds") else np.float32

    def run_once():
        start = time.perf_counter()
        values = dict(zip(init_outputs, init.run(None, init_inputs)))
        prefill_ms = (time.perf_counter() - start) * 1000
        logits = values["logits"]
        keys, values_kv = values["present_keys"], values["present_values"]
        tokens = []
        token = int(logits[0, -1].argmax())
        decode_start = time.perf_counter()
        for index in range(args.max_new_tokens):
            if token in eos:
                break
            tokens.append(token)
            outputs = dict(zip(step_outputs, step.run(None, {
                "input_embeds": embed[token][None, None, :].astype(decoder_dtype),
                "position_ids": np.array([[prompt_len + index]], dtype=np.int64),
                "past_keys": keys,
                "past_values": values_kv,
            })))
            logits = outputs["logits"]
            keys, values_kv = outputs["present_keys"], outputs["present_values"]
            token = int(logits[0, -1].argmax())
        return prefill_ms, (time.perf_counter() - decode_start) * 1000, tokens

    for _ in range(args.warmups):
        run_once()
    prefill_samples, decode_samples, token_counts = [], [], []
    result_tokens = []
    for _ in range(args.repeats):
        prefill, decode, result_tokens = run_once()
        prefill_samples.append(prefill)
        decode_samples.append(decode)
        token_counts.append(len(result_tokens))
    text = clean_text(tokenizer.decode(result_tokens, skip_special_tokens=True))
    prefill_ms = float(np.median(prefill_samples))
    decode_ms = float(np.median(decode_samples))
    seconds = len(pcm) / 16000
    total_ms = mel_ms + encoder_ms + prefill_ms + decode_ms
    report = {
        "engine": "onnxruntime",
        "provider": args.provider,
        "model": str(folder),
        "audioSeconds": seconds,
        "melMs": mel_ms,
        "encoderMs": encoder_ms,
        "prefillMs": prefill_ms,
        "decodeMs": decode_ms,
        "meanMs": total_ms,
        "rtf": total_ms / (seconds * 1000),
        "tokens": int(np.median(token_counts)),
        "tokensPerSecond": float(np.median(token_counts)) / (decode_ms / 1000),
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
