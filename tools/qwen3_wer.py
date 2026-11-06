"""Evaluate Qwen3-ASR WER over a sampled LibriSpeech manifest."""
import argparse
import contextlib
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from qwen3_asr_ort import build_prompt, clean_text, find_files, make_session, tensor_dtype
from windows_benchmark import normalize

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / ".native/librispeech"
DEFAULT_ORT = ROOT / ".native/models/qwen3-asr/ort-int4/extracted/qwen3-asr-1.7b-int4"
DEFAULT_SHERPA = ROOT / ".native/models/qwen3-asr/sherpa"
DEFAULT_GGUF_REPO = ROOT / ".native/qwen3-asr-gguf"
DEFAULT_TRANSCRIBE = ROOT / ".native/models/qwen3-asr/transcribe/Qwen3-ASR-1.7B-Q8_0.gguf"
DEFAULT_TRT_REPO = ROOT.parent / "Qwen3-ASR-TensorRT"


def edit_distance(reference, hypothesis):
    left, right = normalize(reference).split(), normalize(hypothesis).split()
    row = list(range(len(right) + 1))
    for i, token in enumerate(left, 1):
        current = [i]
        for j, other in enumerate(right, 1):
            current.append(min(current[-1] + 1, row[j] + 1, row[j - 1] + (token != other)))
        row = current
    return row[-1], len(left), len(right)


def load_pcm(path):
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sample_rate != 16000:
        raise ValueError(f"Expected 16 kHz audio: {path} ({sample_rate})")
    return np.asarray(audio, dtype=np.float32)


def transcribe_runner(model_path, device, threads):
    import transcribe_cpp
    devices = transcribe_cpp.backends()
    if device == "cuda":
        target = next((item for item in devices
                       if item.device_type == "gpu" and "CUDA" in item.name), None)
    else:
        target = next((item for item in devices if item.device_type == "cpu"), None)
    if target is None:
        raise RuntimeError(f"No transcribe.cpp {device} device")
    model = transcribe_cpp.Model(str(model_path), device=target)
    session = model.session(n_threads=threads)
    kwargs = {"timestamps": "none"}
    if model.supports("itn"):
        kwargs["itn"] = "off"
    return lambda pcm: session.run(pcm, **kwargs).text, device


def sherpa_runner(model_dir, threads):
    import sherpa_onnx
    recognizer = sherpa_onnx.OfflineRecognizer.from_qwen3_asr(
        conv_frontend=str(model_dir / "conv_frontend.onnx"),
        encoder=str(model_dir / "encoder.int8.onnx"),
        decoder=str(model_dir / "decoder.int8.onnx"),
        tokenizer=str(model_dir / "tokenizer"),
        num_threads=threads,
        provider="cpu",
        max_new_tokens=128,
    )
    def run(pcm):
        stream = recognizer.create_stream()
        stream.accept_waveform(16000, pcm)
        recognizer.decode_stream(stream)
        return clean_text(stream.result.text)
    return run, "cpu"


def ort_runner(model_dir, threads, max_new_tokens):
    from transformers import AutoTokenizer, WhisperFeatureExtractor
    encoder_path, init_path, step_path = find_files(model_dir)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, fix_mistral_regex=True)
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    eos = set(config["special_tokens"]["eos_token_ids"])
    embed_dtype = np.float16 if config.get("embed_tokens_dtype") == "float16" else np.float32
    embed = np.memmap(model_dir / "embed_tokens.bin", dtype=embed_dtype, mode="r").reshape(
        config["decoder"]["vocab_size"], config["decoder"]["hidden_size"])
    encoder = make_session(encoder_path, "cuda", threads)
    init = make_session(init_path, "cuda", threads)
    step = make_session(step_path, "cuda", threads)
    audio_dtype = np.float16 if "float16" in tensor_dtype(init, "audio_features") else np.float32
    decoder_dtype = np.float16 if "float16" in tensor_dtype(step, "input_embeds") else np.float32
    init_outputs = [value.name for value in init.get_outputs()]
    step_outputs = [value.name for value in step.get_outputs()]

    def run(pcm):
        features = feature_extractor(pcm, sampling_rate=16000, return_tensors="np",
                                     padding="longest", truncation=False)["input_features"]
        audio = encoder.run(None, {"mel": features.astype(np.float32)})[0].astype(audio_dtype)
        input_ids, offset, prompt_len = build_prompt(tokenizer, audio.shape[1])
        values = dict(zip(init_outputs, init.run(None, {
            "input_ids": input_ids,
            "position_ids": np.arange(prompt_len, dtype=np.int64)[None],
            "audio_features": audio,
            "audio_offset": np.array([offset], dtype=np.int64),
        })))
        logits = values["logits"]
        keys, values_kv = values["present_keys"], values["present_values"]
        tokens = []
        token = int(logits[0, -1].argmax())
        for index in range(max_new_tokens):
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
        return clean_text(tokenizer.decode(tokens, skip_special_tokens=True))
    return run, "cuda"


def gguf_runner(repo, model_dir, chunk_size, max_new_tokens):
    sys.path.insert(0, str(repo))
    from qwen_asr_gguf.inference import ASREngineConfig, QwenASREngine
    from qwen_asr_gguf.inference.audio import load_audio
    config = ASREngineConfig(
        model_dir=str(model_dir),
        encoder_frontend_fn="qwen3_asr_encoder_frontend.int4.onnx",
        encoder_backend_fn="qwen3_asr_encoder_backend.int4.onnx",
        llm_fn="qwen3_asr_llm.q4_k.gguf",
        onnx_provider="DML",
        llm_use_gpu=True,
        dml_pad_to=chunk_size,
        n_ctx=2048,
        chunk_size=chunk_size,
        memory_num=1,
        enable_aligner=False,
        verbose=False,
    )
    engine = QwenASREngine(config)
    def run(audio_path):
        pcm = load_audio(str(audio_path))
        with contextlib.redirect_stdout(io.StringIO()):
            result = engine.asr(pcm, context="", language="English",
                                chunk_size_sec=chunk_size, memory_chunks=1,
                                temperature=0.0, rollback_num=5)
        stages = {"encoderMs": result.performance["encode_time"] * 1000,
                  "prefillMs": result.performance["prefill_time"] * 1000,
                  "decodeMs": result.performance["decode_time"] * 1000}
        return clean_text(result.text), stages
    return run, "DML+Vulkan"


def tensorrt_exllama_runner(repo, model, engine_path, decoder_dir, max_new_tokens):
    sys.path.insert(0, str(repo / "src"))
    from exllamav3.generator.sampler import GreedySampler
    from qwen3_asr_trt.runtime import TensorRTExLlamaTranscriber

    transcriber = TensorRTExLlamaTranscriber(
        model, decoder_dir, engine_path, max_new_tokens, sampler=GreedySampler()
    )
    def run(audio_path):
        text = transcriber.transcribe(audio_path, "English")
        return text, transcriber.last_timings
    return run, "CUDA (TensorRT + ExLlamaV3)"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_DATA / "manifest.jsonl")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path,
                        default=DEFAULT_DATA / "qwen3-wer-1000")
    parser.add_argument("--engines", nargs="+",
                        default=["transcribe", "sherpa", "ort", "gguf"])
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--ort-model", type=Path, default=DEFAULT_ORT)
    parser.add_argument("--sherpa-model", type=Path, default=DEFAULT_SHERPA)
    parser.add_argument("--gguf-repo", type=Path, default=DEFAULT_GGUF_REPO)
    parser.add_argument("--gguf-model", type=Path, default=DEFAULT_GGUF_REPO / "model")
    parser.add_argument("--gguf-chunk", type=float, default=40.0)
    parser.add_argument("--trt-repo", type=Path, default=DEFAULT_TRT_REPO)
    parser.add_argument("--trt-model", default="Qwen/Qwen3-ASR-0.6B-hf")
    parser.add_argument("--trt-engine", type=Path)
    parser.add_argument("--trt-decoder", type=Path)
    parser.add_argument("--transcribe-model", type=Path, default=DEFAULT_TRANSCRIBE)
    parser.add_argument("--transcribe-device", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()
    if args.sample_size < 1 or args.threads < 1:
        parser.error("sample-size and threads must be positive")
    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines()]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample_path = args.output_dir / "sample.jsonl"
    if sample_path.is_file():
        sample = [json.loads(line) for line in sample_path.read_text(encoding="utf-8").splitlines()]
    else:
        rng = np.random.default_rng(args.seed)
        indices = np.sort(rng.choice(len(rows), size=min(args.sample_size, len(rows)),
                                     replace=False))
        sample = [rows[index] for index in indices]
        with sample_path.open("w", encoding="utf-8", newline="\n") as handle:
            for row in sample:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    if args.limit:
        sample = sample[:args.limit]
    print(f"sample={len(sample)} seed={args.seed}", flush=True)

    for engine in args.engines:
        output = args.output_dir / f"{engine}.jsonl"
        done = set()
        if args.resume and output.exists():
            done = {json.loads(line)["id"] for line in output.read_text(encoding="utf-8").splitlines()}
        selected = [row for row in sample if row["id"] not in done]
        print(f"{engine}: {len(selected)} pending of {len(sample)}", flush=True)
        if not selected:
            continue
        if engine == "transcribe":
            run, provider = transcribe_runner(args.transcribe_model, args.transcribe_device,
                                              args.threads)
        elif engine == "sherpa":
            run, provider = sherpa_runner(args.sherpa_model, args.threads)
        elif engine == "ort":
            run, provider = ort_runner(args.ort_model, args.threads, args.max_new_tokens)
        elif engine == "gguf":
            run, provider = gguf_runner(args.gguf_repo, args.gguf_model,
                                        args.gguf_chunk, args.max_new_tokens)
        elif engine == "tensorrt-exllama":
            artifact_dir = args.trt_repo / "artifacts/qwen3-asr-0.6b"
            run, provider = tensorrt_exllama_runner(
                args.trt_repo,
                args.trt_model,
                args.trt_engine or artifact_dir / "encoder-window.bf16.engine",
                args.trt_decoder or artifact_dir / "decoder-exl3-4bpw",
                args.max_new_tokens,
            )
        else:
            raise SystemExit(f"Unknown engine: {engine}")
        started = time.perf_counter()
        warmups = min(2, len(selected))
        if engine in ("gguf", "tensorrt-exllama"):
            for row in selected[:warmups]:
                run(args.data / row["flac"])
        else:
            warmups = 0
        print(f"{engine}: warmups={warmups}", flush=True)
        with output.open("a" if args.resume else "w", encoding="utf-8", newline="\n") as handle:
            for index, row in enumerate(selected, 1):
                audio_path = args.data / row["flac"]
                audio_input = audio_path if engine in ("gguf", "tensorrt-exllama") else load_pcm(audio_path)
                inference_start = time.perf_counter()
                inference_result = run(audio_input)
                elapsed_ms = (time.perf_counter() - inference_start) * 1000
                if isinstance(inference_result, tuple):
                    hypothesis, stages = inference_result
                else:
                    hypothesis, stages = inference_result, None
                distance, ref_words, hyp_words = edit_distance(row["reference"], hypothesis)
                result = {
                    "id": row["id"],
                    "reference": row["reference"],
                    "hypothesis": hypothesis,
                    "edits": distance,
                    "referenceWords": ref_words,
                    "hypothesisWords": hyp_words,
                    "wer": distance / max(1, ref_words),
                    "elapsedMs": elapsed_ms,
                }
                if stages:
                    result["stages"] = stages
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                if index % 25 == 0:
                    handle.flush()
                    print(f"{engine}: {index}/{len(selected)} ({time.perf_counter()-started:.1f}s)",
                          flush=True)
        all_rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
        edits = sum(item["edits"] for item in all_rows)
        words = sum(item["referenceWords"] for item in all_rows)
        latencies = [item["elapsedMs"] for item in all_rows]
        stage_names = {name for item in all_rows for name in item.get("stages", {})}
        summary = {
            "engine": engine,
            "provider": provider,
            "utterances": len(all_rows),
            "referenceWords": words,
            "hypothesisWords": sum(item["hypothesisWords"] for item in all_rows),
            "edits": edits,
            "corpusWER": edits / max(1, words),
            "emptyOutputs": sum(1 for item in all_rows if not normalize(item["hypothesis"])),
            "elapsedSeconds": time.perf_counter() - started,
            "meanMs": float(np.mean(latencies)),
            "medianMs": float(np.median(latencies)),
            "p90Ms": float(np.percentile(latencies, 90)),
            "audioSeconds": sum(row["durationSeconds"] for row in sample),
            "rtf": sum(latencies) / (1000 * sum(row["durationSeconds"] for row in sample)),
            "medianStagesMs": {
                name: float(np.median([item["stages"][name] for item in all_rows
                                       if name in item.get("stages", {})]))
                for name in stage_names
            },
            "threads": args.threads,
            "warmups": warmups,
            "sampleSize": len(sample),
            "seed": args.seed,
        }
        (args.output_dir / f"{engine}.summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
