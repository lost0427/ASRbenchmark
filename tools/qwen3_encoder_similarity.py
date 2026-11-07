"""Compare TensorRT Qwen3-ASR encoder outputs against an FP16 engine."""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--reference-engine", type=Path, required=True)
    parser.add_argument("--candidate-engine", type=Path, required=True)
    parser.add_argument("--trt-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("limit must be positive")

    sys.path.insert(0, str(args.trt_repo / "src"))
    import torch
    from transformers import AutoProcessor
    from qwen3_asr_trt.tensorrt_encoder import TensorRTAudioEncoder

    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines()]
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit("Manifest contains no audio rows")
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    reference = TensorRTAudioEncoder(args.reference_engine)
    candidate = TensorRTAudioEncoder(args.candidate_engine)
    similarities = []
    utterance_means = []
    reference_times = []
    candidate_times = []

    first_audio = args.data / rows[0]["flac"]
    warmup = processor.apply_transcription_request(audio=str(first_audio), language="English")
    for _ in range(2):
        reference.encode(warmup["input_features"], warmup["input_features_mask"])
        candidate.encode(warmup["input_features"], warmup["input_features_mask"])

    for index, row in enumerate(rows, 1):
        audio = args.data / row["flac"]
        inputs = processor.apply_transcription_request(audio=str(audio), language="English")
        start = time.perf_counter()
        expected = reference.encode(inputs["input_features"], inputs["input_features_mask"]).float()
        reference_times.append((time.perf_counter() - start) * 1000)
        start = time.perf_counter()
        actual = candidate.encode(inputs["input_features"], inputs["input_features_mask"]).float()
        candidate_times.append((time.perf_counter() - start) * 1000)
        if expected.shape != actual.shape:
            raise RuntimeError(f"Embedding shape mismatch for {row['id']}: {expected.shape} != {actual.shape}")
        cos = torch.nn.functional.cosine_similarity(expected, actual, dim=-1).cpu().numpy()
        similarities.extend(cos.tolist())
        utterance_means.append(float(np.mean(cos)))
        if index % 250 == 0:
            print(f"{index}/{len(rows)}", flush=True)

    result = {
        "reference": str(args.reference_engine),
        "candidate": str(args.candidate_engine),
        "audioCount": len(rows),
        "frameCount": len(similarities),
        "frameCosineMean": float(np.mean(similarities)),
        "frameCosineP05": float(np.percentile(similarities, 5)),
        "frameCosineMin": float(np.min(similarities)),
        "utteranceCosineMean": float(np.mean(utterance_means)),
        "referenceEncoderMedianMs": statistics.median(reference_times),
        "candidateEncoderMedianMs": statistics.median(candidate_times),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
