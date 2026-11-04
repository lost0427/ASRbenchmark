# LibriSpeech test-clean: corpus WER

This is the first corpus-level accuracy check for SenseVoiceSmall and a
1,000-utterance check for Qwen3-ASR. The fixed 23.86-second sample used elsewhere
is useful for latency and smoke testing, but it is too small to rank models.

## Dataset and method

- Archive: `test-clean.tar.gz`, 2,620 FLAC utterances, 87 chapters, 40 speakers.
- Audio: 16 kHz mono; 19,452.48 seconds (5.40 hours); 53,120 reference words.
- Reference: the original LibriSpeech `trans.txt` text.
- Normalization: uppercase, strip punctuation, collapse whitespace, matching the
  existing `tools/windows_benchmark.py` metric. References were not corrected.
- Primary metric: **corpus WER = total edit distance / total reference words**.
- Confidence intervals: 2,000 utterance-level bootstrap resamples (95%).
- Models run once per utterance; no warmup or repeat. Model load is outside the
  audio loop.
- Language English, ITN off, timestamps off, 12 CPU threads. transcribe.cpp used
  its CUDA backend; the other engines used CPU.

## Results

![SenseVoice corpus WER](windows-librispeech-sensevoice-wer.png)

| Engine | Configuration | Corpus WER | 95% CI | Mean utterance WER | Empty outputs |
| --- | --- | ---: | ---: | ---: | ---: |
| LiteRT | Q8 | **3.2925%** | 3.0835–3.5053% | 3.9007% | 0 |
| transcribe.cpp | Q8, CUDA | **3.2944%** | 3.0770–3.5066% | 3.8951% | 0 |
| MNN | Q8 | 3.3151% | 3.1179–3.5420% | 3.9242% | 0 |
| ONNX Runtime | int8 | 3.3490% | 3.1446–3.5601% | 3.9550% | 0 |
| sherpa-onnx | int8 | 3.3754% | 3.1737–3.5919% | 4.0058% | 0 |
| ncnn | INT8 | 12.4078% | 12.0195–12.8184% | 12.7594% | 0 |

The five Q8/int8 engines are within **0.083 percentage points** of each other
(3.2925% to 3.3754%), and their confidence intervals overlap. This corpus does
not justify ranking them by WER. LiteRT Q8 and transcribe.cpp Q8 are tied at the
front within noise. ncnn INT8 is clearly worse at 12.41%; it also produced no
empty outputs, so this is a decoding/quantization quality issue rather than a
failed run.

WER by duration:

| Engine | 0–5 s | 5–10 s | 10–20 s | 20 s+ |
| --- | ---: | ---: | ---: | ---: |
| LiteRT Q8 | 4.19% | 3.13% | 2.87% | 3.64% |
| transcribe.cpp Q8 | 4.21% | 3.12% | 2.87% | 3.64% |
| MNN Q8 | 4.13% | 3.18% | 2.89% | 3.71% |
| ONNX Runtime int8 | 4.24% | 3.16% | 2.97% | 3.64% |
| sherpa-onnx int8 | 4.36% | 3.12% | 2.97% | 3.77% |
| ncnn INT8 | 11.63% | 11.17% | 12.82% | 16.55% |

Short utterances have higher WER because a single word error is a larger
fraction of the reference. Longer utterances are not uniformly better for ncnn,
which degrades further above 20 seconds.

## Qwen3-ASR: 1,000-utterance subset

Qwen3 has not yet been run on all 2,620 utterances. To get a fast, directly
comparable result, a fixed random subset was drawn once with seed 1337 and used
for every Qwen3 engine and for aggregation of the existing SenseVoice results:

- **1,000 utterances**, 19,972 reference words.
- Same audio, references, normalization, and corpus-WER formula as above.
- Same random sample IDs for both models.

![Qwen3 vs SenseVoice WER](windows-librispeech-qwen3-wer.png)

| Engine | Configuration | Corpus WER | 95% CI | Mean utterance WER | Empty outputs |
| --- | --- | ---: | ---: | ---: | ---: |
| Qwen3 transcribe.cpp | Q8, CUDA | **1.7825%** | 1.5399–2.0456% | 2.1473% | 0 |
| Qwen3-ASR-GGUF | q4_k, DML+Vulkan | 2.0228% | 1.7534–2.3233% | 2.2524% | 0 |
| Qwen3 ONNX Runtime | int4, CUDA | 2.0328% | 1.7772–2.3215% | 2.4056% | 0 |
| Qwen3 sherpa-onnx | int8, CPU | 3.0893% | 2.4813–3.8444% | 3.3640% | 2 |
| SenseVoice transcribe.cpp | Q8, CUDA | 3.2846% | 2.9515–3.6492% | 3.8964% | 0 |

Qwen3 transcribe.cpp is **1.50 percentage points better** than SenseVoice
transcribe.cpp on the same sample (1.78% vs 3.28%), a 46% relative reduction;
the confidence intervals do not overlap. Qwen3 ONNX Runtime int4 and
Qwen3-ASR-GGUF q4_k are also better than the best SenseVoice result on this
sample, at about 2.02%. Qwen3 sherpa int8 is nominally better than SenseVoice
(3.09% vs 3.28%) but its interval is wide because two of the 1,000 outputs were
garbage/non-English; those two are counted as errors.

This reverses the impression from the single 23.86-second clip, where both
models happened to score 5.80%. The 1,000-utterance sample is a better accuracy
signal, but it is still a subset and only read English. Do not call it a full
Qwen3 corpus result.

## What changed from the single clip

The 23.86-second sample produced 5.80% WER for the Q8 engines; the full corpus
produces about 3.3%. The single clip is not a miniature estimate of corpus WER:
it contains unusual spellings and a small number of words. The corpus result is
the number to use for accuracy comparisons.

The earlier single-clip Qwen3 numbers (5.80%) should not be compared directly
to the full SenseVoice table. The 1,000-utterance Qwen3 subset above is the
current apples-to-apples accuracy check. The next step is to run Qwen3 on all
2,620 utterances if a full-corpus number is required.

## Coverage and limitations

- Included: full-corpus SenseVoice with LiteRT Q8, MNN Q8, ONNX Runtime int8,
  sherpa-onnx int8, ncnn INT8, and transcribe.cpp Q8.
- Qwen3 is included on the common 1,000-utterance subset only: transcribe.cpp
  Q8, ONNX Runtime int4, Qwen3-ASR-GGUF q4_k, and sherpa-onnx int8.
- The custom `benchmark_gguf_smoke.exe` and TensorRT runners are single-file or
  fixed-shape tools. They are not included in this full-corpus table; the
  transcribe.cpp result covers the same GGUF Q8 weights with a persistent model.
- This is read English speech. It does not measure noisy audio, multilingual
  audio, long-form segmentation, or ITN.
- Raw per-utterance JSONL and summaries are in
  `.native/librispeech/sensevoice-wer/` and
  `.native/librispeech/qwen3-wer-1000/` and are not committed.

## Reproduction

```powershell
tar -xf test-clean.tar.gz -C .native/librispeech
.native/model-env/Scripts/python.exe tools/prepare_librispeech.py
.native/model-env/Scripts/python.exe tools/sensevoice_wer.py `
  --output-dir .native/librispeech/sensevoice-wer `
  --engines sherpa ort mnn ncnn litert
.native/transcribe-env/Scripts/python.exe tools/sensevoice_wer.py `
  --output-dir .native/librispeech/sensevoice-wer `
  --engines transcribe --transcribe-device cuda
```

Qwen3 on the same 1,000-utterance sample:

```powershell
.native/transcribe-env/Scripts/python.exe tools/qwen3_wer.py `
  --output-dir .native/librispeech/qwen3-wer-1000 `
  --engines transcribe --transcribe-device cuda
.native/model-env/Scripts/python.exe tools/qwen3_wer.py `
  --output-dir .native/librispeech/qwen3-wer-1000 --engines sherpa
.native/gpu-env/Scripts/python.exe tools/qwen3_wer.py `
  --output-dir .native/librispeech/qwen3-wer-1000 --engines ort
.native/qwen-gguf-env/Scripts/python.exe tools/qwen3_wer.py `
  --output-dir .native/librispeech/qwen3-wer-1000 --engines gguf
```

Each engine writes `*.jsonl` plus a `*.summary.json`. The report chart is
generated by `tools/plot_sensevoice_wer.py` and `tools/plot_qwen3_wer.py`.
