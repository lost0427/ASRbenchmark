# `.native` inventory and what reproduction actually needs

`.native/` is git-ignored and holds 54 GB of downloaded corpora, model weights,
virtual environments and benchmark outputs. Nothing in it is required to view the
published results. This file records what each part is for, how to rebuild it,
and which parts matter if the numbers have to be re-measured.

## Already committed, no download needed

| Path | Contents | Size |
| --- | --- | --- |
| `data/runs/` | the 14 run directories that `tools/plot_*.py` read (`*.jsonl`, `*.summary.json`, `sample.jsonl`) | 97 MB |
| `data/manifest.jsonl` | LibriSpeech test-clean manifest, 2,620 rows, seed 1337 sampling input | 0.8 MB |
| `data/envs/*.txt` | `pip freeze` for the five virtual environments and the `qwen-asr-pkgs` tree | 6 KB |

`tools/run_data.py` prefers `.native/librispeech` and falls back to
`data/runs`, so every figure rebuilds from a clone. Verified byte-identical for
all twelve published charts.

## Needed to re-run a benchmark

| Path | Size | Reproduce with |
| --- | ---: | --- |
| `librispeech/LibriSpeech/test-clean` | 0.34 GB | `tar -xf test-clean.tar.gz -C .native/librispeech` from `openslr.org/12`; the manifest is committed |
| `qwen3-asr-gguf/` | 5.4 GB | `git clone --depth 1 https://github.com/HaujetZhao/Qwen3-ASR-GGUF.git .native/qwen3-asr-gguf` at commit `e790e3b`, then unpack the release zips `Qwen3-ASR-{0.6B,1.7B}-gguf.zip` and `Qwen3-ForceAligner-0.6B-gguf.zip` from the `models` tag; apply `python tools/patch_gguf_aligner_timing.py .native/qwen3-asr-gguf` before running the aligner |
| `qwen-gguf-env/` | 0.41 GB | Python 3.13 venv + `data/envs/qwen-gguf-env.txt`; the llama bindings are the vendored ctypes module inside the GGUF checkout, no wheel required |
| `qwen-asr-pkgs/` | 0.73 GB | the official Transformers backend (`qwen_asr 0.0.6`, `transformers 4.57.6`); `data/envs/qwen-asr-pkgs.txt` lists the exact tree, normally produced with `pip install --target .native/qwen-asr-pkgs qwen-asr` |
| `models/qwen3-asr/` | 23 GB | exported by `tools/prepare_qwen3_asr.py` (ORT fp16/int4, sherpa int8, transcribe.cpp GGUF); only needed for the ORT/sherpa/transcribe sections |
| `calibration/` | 6.35 GB | `train-clean-100.tar.gz` from `openslr.org/12`; `tools/prepare_exl3_calibration.py` builds the packed tensors |
| `tensorrt/` | 4.03 GB | TensorRT 10.13.3.9 SDK, URL listed in `WINDOWS_BENCHMARK_RESULTS.md` |
| `transcribe-bin/`, `transcribe-env/` | 1.9 GB | `transcribe.cpp` v0.4.1 release plus Python 3.13 venv |
| `gpu-env/`, `model-env/`, `trt-env/` | 2.3 GB | Python 3.13 venvs matching the corresponding `data/envs/*.txt` |

## Disposable

- `trt/*.engine` — TensorRT engines are tied to the GPU, driver and TensorRT
  build that produced them; rebuild instead of copying.
- `lowprecision-source/`, `onnxruntime-src/`, `mnn-archive-source/`,
  `ncnn-source/`, `onnx-archive-source/`, `gguf-archive-source/`,
  `transcribe.cpp/` — source trees for the low-precision investigation.
- `*.log`, `*.json` scratch files in the `.native` root (bf16 profiling dumps,
  host checks, smoke runs).

## Environment versions

| Environment | Python | Key packages |
| --- | --- | --- |
| `gpu-env` | 3.13.5 | onnxruntime-gpu, numpy |
| `model-env` | 3.13.5 | onnxruntime-directml, librosa, soundfile |
| `transcribe-env` | 3.13.5 | soundfile, numpy |
| `trt-env` | 3.13.5 | tensorrt |
| `qwen-gguf-env` | 3.13.5 | numpy, onnxruntime-directml |
| `qwen-asr-pkgs` | 3.13.5 | `qwen_asr 0.0.6`, `transformers 4.57.6`, `numpy 2.5.4` |

The TensorRT and EXL3 artifacts themselves (engines, quantized decoders) live in
the `Qwen3-ASR-TensorRT` repository under `artifacts/`, not here.
