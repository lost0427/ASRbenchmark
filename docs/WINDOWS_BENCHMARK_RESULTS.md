# Windows benchmark

The first table is a historical 4-thread run on Windows 11 with Python 3.13.5,
an AMD Ryzen AI 9 HX 370, one warmup and three measured repetitions. Later
sections use 12 threads, optimized AVX-512 builds, CUDA, or external engines
where explicitly noted. The workload is always the same 23.86-second 16 kHz
LibriSpeech clip used by the Android benchmark. Results include feature
extraction and decoding unless a row is labelled engine-only.

| Engine / model | Mean ms | Median ms | WER |
| --- | ---: | ---: | ---: |
| sherpa-onnx Q8 | 560 | 575 | 5.80% |
| ONNX Runtime Q8 | 899 | 900 | 5.80% |
| ONNX Runtime Q4 | 1,320 | 1,290 | 7.25% |
| ONNX Runtime Q2 | 1,176 | 1,200 | 89.86% |
| MNN Q8 | 798 | 791 | 5.80% |
| MNN Q7 | 804 | 797 | 5.80% |
| MNN Q6 | 804 | 807 | 4.35% |
| MNN Q5 | 812 | 796 | 4.35% |
| MNN Q4 | 687 | 697 | 7.25% |
| MNN Q3 | 969 | 962 | 44.93% |
| MNN Q2 | 933 | 959 | 100.00% (empty output) |
| LiteRT Q8 | 1,061 | 1,061 | 5.80% |
| LiteRT Q4 | 964 | 945 | 7.25% |
| LiteRT Q2 | 1,119 | 1,142 | 100.00% (empty output) |

This is a short single-clip run intended to validate Windows model compatibility
and the CLI measurement path. Three samples are not enough for a stable ranking,
and one utterance does not establish corpus accuracy. The checked-in CLI defaults
to three warmups and ten measured repetitions for follow-up runs.

The CLI now also covers ncnn (INT8 and FP16), GGUF (Q3-K through FP16) and ONNX
Runtime with `--provider cuda`. The table above predates those additions; rerun
the CLI to refresh it.

## All engines (CPU)

![All engines on CPU](windows-all-engines.png)

Every manifest model folder on CPU, sorted by mean latency (3 warmups, 10
repeats). This chart uses 4 threads and the default AVX2 GGUF build; the CUDA
comparison below uses 12 threads and the AVX-512 build. Best 4-thread result per
engine:

| Engine | Best model | Mean ms | WER |
| --- | --- | ---: | ---: |
| ncnn | `ncnn` (INT8) | 623 | 13.04% |
| ONNX Runtime | `onnx` (Q8) | 904 | 5.80% |
| sherpa-onnx | `onnx` (Q8) | 933 | 5.80% |
| LiteRT | `litert-q4` | 1,047 | 7.25% |
| MNN | `mnn-q4` | 1,075 | 7.25% |
| GGUF | `gguf-q8_0` | 2,169 | 5.80% |

## CPU vs CUDA

![CPU vs CUDA latency and WER](windows-gpu-vs-cpu.png)

ONNX Runtime 1.26.0 and the GGUF CUDA runner, RTX 4060 Laptop, same clip, 3
warmups and 10 repeats. ORT rows use the same 1.26.0 runtime on both providers.
CPU rows use 12 threads; ORT CUDA lets ORT choose its intra-op thread count
because CPU fallback nodes dominate int8 Q8.

| Engine | Model | CPU mean | CUDA mean | Speedup | WER CPU | WER CUDA |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ORT | Q8 (int8) | 609 ms | 471 ms | 1.29x | 5.80% | 5.80% |
| ORT | Q4 | 790 ms | 99 ms | 8.0x | 7.25% | 7.25% |
| ORT | Q2 | 1,108 ms | 104 ms | 10.6x | 89.86% | 100.00% (empty) |
| ORT | FP32 | 831 ms | 89 ms | 9.3x | 5.80% | 5.80% |
| GGUF | Q8 | 1,236 ms | 101 ms | 12.2x | 5.80% | 5.80% |
| GGUF | Q4 | 1,461 ms | 92 ms | 15.8x | 7.25% | 7.25% |
| GGUF | FP16 | 1,753 ms | 90 ms | 19.5x | 5.80% | 100.00% (empty) |

CUDA wins on every usable row, but int8 Q8 only by 1.3x. GGUF FP16 runs on CUDA
without errors yet returns empty text, so it is not usable for accuracy.
An ONNX FP16 conversion was also tried: the CPU graph failed validation and the
CUDA graph produced all-NaN logits, so FP16 ONNX is omitted. The working
floating-point GPU result is FP32. `onnx-fp32` is the official unquantized model
at the pinned revision; it is not downloaded by `prepare_models.py`.

## Phase A tuning (Windows)

- The ORT pipeline splits into roughly 43-47 ms fbank/CMVN, 50-60 ms FP32/Q4
  inference and 1.4 ms decode. Feature extraction is almost half of the total,
  so a faster model engine alone cannot go below about 45 ms.
- Setting ORT CUDA `intra_op_num_threads` from 4 to automatic cut Q8 from
  1,050 ms to 471 ms. CPU fallback nodes were the bottleneck, not the GPU.
- GGUF flash attention (`--gguf-flash`) cut CUDA Q8 from 116 to 101 ms and Q4
  from 107 to 92 ms. FP16 is still empty.
- CUDA provider options (`cudnn_conv_algo_search=EXHAUSTIVE`,
  `arena_extend_strategy=kSameAsRequested`, `do_copy_in_default_stream=0`) were
  neutral; `use_tf32=0` was slower (50 to 67 ms for the FP32 graph).
- AVX-512: this Ryzen AI 9 HX 370 reports AVX-512F/DQ/BW/VL/VNNI/VBMI/BF16. The
  ORT wheel contains AVX-512F kernels and the sherpa-ncnn wheel contains AVX-512
  convolution kernels; the MNN and LiteRT wheels do not. The GGUF CPU build was
  AVX2-only, so `tools/build_gguf.py --host --avx512` was added. It improves
  GGUF at 12 threads (Q8 1,346 to 1,236 ms, Q4 1,534 to 1,461 ms) but is worse
  at 4 threads (Q8 2,169 to 2,529 ms).

### BF16

No published BF16 model exists: the upstream ONNX repository ships FP32 and
int8, and the GGUF repository ships FP32, FP16 and Q3-Q8.

- GGUF BF16: the pinned ggml CUDA backend only references BF16 in flash
  attention; it has no general BF16 matmul kernels, so a BF16 GGUF would not be
  GPU-accelerated.
- ONNX FP16: ORT CUDA loads and runs an FP16 graph, but this model produces
  all-NaN logits. Even a MatMul/Gemm-only FP16 graph with FP32 boundary casts
  returned 100% WER (empty text), so FP16 is not usable for SenseVoice here.
  BF16 is stable because it keeps FP32's exponent range.
- ONNX BF16: ORT CUDA can execute a BF16 MatMul (verified on this GPU). A
  MatMul/Gemm-weight BF16 graph with FP32 boundary casts reproduced WER 5.80%
  at 98.2 ms, slower than FP32 at 89.1 ms because of cast and fusion overhead.
  ORT CPU has no BF16 MatMul kernel, so the same model fails on CPU.

ONNX-level BF16 is not a speed win. TensorRT BF16 is the exception and is
covered next.

## TensorRT (external engine)

TensorRT has no SenseVoice API. `tools/tensorrt_benchmark.py` builds a
fixed-shape plan from the official FP32 ONNX and runs it with BF16, FP16, or
FP32.

Results on the RTX 4060 Laptop, 23.86 s clip:

| Configuration | Engine ms | End-to-end | WER |
| --- | ---: | ---: | ---: |
| BF16, fixed 398 frames | 32-40 | ~70-90 ms | 5.80% |
| BF16, dynamic 64-3000 frames | 65.6 | ~109 ms | 5.80% |
| FP32, fixed 398 frames | 39.3-41.8 | ~78-84 ms | 5.80% |

BF16 is the fastest measured GPU path, about 1.4-1.7x faster than ORT FP32 on the
model segment; end-to-end the two are closer (70-90 ms vs 89-96 ms) because
fbank dominates and laptop clocks vary. The gain requires a fixed or bucketed
shape; the dynamic plan is slower than ORT. Engine builds took 50-172 s and
produced 544-990 MB files.

Setup:

1. Download the TensorRT 10.13.3 Windows CUDA 12.9 zip, extract it, and keep
   `lib/` available. The tested URL was
   `https://developer.download.nvidia.com/compute/machine-learning/tensorrt/10.13.3/zip/TensorRT-10.13.3.9.Windows.win10.cuda-12.9.zip`.
   The Python wheel is inside `python/`; this machine used
   `tensorrt-10.13.3.9-cp313-none-win_amd64.whl`.
2. Create the environment and install the Python dependencies:

   ```powershell
   python -m venv .native/trt-env
   .native/trt-env/Scripts/python.exe -m pip install -r tools/tensorrt-requirements.txt
   .native/trt-env/Scripts/python.exe -m pip install .native/tensorrt/TensorRT-10.13.3.9/python/tensorrt-10.13.3.9-cp313-none-win_amd64.whl
   ```

3. Prepare the official FP32 ONNX in the Windows benchmark environment. The
   baseline models are required for tokens and license files:

   ```powershell
   python tools/prepare_models.py
   python tools/prepare_low_precision.py --fp32
   ```

4. Build and run. The engine is built on first use and reused afterwards;
   `--frames` is the LFR frame count (398 for the 23.86 s clip). Pass
   `--tensorrt-lib` if the SDK `lib/` is not on `PATH`.

   ```powershell
   .native/trt-env/Scripts/python.exe tools/tensorrt_benchmark.py `
     --precision bf16 --frames 398 --build `
     --tensorrt-lib .native/tensorrt/TensorRT-10.13.3.9/lib
   ```

   Use `--frames 200` for 12 s, `400` for 24 s, `500` for 30 s, or `800` for
   48 s. Use `--precision fp16` or `fp32` for the other builder modes. The tool
   writes JSON with `--output`.

## transcribe.cpp (external engine)

The same 23.86 s clip was also run through handy-computer/transcribe.cpp
SenseVoiceSmall GGUFs (release 0.4.1) as an external cross-check. CPU used 12
threads with the CPU+Vulkan package's cascadelake AVX-512 module; CUDA used
CUDA0. All rows produced WER 5.80%.

| Model | CPU | CUDA | CUDA speedup |
| --- | ---: | ---: | ---: |
| Q8_0 | 1,255 ms | 125 ms | 10.0x |
| F16 | 1,353 ms | 137 ms | 9.9x |
| F32 | 1,365 ms | 152 ms | 9.0x |
| BF16 | 1,449 ms | 136 ms | 10.7x |

BF16 is not in the upstream catalog; `tools/convert_gguf_bf16.py` converts the
F16 GGUF's 281 F16 tensors to BF16 while keeping the 638 F32 tensors. The CUDA
wheel's bundled generic CPU backend was 6-12x slower than the CPU+Vulkan
package's AVX-512 module, so CPU numbers depend heavily on which ggml-cpu module
is loaded. transcribe.cpp CUDA Q8 (125 ms) is slower than GGUF CUDA Q8 (97 ms),
ORT FP32 (89-96 ms) and fixed-shape TensorRT BF16 (70-90 ms).

Setup:

1. Install the release 0.4.1 Python wheels from the transcribe.cpp GitHub
   release (`https://github.com/handy-computer/transcribe.cpp/releases/tag/v0.4.1`):
   `transcribe_cpp-0.4.1-py3-none-any.whl` and
   `transcribe_cpp_native_cu12-0.4.1-py3-none-win_amd64.whl`.
2. Download the SenseVoice GGUFs from
   `https://huggingface.co/handy-computer/SenseVoiceSmall-gguf`, for example
   `SenseVoiceSmall-Q8_0.gguf` (2.19 GB), `SenseVoiceSmall-F16.gguf` (4.09 GB),
   or `SenseVoiceSmall-F32.gguf`.
3. For the fast CPU path, download the
   `transcribe-native-0.4.1-windows-x86_64-cpu-vulkan.tar.gz` release asset and
   pass its `transcribe.dll` with `--library`. Without it, the CUDA wheel's
   generic CPU module is used and is much slower.

```powershell
python -m venv .native/transcribe-env
.native/transcribe-env/Scripts/python.exe -m pip install -r tools/transcribe-requirements.txt
.native/transcribe-env/Scripts/python.exe -m pip install `
  .native/transcribe-bin/transcribe_cpp-0.4.1-py3-none-any.whl `
  .native/transcribe-bin/transcribe_cpp_native_cu12-0.4.1-py3-none-win_amd64.whl
```

Run CUDA, CPU, or another backend:

```powershell
.native/transcribe-env/Scripts/python.exe tools/transcribe_benchmark.py `
  --model .native/models/transcribe/SenseVoiceSmall-Q8_0.gguf --device cuda
.native/transcribe-env/Scripts/python.exe tools/transcribe_benchmark.py `
  --model .native/models/transcribe/SenseVoiceSmall-Q8_0.gguf `
  --device cpu --threads 12 `
  --library .native/transcribe-bin/cpu/transcribe-native-windows-x86_64-cpu-vulkan/transcribe.dll
```

Create a BF16 GGUF with the Windows benchmark environment:

```powershell
.native/model-env/Scripts/python.exe tools/convert_gguf_bf16.py `
  .native/models/transcribe/SenseVoiceSmall-F16.gguf `
  .native/models/transcribe/SenseVoiceSmall-BF16.gguf
```

`--device` also accepts `vulkan` and `igpu`. The tool writes JSON with
`--output`; the model must be a 16 kHz mono WAV.

## GPU-only comparison

![GPU-only engine comparison](windows-gpu-engines.png)

End-to-end latency on the same 23.86 s clip. TensorRT uses a prebuilt 398-frame
BF16 engine; the dynamic-profile engine is shown for contrast and is slower than
ORT. transcribe.cpp includes its own mel/encoder/decoder. All rows produced WER
5.80%. TensorRT's advantage is the model segment (32-40 ms); fbank (~40-50 ms)
and laptop clocks dominate the total, so the end-to-end gap over ORT varies from
about 7% to 20%.

## Shape and chunk length

The model input is `x: [1, T, 560]`; each `T` step is one LFR frame, or 60 ms of
audio. `T` fixes the audio duration, the attention cost (quadratic in `T`), the
output size (`(T + 4) x 25055` floats) and which shapes a fixed engine accepts.

Fixed-shape BF16 TensorRT, engine-only:

| Audio | T | Engine ms | RTF | xRealtime |
| --- | ---: | ---: | ---: | ---: |
| 12 s | 200 | 23.2 | 0.0019 | 517x |
| 24 s | 400 | 34.8 | 0.0014 | 690x |
| 48 s | 800 | 56.3 | 0.0012 | 853x |
| 96 s | 1,600 | 162.2 | 0.0017 | 592x |

Padding the 23.86 s clip to 48 s kept WER at 5.80%; padding to 96 s raised it to
7.25%. The practical chunk is 20-30 s: near-best RTF, lower per-chunk latency,
and close to the training distribution. Use 48 s only when throughput matters;
avoid chunks beyond 60-96 s because attention cost and accuracy both degrade.

A direct 30 s (T=500) test, using isolated component medians:

| Audio | Engine ms | Engine RTF | fbank ms | fbank ms/s | Est. total ms/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 24 s | 34.1 | 0.00142 | 37.9 | 1.58 | 3.06 |
| 30 s | 37.4 | 0.00125 | 45.9 | 1.53 | 2.83 |
| 48 s | 55.9 | 0.00116 | 72.8 | 1.52 | 2.74 |

48 s is about 3% more efficient than 30 s and about 10% more than 24 s in
sequential end-to-end throughput. 30 s is not the absolute optimum, but it has
lower per-chunk latency and is closer to the training length. Combined CPU+GPU
runs on this laptop varied by 10-20% because the CPU and GPU share a power
budget; if throughput matters, pipeline fbank for chunk N+1 with inference for
chunk N instead of comparing sequential totals.
Feature extraction adds roughly 1.8 ms per audio second, so end-to-end RTF
differences between 24 s and 48 s are smaller than the engine-only table shows.
