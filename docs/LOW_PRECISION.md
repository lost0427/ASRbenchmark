# Low-precision SenseVoice benchmark

The APK uses official ONNX Runtime 1.30.0 from Maven. The custom ORT build
experiment is retired. The baseline/aggressive APK flavors describe the existing
native CPU builds; both use the same official ORT and the same GGUF bridge.

Models stay outside the APK. One manifest-based folder contains every supported
variant, and each variant can be selected independently. JSON schema 2 records
`buildVariant`, `ortDistribution`, `engineId`, `modelVariant`, model provenance,
file sizes and SHA-256 hashes. Older Q8-only model folders still import.

## Test matrix

| Engine | Included variants | Quantization scope |
| --- | --- | --- |
| sherpa-onnx and direct ORT | Q8, Q4, Q2 | Q8 is the official dynamic model; Q4/Q2 quantize constant MatMul weights with MatMulNBits, block 64 |
| MNN | Q8, Q7, Q6, Q5, Q4, Q3, Q2 | Weight-only, block 64, floating activations; requested bit width does not establish native arithmetic width |
| ncnn | INT8, FP16 storage | Official 2024-07-17 exports; CPU only |
| LiteRT | Q8, Q4, Q2 | Q4/Q2 requantize the original W8 fully-connected weights; depthwise weights remain W8 and activations remain FP32 |
| SenseVoice.cpp / GGML | Q8_0, Q6_K, Q5_0, Q5_K, Q4_0, Q4_1, Q4_K, Q3_K, FP16 | Mixed tensor types; actual types are counted in provenance |

INT2/INT3 ncnn exports are unavailable in this integration. The pinned GGUF
quantizer requires an importance matrix for Q2_K and IQ1/IQ2; those conversions
are not included. No unsupported configuration is labeled as a working model.

The LiteRT Q4/Q2 models are derived from W8 rather than original floating weights.
That extra quantization step is disclosed in their provenance and precision labels.
No calibration dataset is used for these weight-only conversions.

## Prepare models on the computer

Use Python 3.13 and a separate virtual environment:

```powershell
python -m venv .native/model-env
.native/model-env/Scripts/python.exe -m pip install -r tools/model-requirements.txt
.native/model-env/Scripts/python.exe tools/prepare_models.py --downloader requests
.native/model-env/Scripts/python.exe tools/prepare_low_precision.py
```

The result is `models/low-precision-models`, approximately 5.03 GB for 24 model
folders and 27 selectable engine/precision combinations. The two ONNX engines
share each ONNX model. The preparation script checks the source floating ONNX
hash, pinned GGUF hashes, and writes hashes for every importable file.
On Windows it downloads a pinned FlatBuffers compiler to generate LiteRT 2.1.6's
schema; other platforms need `flatc` on PATH.

Copy the whole folder, including `manifest.json`, to the phone. Install either
APK and select **Import model folder**, then **Select imported** or choose a
smaller group of variants. Set identical threads, warmups and repeats, run the
benchmark, and export JSON. Running every variant sequentially increases thermal
drift; individual engine groups are easier to compare across repeated runs.

The new APK package IDs begin with `org.sensevoice.benchmark.precision`, allowing
installation alongside the earlier benchmark and custom ORT experiment apps.

## Desktop checks and Android build

```powershell
python tools/build_gguf.py --sdk "$env:LOCALAPPDATA/Android/Sdk"
.\gradlew.bat assembleDebug lint
```

The usual `tools/build_native.py` also builds the GGUF bridge, including in CI.
`build_gguf.py --host` builds `benchmark_gguf_smoke` with the selected host C/C++
compiler. It uses the same adapter as Android, runs the whole WAV twice, disables
VAD and debug dumps, selects English, and disables ITN.

```powershell
.native/model-env/Scripts/python.exe tools/check_low_precision_models.py
```

Desktop checks verify manifest hashes, required ONNX metadata, loading and two
consistent recognition passes. ncnn still requires an Android run. Desktop
recognition establishes compatibility and a one-clip output check; phone latency
must come from the APK report. Empty output is retained as an accuracy failure,
and its inference latency can still be measured in the APK.

On the fixed clip, desktop checks found substantial degradation for ORT Q2,
MNN Q3/Q2 and GGUF Q3_K; LiteRT Q2 produced empty text. These variants remain
available as experiments. Q4 generally retained much more of the Q8 output.
These checks are not a corpus accuracy evaluation.

## Pinned sources

- Floating ONNX: `csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17`,
  revision `2365baeacb507f821a0c8120fcee3d484dba7a07`, SHA-256
  `977016bd9c79f9eb343430b5cc305e07ab64d5212dff41b0dcfa1694bee9a8cb`.
- MNN converter and runtime: 3.6.1.
- ORT quantizer and Android runtime: 1.30.0.
- LiteRT runtime/schema: 2.1.6; W8 model revision
  `8d1dfd033bafc5d40e03cb1dc714134386eb6895`.
- GGUF models: `lovemefan/sense-voice-gguf`, revision
  `c3dc3564b1795b03f4c2cfa8549af5797faca4ad`.
- SenseVoice.cpp: v1.4.0, commit `c78e6919351ac83255e96de46169f518097f1ef3`.
- GGML: submodule commit `a5960e80d3e65ce6ff18f90315ab96f63cf9c4cc`.

The adapter releases per-inference feature buffers that upstream replaces, so
repeated benchmarking does not accumulate those allocations. The source overlay
omits the optional VAD allocator, because these published GGUF files lack Silero
weights and the benchmark recognizes a complete, untrimmed fixed clip.
