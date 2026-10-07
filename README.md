# SenseVoice Benchmark

Android app for comparing **CPU** SenseVoiceSmall inference with sherpa-onnx,
ONNX Runtime, MNN, ncnn and LiteRT. GPU support is planned for a later version.

The APK includes one fixed **23.86-second** LibriSpeech test-clean utterance,
`3729-6852-0008`, and its reference transcript:

> SHE WAS HONOURABLY BURIED IN THE CHURCH OF SAINT SAUVEUR WITHOUT THE SLIGHTEST OPPOSITION FROM THE VENERABLE PRIEST WHO FAR FROM SHARING THE ANTI CHRISTAIN INTOLERANCY OF THE CLERGY IN GENERAL SAID THAT HER PROFESSION AS AN ACTRESS HAD NOT HINDERED HER FROM BEING A GOOD CHRISTIAN AND THAT THE EARTH WAS THE COMMON MOTHER OF ALL HUMAN BEINGS AS JESUS CHRIST HAD BEEN THE SAVIOUR OF ALL MANKIND

All engines use English, ITN disabled and the same decoded PCM. Models are imported
separately and never bundled into Git or the APK. The benchmark runs offline.

## Use

1. Download the `sensevoice-cpu-arm64-debug` artifact from the repository's GitHub
   Actions page and install its APK on an arm64 Android 8.0+ phone.
2. Prepare models on a computer with Python 3.10+ and `requests`:

   ```powershell
   python -m pip install requests==2.32.3
   python tools/prepare_models.py
   ```

   For a smaller download, use `--engines onnx mnn` or select any of
   `onnx mnn ncnn litert`. The ONNX folder supports both sherpa-onnx and direct ORT.

3. Copy the generated `models/cpu-models` folder to the phone, including
   `manifest.json` and its subdirectories. Tap **Import model folder** and choose
   that folder. Import checks sizes and SHA-256 before activating the new models.
4. Choose engines, threads, warmup and repeats, then tap **Run benchmark**.
   Defaults: 2 threads, 3 warmups, 10 measured repetitions. The first inference
   is measured separately, before warmup.
5. Export JSON for full configuration, model provenance and individual samples;
   export CSV for a compact comparison. Results also remain in app storage.

Model import needs temporary room for the new model set and any previously
installed set. Keep the app open during import and benchmarking. Stop waits for
the current native inference to finish, then saves the partial report.

## Models

| Entry | CPU model | Runtime |
| --- | --- | --- |
| sherpa-onnx | Official dynamic-quantized ONNX | sherpa-onnx 1.13.8 + ORT 1.30.0 |
| ONNX Runtime | The same ONNX and tokens | ORT 1.30.0 CPU EP |
| MNN | PocketASR Q8, block 64 | MNN 3.6.1 through sherpa-mnn |
| ncnn | Official ncnn INT8 export | ncnn through sherpa-ncnn |
| LiteRT | `dynamic_wi8_afp32` | LiteRT 2.1.6, XNNPACK requested |

Quantized weights do not imply every activation or operator uses INT8.
Quantization schemes are shown in the app and report. MNN and ncnn include
their Sherpa wrappers' front end and decoder. See [model sources](docs/MODEL_SOURCES.md)
for links, pinned revisions and conversion details.

## Measurement

- Latency covers **PCM to text**: features, tensor preparation, inference and
  decoding. It excludes importing models, reading the WAV and UI updates.
- Load time and first inference are separate from steady repetitions.
- Mean, median, nearest-rank P90, min, max, RTF and raw samples are saved.
- PSS is sampled with a 100 ms delay between reads across loading and recognition. A sampled peak
  may miss short allocations and includes the whole process, including already
  loaded runtimes. It is not isolated model memory.
- CER/WER use uppercase ASCII letters/digits, punctuation removal and collapsed
  whitespace. CER excludes spaces. The single clip checks output consistency;
  it does not establish corpus accuracy.
- LiteRT uses the smallest signature bucket: 500 LFR frames for this clip. Other
  paths use dynamic length. Record this extra padding when comparing latency.
- Language is English and ITN is disabled. No VAD or punctuation model runs.
- Core-only inference time is unavailable; it is never substituted with
  end-to-end latency. Requested thread counts do not guarantee identical
  scheduling across runtimes.

Keep thermal conditions consistent when comparing runs. The report includes
device information, thermal status where available, battery temperature and
execution order.

## Results

Measured on a OnePlus 15 (Snapdragon 8 Elite Gen 5, SM8850), Android 16, 4
threads, 3 warmups and 10 measured repetitions on the fixed clip above.
Thermal status 0 throughout; battery 32.6-33.1 °C.

| Engine | Status | Mean ms | Median ms | P90 ms | RTF | WER | CER | Peak PSS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sherpa-onnx | ok | 489.8 | 488.6 | 501.2 | 0.0205 | 5.80% | 3.38% | 810 MB |
| ONNX Runtime | ok | 530.1 | 530.3 | 534.3 | 0.0222 | 5.80% | 3.38% | 890 MB |
| LiteRT | ok | 658.4 | 650.8 | 708.0 | 0.0276 | 5.80% | 3.38% | 1611 MB |
| MNN | ok | 885.3 | 912.0 | 966.3 | 0.0371 | 5.80% | 2.77% | 1120 MB |
| ncnn | ok | 1691.7 | 1692.0 | 1711.7 | 0.0709 | 13.04% | 7.38% | 717 MB |

sherpa-onnx and direct ORT share the ONNX model and produce identical text.
LiteRT pads this clip to the 500-frame signature bucket. ncnn is the slowest and
has the highest error rate; MNN has the lowest CER. These numbers cover one clip
and one device, so they are a latency/consistency check, not a corpus benchmark.

## Build

Requirements: JDK 17+, Android SDK 35, NDK `28.2.13676358`, CMake `3.22.1`,
Ninja, Git, Python and `requests==2.32.3`.

Windows:

```powershell
$env:ANDROID_HOME = "$env:LOCALAPPDATA\Android\Sdk"
$env:PATH = "$env:ANDROID_HOME\cmake\3.22.1\bin;$env:PATH"
python tools/build_native.py --sdk $env:ANDROID_HOME
.\gradlew.bat assembleDebug
```

Linux / GitHub Actions:

```sh
export PATH="$ANDROID_HOME/cmake/3.22.1/bin:$PATH"
python tools/build_native.py --jobs 2
./gradlew assembleDebug
```

The build script stages four JNI bridges and the shared C++ runtime into
`app/src/main/jniLibs/arm64-v8a`. Downloaded dependency sources and model weights stay in ignored
directories. The ORT AAR supplies the shared ONNX runtime for both ONNX paths.
Native dependency archives are pinned in `tools/build_native.py`; upstream CMake files
also pin their transitive downloads. GPU backends are disabled.
Lint is pinned to 8.13.2 to support the Lifecycle 2.10 checks brought in by LiteRT.

Publish the prepared local repository with GitHub Desktop's **Publish repository**
button. Keep the `.github/workflows` files enabled; the first push starts the APK
build. No repository URL, access token or local SDK path is committed.

[GitHub Actions](.github/workflows/android-build.yml) builds the native runtimes,
APK and lint report on push, pull request or manual dispatch. APK artifacts are
retained for 14 days. This first version distributes a debug-signed APK; release
signing is a subsequent packaging task. CI does not run device performance tests.

Local validation (2026-10-06): CPU native libraries built on Windows, and
`assembleDebug lintDebug` completed successfully. Phone inference verified on a
OnePlus 15 (Snapdragon 8 Elite Gen 5); see [Results](#results). The first GitHub
Actions run remains to be verified after publication.

## Fixed audio provenance

The supplied `test-clean.tar.gz` is kept locally and ignored by Git. To regenerate
the selected WAV with FFmpeg installed:

```powershell
python tools/prepare_audio.py test-clean.tar.gz
```

The script extracts only the selected FLAC, chapter transcript and dataset
license; no dataset-wide unpacking is needed. It preserves the complete utterance
without trimming, denoising or loudness normalization.

Source: [LibriSpeech / OpenSLR 12](https://www.openslr.org/12/), licensed under
CC BY 4.0. The source FLAC is decoded to PCM WAV. Attribution, reference, audio
hash and license are stored in `app/src/main/assets/audio` and carried into reports.
Model and runtime licenses are supplied alongside their downloaded files.

See [PLAN.md](PLAN.md) for CPU milestones and the later GPU scope.
