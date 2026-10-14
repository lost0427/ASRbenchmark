# Windows CPU benchmark

Run on Windows 11 with Python 3.13.5, an AMD Ryzen AI 9 HX 370, four threads,
one warmup and three measured repetitions. The workload is the same 23.86-second
16 kHz LibriSpeech clip used by the Android benchmark. Results include feature
extraction and decoding. Only CPU backends were used.

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
