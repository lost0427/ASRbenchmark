# SenseVoice Model Sources

查询日期：2026-10-06。本文记录在线发布资产、模型文件清单、模型卡及框架源码调查结果。`已找到`表示发布方列出了文件；尚未下载权重、计算本地 SHA-256、检查完整模型图或进行 Android 真机验证。

## 1. sherpa-onnx 与 ONNX Runtime：INT8 CPU

优先共用 k2-fsa 的原始 SenseVoiceSmall 2024-07-17 导出，直接比较封装路径：

- [官方 INT8 下载包](https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17.tar.bz2)。
- [官方浮点下载包](https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2)。
- [Hugging Face 文件目录](https://huggingface.co/csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/tree/main)，已列出 `model.int8.onnx`、`model.onnx`、`tokens.txt`、`export-onnx.py` 和五语音频。

CPU 后端使用 ONNX Runtime CPU EP。模型内的量化算子、浮点保留算子及动态／静态量化方式，实施时检查导出脚本和模型图后登记。sherpa-onnx 端到端计时包含其原生前处理和解码；直接 ORT 路径使用对齐的前处理参数。未插桩获得内部执行时间时，sherpa-onnx 核心推理耗时标为 unavailable。

## 2. sherpa-onnx 与 ONNX Runtime：FP16 GPU

已找到 [社区 FP16 ONNX](https://huggingface.co/ruska1117/SenseVoiceSmall-onnx-fp16/tree/main)，模型卡说明源于 `iic/SenseVoiceSmall`，需要 `model.onnx` 与 `model.onnx.data`，以及 `am.mvn`、`config.yaml`、`tokens.json`。查询时 revision：`03a199168d32793c49683ef4966b156714314858`。

该导出不保证兼容 sherpa-onnx 的张量接口与 metadata。用于候选图研究；正式对比优先从第一节同源 FP32 图转换，并保持 sherpa-onnx 所需 metadata、输入输出接口和词表。

Android GPU 路线目前需要验证：

1. 接入 ONNX Runtime NNAPI EP，评估显式 FP16 图与 FP32 图的 FP16 relaxation 两种方式；报告分别命名。
2. 查询实际 NNAPI 设备类型和执行分配；仅设置 `USE_FP16` 或禁用 NNAPI CPU 设备都不能证明使用 GPU，因为 NPU 也可能被选中，ORT 本身仍可能执行未委托算子。
3. 如果无法确认 GPU 分配，显示 `NNAPI / accelerator unknown`，不计入 FP16 GPU 对比。
4. 当前 sherpa-onnx [session.cc](https://github.com/k2-fsa/sherpa-onnx/blob/master/sherpa-onnx/csrc/session.cc) 的 NNAPI 分支设 flags 为 0，FP16 与 CPU_DISABLED 设置被注释；实施时需固定源码版本、增加配置及日志，再构建 Android JNI。
5. 框架 NNAPI 算子清单并不覆盖任意 SenseVoice 图；需评估 attention、动态 shape 等造成的分区和 CPU 回退。无法满足要求时报告不支持，而不是用桌面 CUDA 结果填充。

参考：[ONNX Runtime NNAPI EP](https://onnxruntime.ai/docs/execution-providers/NNAPI-ExecutionProvider.html)。NNAPI 需要 API 27+，CPU_DISABLED 需要 API 29+；低系统版本保留 CPU 模式。

## 3. MNN：模型候选与两种目标配置

已找到 [xinliu/sensevoice_mnn](https://huggingface.co/xinliu/sensevoice_mnn/tree/main)：`sensevoice.mnn`、`tokens.txt`。查询时 revision：`7ca517021b1a780246859989f2dd12c1f1db3520`。

模型卡没有标明权重类型、量化配置或 GPU 支持；所链接的 Android 项目 `xinliu9451/sensevoice_android_mnn` 在本次查询返回 404。因此该包仅作候选，不能认定为 INT8 或 FP16，也不引用其速度宣传作为 benchmark 结论。

- **INT8 CPU**：从第一节 FP32 ONNX 转换为 MNN，检查算子覆盖；再用固定版本 MNN 工具量化。静态量化需要可复查的语音校准集；仅权重压缩时明确标记 weight-only，不能称作 W8A8。检查 MatMul/Gemm 等是否真正进入 INT8 CPU 路径。
- **FP16 GPU**：同源浮点 MNN 图，明确指定 `MNN_FORWARD_OPENCL`，使用 `BackendConfig::Precision_Low`；Vulkan 作为另一个明确标注的 GPU 配置。检查设备 FP16 能力、执行精度与 CPU 回退。

参考：[MNN](https://github.com/alibaba/MNN)、[后端和精度配置源码](https://github.com/alibaba/MNN/blob/master/include/MNN/MNNForwardType.h)。目前未找到能够从模型卡确认精度的 SenseVoice MNN INT8 CPU／FP16 GPU 成品包。

## 4. ncnn：INT8 CPU 与 FP16 GPU 模型均已找到

k2-fsa 已发布三种同日期 SenseVoice ncnn 包：

- [INT8 模型](https://github.com/k2-fsa/sherpa-ncnn/releases/download/asr-models/sherpa-ncnn-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17.tar.bz2)。
- [FP16 模型](https://github.com/k2-fsa/sherpa-ncnn/releases/download/asr-models/sherpa-ncnn-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2)。该名称不含 fp16，精度依据下述导出工作流确认。
- [FP32 参考模型](https://github.com/k2-fsa/sherpa-ncnn/releases/download/asr-models/sherpa-ncnn-sense-voice-zh-en-ja-ko-yue-float32-2024-07-17.tar.bz2)。

[官方导出工作流](https://github.com/k2-fsa/sherpa-ncnn/blob/master/.github/workflows/export-sense-voice.yaml) 使用原始 FunAudioLLM 权重、`export-ncnn.py --fp16 1/0`，并调用 `ncnn2int8` 产生 INT8 包，输出 `model.ncnn.param`、`model.ncnn.bin`、`tokens.txt`。不同日期包可能使用不同 checkpoint，不能混用。

- **INT8 CPU**：关闭 Vulkan，启用 `use_int8_inference`，检查实际量化层；工作流传入空校准表的行为需核对对应工具版本和图，不能宣称所有算子全 INT8。
- **FP16 GPU**：启用 Vulkan，按设备能力设置 `use_fp16_storage`、`use_fp16_packed` 与 `use_fp16_arithmetic`。FP16 权重文件本身不能证明 FP16 GPU 算术；必须检查 GPU 能力和各层 Vulkan 支持。
- Android 接入可参考 sherpa-ncnn 的原生前处理与模型接口，但应用中保留用户指定的 `ncnn` 名称，并报告封装版本。

参考：[ncnn 配置源码](https://github.com/Tencent/ncnn/blob/master/src/option.h)、[另一个 SenseVoice ncnn 示例](https://github.com/FeiGeChuanShu/FunASR-demo-ncnn)。

## 5. LiteRT：已找到动态 INT8 CPU，FP16 GPU 需要导出

已找到 [Luigi/sensevoice-litert](https://huggingface.co/Luigi/sensevoice-litert/tree/main)：`sensevoice_small_q8.tflite`、`cmvn.json`、`tokens.txt`。查询时 revision：`8d1dfd033bafc5d40e03cb1dc714134386eb6895`。

模型卡声明从 FunAudioLLM/SenseVoiceSmall 经 litert-torch 导出，并使用 ai-edge-quantizer `dynamic_wi8_afp32`。这是动态 INT8 权重量化、FP32 激活模式，不能与静态全 INT8 混为一谈。发布方报告了主机／RPi4 CPU 验证，本项目尚未复现。

- **INT8 CPU**：固定满足该模型要求的 LiteRT 版本（模型卡写明 ≥2.1.6），启用 XNNPACK，确认量化算子被执行。
- **输入契约**：多 signature `sv_63`、`sv_125`、`sv_250`、`sv_500`；特征 `[1,T,560]`、真实长度、language ID、textnorm ID；输出含 4 个 prompt 帧的 CTC logits。bucket 约覆盖 3.8、7.5、15、30 秒音频。
- **公平比较**：尽可能为其他框架匹配 bucket 和 padding；同时保存真实音频时长、真实帧数和计算帧数。更长音频统一分段，记录策略。
- **FP16 GPU**：尚未找到已确认的成品。优先从同源 PyTorch 经 litert-torch 导出浮点图，再用兼容的 FP16 权重量化／转换路线生成 `.tflite`；固定 shape/bucket 以评估 GPU delegate 算子支持。若工具链不能直接生成所需图，记录阻碍并继续处理转换，不把 q8 包改名为 fp16。
- 采用独立打包的 LiteRT GPU delegate，检查 FP16 计算设置、分区和 CPU 回退；创建、运行、释放保持同一线程。INT8 模型被 GPU 解量化执行不能作为 INT8 CPU 测量，也不能据此证明已有 FP16 GPU 模型。

参考：[模型卡与详细契约](https://huggingface.co/Luigi/sensevoice-litert)、[LiteRT Android GPU](https://ai.google.dev/edge/litert/android/gpu)、[litert-torch](https://github.com/google-ai-edge/litert-torch)。

## 6. 实施时应补齐的记录

每个配置保存模型 SHA-256、来源 revision、框架版本、转换脚本与命令、量化类型、实际执行后端、GPU 型号、算子覆盖或回退信息、语言／ITN 参数、音频特征配置、shape/bucket 与识别质量。

首先采用已找到的官方 ONNX INT8 和 ncnn INT8/FP16 包，再核验 LiteRT 动态 INT8 与 MNN 候选，补齐 MNN 目标精度转换及 LiteRT FP16 导出。sherpa-onnx／ORT 的 GPU 路线需要单独验证，不能把 NNAPI、NPU 或 CPU 回退结果标成 GPU。
