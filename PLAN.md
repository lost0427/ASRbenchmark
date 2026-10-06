# SenseVoice Android Benchmark Plan

## 1. 目标与范围

制作一个在 Android 真机上运行的 SenseVoice benchmark 应用，比较不同推理框架、模型精度与执行配置下的性能，并通过 GitHub Actions 自动构建 APK。

- 应用名称：SenseVoice Benchmark。
- 首版面向 arm64-v8a 手机，建议最低 Android 8.0（API 26）；最终版本要求依据依赖兼容性确定。
- 本地导入模型与音频，离线完成推理和结果查看。
- 使用简洁的英文界面与英文 Git 提交标题。
- 对比入口固定为 sherpa-onnx、直接调用 ONNX Runtime、MNN、ncnn、LiteRT 五项，每项寻找并接入 INT8 CPU 与 FP16 GPU 两种配置。
- sherpa-onnx 与直接调用 ONNX Runtime 单独显示，用于比较封装、前后处理与调度开销；报告同时注明两者共用 ONNX Runtime 计算内核。
- 本文件为实施计划；当前阶段仅提交计划，不创建应用代码。

## 2. 推理框架与精度矩阵

| 对比入口 | INT8 CPU 模型来源 | FP16 GPU 模型与后端路径 | 当前证据 |
| --- | --- | --- | --- |
| sherpa-onnx | k2-fsa 官方 SenseVoice INT8 ONNX 包 | 同源浮点 ONNX 转换或 FP16 relaxation，接入 NNAPI；必须确认 GPU 设备 | INT8 包已找到；Android FP16 GPU 未验证，可能需要修改原生构建 |
| ONNX Runtime | 与 sherpa-onnx 共用官方 `model.int8.onnx` | 同源 ONNX 转 FP16，或 FP32 图加 NNAPI FP16 relaxation；确认 GPU 设备 | INT8 包已找到，社区 FP16 ONNX 已找到；NNAPI 不保证 GPU |
| MNN | PocketASR 已发布同源 Q8 `model.mnn`，8 位权重量化 | 同源浮点模型转 MNN，OpenCL 优先、Vulkan 备选，低精度模式 | Q8 成品与转换记录已找到；PocketASR 仅支持 CPU，GPU 使用独立运行时 |
| ncnn | k2-fsa 官方 SenseVoice INT8 ncnn 包 | k2-fsa 官方 FP16 ncnn 包，Vulkan 与 FP16 存储／算术配置 | 两种模型包与导出工作流均已找到；Android GPU 算子覆盖待验证 |
| LiteRT | 社区 `sensevoice_small_q8.tflite`，CPU/XNNPACK | 从同源 PyTorch 经 litert-torch 导出，生成 FP16 权重 `.tflite`，接入 GPU delegate | 动态 INT8 包已找到；未找到已确认的 SenseVoice FP16 GPU 成品 |

模型链接、证据和转换要求见 [docs/MODEL_SOURCES.md](docs/MODEL_SOURCES.md)。现阶段核实了在线模型清单、发布资源、说明与源码，没有下载大型权重或执行真机推理。

五项均纳入实施范围，MNN、ncnn、LiteRT 不互相替代。为十个目标组合建立状态：`source_found`、`needs_conversion`、`needs_device_validation`、`verified`、`unsupported`。目标是完成全部可行组合；确有算子或后端障碍的组合保留入口、证据和原因，不隐藏或冒充可运行结果。

FP32 作为识别质量参考。优先锁定 2024-07-17 FunAudioLLM SenseVoiceSmall 路线，避免与 2025-09-09 WSYue-ASR 路线混用。不同导出图和静态 bucket 的性能差异需要记录。

精度记录规则：

- 分开记录权重存储类型、请求的计算精度、量化方式与执行后端。
- FP16 权重不保证所有算子使用 FP16；INT8 模型可能混合执行浮点算子。
- 区分动态 INT8、权重量化和静态 W8A8；LiteRT 已找到的包明确为 `dynamic_wi8_afp32`，不能标为全 INT8。
- FP16 GPU 必须记录实际 GPU 后端与 CPU 回退；NNAPI 的 GPU、NPU 和 CPU 分配分别标注，未确认 GPU 的结果不进入 GPU 排名。
- 无法确认实际算子执行精度时标记为 unknown，不作推断。
- 后端回退需要记录；无法检测的回退明确标记为未验证。
- 不支持的组合显示具体原因，不返回零耗时或伪造结果。

## 3. 技术方案

- Android：Kotlin、Jetpack Compose、单 Activity 架构。
- 任务执行：协程在后台顺序执行，同一时刻只运行一个 benchmark。
- 推理接口：统一的 `InferenceEngine`，包含模型加载、预热、单次执行和资源释放。
- 原生接入：sherpa-onnx Android/JNI、ONNX Runtime Android API、MNN/ncnn JNI 与 CMake、独立打包的 LiteRT 与 GPU delegate；首版无需 Google Play 服务。
- GPU 任务固定到专用线程，保证 delegate 创建、运行与释放的线程要求；计时包含实际完成与同步。
- 前处理与解码：统一音频读取、特征参数、token 词表和文本规范化；可共用的部分只实现一份。
- 文件访问：使用 Android Storage Access Framework 导入模型与音频、导出报告。
- 数据保存：应用私有目录存储模型清单和 benchmark 历史，首版采用结构化 JSON。
- 版本管理：固定 Gradle、JDK、Android SDK、NDK 与推理依赖版本，提交 Gradle Wrapper。

建议目录：

```text
app/                         Compose 界面、任务配置、结果展示
benchmark/                   指标计算、运行调度、报告结构
inference/                   统一接口与五项适配
native/                      sherpa-onnx、MNN/ncnn JNI 与原生构建配置
tools/model-conversion/      模型导出、转换、量化说明与脚本
docs/                        模型兼容性、使用方法、测量定义
.github/workflows/           APK 自动构建与发布工作流
```

## 4. 模型与音频输入

- 以同一来源、同一版本的 SenseVoiceSmall 为起点，登记来源、许可证、版本及 SHA-256。
- 为每种框架准备独立模型文件，但保持相同参数来源、词表、语言参数与音频处理设置。
- 模型清单描述框架、文件路径、精度、输入输出名称、张量形状、前处理参数与校验和。
- 清单增加模型来源 revision、量化类型、GPU backend、转换版本和支持的音频长度；外部 ONNX 权重、ncnn param/bin、词表与 CMVN 一起导入。
- 量化转换记录工具版本、命令与量化配置；需要校准时记录校准集来源及规模。
- 不把大型模型提交进 Git；首版通过文件导入，后续可增加带校验和的下载入口。
- 首版支持 16 kHz 单声道 PCM WAV。其他格式或采样率显示明确提示，后续再增加解码与重采样。
- 提供许可证允许分发的小型示例音频；用户也可导入自己的样本与参考转写。
- 性能对比使用相同音频集；有参考转写时同时评估精度，防止仅比较速度而忽略识别退化。

## 5. Benchmark 方法与指标

默认配置为预热 3 次、正式重复 10 次，用户可调整线程数、预热次数和重复次数。批量运行按模型配置顺序执行，并记录执行顺序；可增加交错或随机顺序以观察温升影响。

| 指标 | 定义与测量方式 |
| --- | --- |
| 模型加载耗时 | 从初始化引擎到模型可执行，不包含用户导入文件 |
| 首次推理耗时 | 模型加载后首次识别，单独记录，不混入预热后统计 |
| 核心推理耗时 | 引擎执行张量计算的耗时，明确是否包含张量拷贝；异步后端需要同步后计时 |
| 端到端耗时 | 从已解码 PCM 开始，包含特征提取、推理与文本解码 |
| 延迟统计 | 正式执行的平均值、中位数、P90、最小值、最大值及原始样本 |
| RTF | 端到端识别耗时 / 音频时长，越低越快 |
| 内存 | 加载前、加载后及运行中采样的进程 PSS，记录采样间隔；采样峰值不等于精确分配峰值 |
| 模型大小 | 模型相关文件的总字节数 |
| 识别质量 | 有参考转写时计算 CER/WER，并记录语言与规范化规则 |
| 设备与环境 | 手机型号、SoC（可获取时）、Android 版本、ABI、线程数、后端、框架版本、温度或热状态、电量与充电状态 |

使用单调时钟计时，预热和正式执行分开。CPU 同线程数对比作为基础；GPU、NNAPI、Vulkan 等按框架支持单独列出，不与 CPU 结果混合汇总。首版不提供缺乏可靠测量依据的功耗数字。

运行失败、取消、内存不足和热状态异常均保留状态。支持在执行间隙取消；无法中断的原生推理等待当前调用返回后释放资源。运行期间保持屏幕唤醒，长音频与重复任务允许使用前台服务。

## 6. 应用界面

1. **Models**：导入模型包，查看框架、精度、大小与可用状态。
2. **Benchmark**：选择模型组合与音频，配置线程和重复次数，开始或取消任务。
3. **Results**：表格比较耗时、RTF、内存、识别文本与 CER/WER，查看单次执行记录。
4. **Export**：导出包含配置和设备信息的 JSON，以及便于整理的 CSV。

界面保持简单，以模型选择、开始测量、结果比较为主要操作。执行进度显示当前配置、音频和完成次数，错误提示包含可采取的修复操作。

## 7. GitHub 自动构建

- `android-build.yml`：由 push、pull request 和手动触发运行。
- 在 GitHub 托管 Linux runner 上安装固定 JDK、SDK 和 NDK，配置 Gradle 缓存。
- 运行项目静态检查、单元测试与 `assembleDebug`，上传 APK、构建日志及必要报告。
- 默认 APK 为可直接安装的 debug 产物，保留期在工作流中明确设置。
- `android-release.yml`：由版本标签触发构建 Release APK，并创建 GitHub Release。
- Release 签名通过 GitHub Secrets 提供；签名密钥不进入仓库或构建日志。未配置签名时仅产出明确标记的未签名文件。
- 工作流只验证构建与逻辑；真机性能 benchmark 在手机上执行，不用 CI 模拟器代替性能结论。
- 启用工作流需要将仓库推送到 GitHub；当前本地仓库未配置远程地址，实施阶段补齐。

## 8. 实施阶段与验收

### 阶段一：模型可行性

确定同源 SenseVoiceSmall，下载并校验已找到的 ONNX、ncnn、LiteRT 与 PocketASR MNN Q8 包。逐项核对五个入口的 INT8 CPU 和 FP16 GPU 路线，完成缺失模型的导出／量化方案与 GPU 算子覆盖调查。GPU 优先使用 ncnn 现有 FP16 模型和 MNN 现有 GPU benchmark 工具进行可行性测量，再接入应用。输出十组合清单及转换说明。

验收：十个组合均有具体模型来源或可复现转换步骤；缺失或不兼容项记录证据，已有模型与转换模型注明实际量化方式，不能仅凭文件名认定精度。

### 阶段二：Android 基线应用

创建 Gradle 工程、Compose 页面、模型和 WAV 导入、统一引擎接口，先接入 sherpa-onnx 与直接调用 ONNX Runtime 的 INT8 CPU，使用同一模型，完成单配置 benchmark 与 JSON 导出。

验收：arm64 Android 真机可安装、离线识别、显示有效时间与 RTF，并能导出可复查的数据。

### 阶段三：多框架与多精度

接入 MNN、ncnn、LiteRT，补齐五个入口的 INT8 CPU 与 FP16 GPU 配置、批量比较、质量指标与 CSV 导出。增加 GPU 类型、委托覆盖、CPU 回退与 bucket 长度展示。

验收：五个入口均能呈现十组合状态；可行组合有真机输出与性能记录，不可行组合有明确原因。INT8 CPU 与 FP16 GPU 分表比较，未证实 GPU 执行的结果不计入 GPU 排名。

### 阶段四：自动构建与交付

完成 GitHub Actions、APK artifact、标签发布流程和使用文档。补充指标计算、错误处理与报告序列化的单元测试，并执行真机检查。

验收：GitHub push 自动生成可安装 APK；文档说明模型准备、运行方法、结果含义与已知限制。Release 发布在具备签名配置时提供已签名 APK。

## 9. 提交约定

每个可独立审阅的阶段使用简洁英文提交标题，例如：

- `Add benchmark plan`
- `Create Android app`
- `Add ONNX Runtime inference`
- `Add sherpa-onnx inference`
- `Add MNN inference`
- `Add ncnn inference`
- `Add LiteRT inference`
- `Add precision benchmarks`
- `Add GitHub APK builds`

先完成模型兼容性结论，再确定最终框架与精度支持列表。所有性能结论附设备与运行配置，避免将不同条件下的结果直接等同。
