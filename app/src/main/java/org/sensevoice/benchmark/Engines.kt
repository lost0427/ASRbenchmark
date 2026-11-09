package org.sensevoice.benchmark

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.providers.NNAPIFlags
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import org.json.JSONObject
import org.tensorflow.lite.Interpreter
import java.io.Closeable
import java.io.File
import java.nio.FloatBuffer
import java.nio.IntBuffer
import java.util.EnumSet

object SherpaNative {
    init { System.loadLibrary("benchmark_sherpa") }
    external fun create(model: String, tokens: String, threads: Int, provider: String): Long
    external fun recognize(handle: Long, samples: FloatArray): String
    external fun destroy(handle: Long)
}
object MnnNative {
    init { System.loadLibrary("benchmark_mnn") }
    external fun create(model: String, tokens: String, threads: Int, provider: String): Long
    external fun recognize(handle: Long, samples: FloatArray): String
    external fun destroy(handle: Long)
}
object NcnnNative {
    init { System.loadLibrary("benchmark_ncnn") }
    external fun create(model: String, tokens: String, threads: Int, vulkan: Boolean, fp16: Boolean): Long
    external fun recognize(handle: Long, samples: FloatArray): String
    external fun deviceInfo(): String
    external fun destroy(handle: Long)
}
object FeatureNative {
    init { System.loadLibrary("benchmark_features") }
    external fun fbank(samples: FloatArray): FloatArray
}
object GgufNative {
    init { System.loadLibrary("benchmark_gguf") }
    external fun create(model: String, threads: Int): Long
    external fun recognize(handle: Long, samples: FloatArray): String
    external fun destroy(handle: Long)
}

data class EngineSpec(
    val id: String,
    val title: String,
    val folder: String,
    val precision: String,
    val runtime: String,
    val modelFile: String = "",
    /** Backend selector passed to the bridge: cpu, nnapi, vulkan, opencl, opencl-fp32, vulkan-fp32, gpu. */
    val backend: String = "cpu",
    /** GPU compute precision toggle (FP16 arithmetic when true). */
    val fp16: Boolean = false,
    val accelerator: String = "CPU",
    val compute: String = "",
    val nnapiFp16: Boolean = false,
    /** Excluded from "Run all" / "Select imported"; still manually selectable. */
    val experimental: Boolean = false,
) {
    val engineId get() = id.substringBefore('-')
    val variant get() = id.substringAfter('-', "q8")
    val label get() = "$title · $variant"
    val requiredFiles get() = when (engineId) {
        "gguf" -> listOf(modelFile)
        "ncnn" -> listOf("model.ncnn.param", "model.ncnn.bin", "tokens.txt")
        "litert" -> listOf(modelFile.ifEmpty { "sensevoice_small_q8.tflite" }, "tokens.txt", "cmvn.json")
        "mnn" -> listOf("model.mnn", "tokens.txt")
        else -> listOf(modelFile.ifEmpty { "model.int8.onnx" }, "tokens.txt")
    }
}

val cpuEngineSpecs = listOf(
    EngineSpec("sherpa", "sherpa-onnx", "onnx", "Dynamic INT8 (MatMul weights UINT8)", "sherpa-onnx 1.13.8 / ORT 1.30.0 CPU"),
    EngineSpec("ort", "ONNX Runtime", "onnx", "Dynamic INT8 (MatMul weights UINT8)", "ONNX Runtime 1.30.0 CPU EP"),
    EngineSpec("mnn", "MNN", "mnn", "Q8 weights / floating activations", "MNN 3.6.1 / sherpa-mnn CPU"),
    EngineSpec("ncnn", "ncnn", "ncnn", "INT8 mixed model", "ncnn c4193aa / sherpa-ncnn c61e50d CPU"),
    EngineSpec("litert", "LiteRT", "litert", "Dynamic W8 / FP32 activations", "LiteRT 2.3.2 / CPU, XNNPACK requested"),
) + listOf(4, 2).flatMap { bits ->
    listOf("sherpa" to "sherpa-onnx", "ort" to "ONNX Runtime").map { (id, title) ->
        EngineSpec("$id-q$bits", title, "onnx-q$bits", "W$bits / floating activations, MatMulNBits",
            if (id == "sherpa") "sherpa-onnx 1.13.8 / official ORT 1.30.0 CPU" else "Official ONNX Runtime 1.30.0 CPU EP", "model.onnx")
    }
} + (7 downTo 2).map { bits ->
    EngineSpec("mnn-q$bits", "MNN", "mnn-q$bits", "Q$bits weights / floating activations", "MNN 3.6.1 / sherpa-mnn CPU")
} + listOf(
    EngineSpec("ncnn-fp16", "ncnn", "ncnn-fp16", "FP16 weight storage / CPU", "ncnn c4193aa / sherpa-ncnn c61e50d CPU"),
    EngineSpec("litert-q4", "LiteRT", "litert-q4", "W4 / FP32 activations, requantized from W8", "LiteRT 2.3.2 / CPU, XNNPACK requested", "model.tflite"),
    EngineSpec("litert-q2", "LiteRT", "litert-q2", "W2 / FP32 activations, requantized from W8", "LiteRT 2.3.2 / CPU, XNNPACK requested", "model.tflite"),
) + listOf("q8_0", "q6_k", "q5_0", "q5_k", "q4_0", "q4_1", "q4_k", "q3_k", "fp16").map { variant ->
    EngineSpec("gguf-$variant", "SenseVoice.cpp", "gguf-$variant", "${variant.uppercase()} mixed weights / floating activations",
        "SenseVoice.cpp 1.4.0 / GGML CPU", "model.gguf")
}

val gpuEngineSpecs = listOf(
    EngineSpec("ncnn-vk-fp16", "ncnn", "ncnn-fp16", "FP16 weight storage", "ncnn c4193aa / Vulkan",
        backend = "vulkan", fp16 = true, accelerator = "GPU Vulkan", compute = "FP16"),
    EngineSpec("ncnn-vk-int8", "ncnn", "ncnn", "INT8 mixed model", "ncnn c4193aa / Vulkan",
        backend = "vulkan", fp16 = false, accelerator = "GPU Vulkan", compute = "INT8 layers fall back to CPU"),
    EngineSpec("ncnn-vk-fp32c", "ncnn", "ncnn-fp16", "FP16 weight storage", "ncnn c4193aa / Vulkan",
        backend = "vulkan", fp16 = false, accelerator = "GPU Vulkan", compute = "FP16 storage / FP32 compute"),
    EngineSpec("mnn-cl-fp16", "MNN", "mnn-fp32", "FP32 weights", "MNN 3.6.1 / OpenCL",
        backend = "opencl", fp16 = true, accelerator = "GPU OpenCL", compute = "FP16"),
    EngineSpec("mnn-cl-fp32", "MNN", "mnn-fp32", "FP32 weights", "MNN 3.6.1 / OpenCL",
        backend = "opencl-fp32", fp16 = false, accelerator = "GPU OpenCL", compute = "FP32"),
    EngineSpec("mnn-vk-fp16", "MNN", "mnn-fp32", "FP32 weights", "MNN 3.6.1 / Vulkan",
        backend = "vulkan", fp16 = true, accelerator = "GPU Vulkan", compute = "FP16"),
    EngineSpec("mnn-cl-q8", "MNN", "mnn", "Q8 weights / block 64", "MNN 3.6.1 / OpenCL",
        backend = "opencl", fp16 = true, accelerator = "GPU OpenCL", compute = "Q8 weights, partial fallback"),
    EngineSpec("mnn-cpu-fp32", "MNN", "mnn-fp32", "FP32 weights", "MNN 3.6.1 / CPU",
        backend = "cpu", accelerator = "CPU", compute = "FP32 control"),
    EngineSpec("mnn-cpu-static", "MNN", "mnn-fp32-static", "FP32 weights, static shapes", "MNN 3.6.1 / CPU",
        backend = "cpu", accelerator = "CPU", compute = "FP32 static control"),
    EngineSpec("mnn-vk-static", "MNN", "mnn-fp32-static", "FP32 weights, static shapes", "MNN 3.6.1 / Vulkan",
        backend = "vulkan", fp16 = true, accelerator = "GPU Vulkan", compute = "FP16, static"),
    EngineSpec("ort-nnapi", "ONNX Runtime", "onnx-fp32", "FP32 (official unquantized ONNX)", "ONNX Runtime 1.30.0 NNAPI EP",
        "model.onnx", backend = "nnapi", accelerator = "NNAPI", compute = "default flags"),
    EngineSpec("ort-nnapi-fp16", "ONNX Runtime", "onnx-fp32", "FP32 (official unquantized ONNX)", "ONNX Runtime 1.30.0 NNAPI EP",
        "model.onnx", backend = "nnapi", accelerator = "NNAPI", compute = "USE_FP16", nnapiFp16 = true),
    EngineSpec("ort-nnapi-q8", "ONNX Runtime", "onnx", "Dynamic INT8 (MatMul weights UINT8)", "ONNX Runtime 1.30.0 NNAPI EP",
        backend = "nnapi", accelerator = "NNAPI", compute = "dynamic INT8"),
    EngineSpec("sherpa-nnapi", "sherpa-onnx", "onnx-fp32", "FP32 (official unquantized ONNX)", "sherpa-onnx 1.13.8 / NNAPI",
        "model.onnx", backend = "nnapi", accelerator = "NNAPI", compute = "default flags"),
    EngineSpec("litert-gpu-fp16", "LiteRT", "litert-fp32", "FP32 (dequantized from W8)", "LiteRT 2.3.2 / CompiledModel GPU",
        "model.tflite", backend = "gpu", fp16 = true, accelerator = "GPU delegate", compute = "FP16"),
    EngineSpec("litert-gpu-fp32", "LiteRT", "litert-fp32", "FP32 (dequantized from W8)", "LiteRT 2.3.2 / CompiledModel GPU",
        "model.tflite", backend = "gpu", fp16 = false, accelerator = "GPU delegate", compute = "FP32"),
    EngineSpec("litert-gpu-q8", "LiteRT", "litert", "Dynamic W8 / FP32 activations", "LiteRT 2.3.2 / CompiledModel GPU",
        backend = "gpu", fp16 = true, accelerator = "GPU delegate", compute = "dynamic INT8, partial fallback"),
    EngineSpec("litert-cpu-fp32", "LiteRT", "litert-fp32", "FP32 (dequantized from W8)", "LiteRT 2.3.2 / CPU, XNNPACK requested",
        "model.tflite", backend = "cpu", accelerator = "CPU", compute = "FP32 control"),
    // Kept last and unselected by default: MNN's OpenCL backend mis-computes one
    // encoder frame for this graph (402 vs 401) and the wrapper segfaults on the
    // null result. Run it alone only when collecting that failure.
    EngineSpec("mnn-cl-static", "MNN", "mnn-fp32-static", "FP32 weights, static shapes", "MNN 3.6.1 / OpenCL",
        backend = "opencl", fp16 = true, accelerator = "GPU OpenCL", compute = "FP16, static (OpenCL crashes)",
        experimental = true),
)

val engineSpecs = cpuEngineSpecs + gpuEngineSpecs

interface Engine : Closeable {
    fun recognize(samples: FloatArray): String
    /** Evidence about the backend actually used, when the runtime can report it. */
    val backendInfo: String? get() = null
}

// 25 ms windows, 10 ms shift, snip_edges=true, followed by LFR(7,6).
fun lfrFrameCount(sampleCount: Int): Int {
    val fbankFrames = if (sampleCount < 400) 0 else 1 + (sampleCount - 400) / 160
    return (fbankFrames + 5) / 6
}

fun liteRtBucket(frameCount: Int): Int =
    listOf(63, 125, 250, 500).firstOrNull { it >= frameCount }
        ?: error("Audio exceeds LiteRT bucket limit")

fun createEngine(spec: EngineSpec, root: File, threads: Int): Engine {
    val folder = File(root, spec.folder)
    val tokens = File(folder, "tokens.txt").absolutePath
    return when (spec.engineId) {
        "sherpa" -> NativeEngine(SherpaNative.create(File(folder, spec.modelFile.ifEmpty { "model.int8.onnx" }).absolutePath, tokens, threads, spec.backend), SherpaNative::recognize, SherpaNative::destroy, spec.backend)
        "mnn" -> NativeEngine(MnnNative.create(File(folder, "model.mnn").absolutePath, tokens, threads, spec.backend), MnnNative::recognize, MnnNative::destroy, "MNN provider: ${spec.backend}")
        "ncnn" -> NativeEngine(NcnnNative.create(folder.absolutePath, tokens, threads, spec.backend != "cpu", spec.fp16), NcnnNative::recognize, NcnnNative::destroy, if (spec.backend == "cpu") "ncnn CPU" else NcnnNative.deviceInfo())
        "ort" -> OrtEngine(folder, threads, spec.modelFile.ifEmpty { "model.int8.onnx" }, spec.backend == "nnapi", spec.nnapiFp16)
        "litert" -> if (spec.backend == "gpu")
            LiteRtCompiledEngine(folder, spec.modelFile.ifEmpty { "sensevoice_small_q8.tflite" }, spec.fp16)
        else LiteRtEngine(folder, threads, spec.modelFile.ifEmpty { "sensevoice_small_q8.tflite" })
        "gguf" -> NativeEngine(GgufNative.create(File(folder, spec.modelFile).absolutePath, threads), GgufNative::recognize, GgufNative::destroy, "GGML CPU")
        else -> error("Unknown engine: ${spec.id}")
    }
}

private class NativeEngine(
    private var handle: Long,
    private val run: (Long, FloatArray) -> String,
    private val release: (Long) -> Unit,
    override val backendInfo: String?,
) : Engine {
    init { check(handle != 0L) { "Native model initialization failed" } }
    override fun recognize(samples: FloatArray) = run(handle, samples)
    override fun close() { if (handle != 0L) { release(handle); handle = 0 } }
}

/** CMVN follows LFR(7,6); edge frames are replicated, without waveform normalization. */
private fun features(samples: FloatArray, shift: FloatArray, scale: FloatArray): Array<FloatArray> {
    require(shift.size == 560 && scale.size == 560)
    val bank = FeatureNative.fbank(samples)
    val count = bank.size / 80
    require(count > 0)
    return Array((count + 5) / 6) { frame ->
        FloatArray(560) { dim ->
            val source = (frame * 6 + dim / 80 - 3).coerceIn(0, count - 1)
            (bank[source * 80 + dim % 80] + shift[dim]) * scale[dim]
        }
    }
}

private fun readTokens(file: File): Map<Int, String> = file.readLines().associate { line ->
    val split = line.lastIndexOf(' ')
    require(split >= 0) { "Invalid token line" }
    line.substring(split + 1).toInt() to line.substring(0, split)
}

private fun decode(logits: Array<FloatArray>, validRows: Int, tokens: Map<Int, String>): String {
    var previous = -1
    val text = StringBuilder()
    for (row in 4 until validRows.coerceAtMost(logits.size)) {
        val values = logits[row]
        require(values.isNotEmpty())
        var best = 0
        for (i in 1 until values.size) if (values[i] > values[best]) best = i
        if (best != 0 && best != previous) {
            val token = tokens[best] ?: error("Missing token $best")
            if (!token.startsWith("<|")) text.append(token)
        }
        previous = best
    }
    return text.toString().replace('▁', ' ').trim()
}

private class OrtEngine(
    folder: File,
    threads: Int,
    modelFile: String,
    private val nnapi: Boolean,
    private val nnapiFp16: Boolean,
) : Engine {
    private val env = OrtEnvironment.getEnvironment()
    private val session = OrtSession.SessionOptions().use { options ->
        options.setIntraOpNumThreads(threads)
        options.setInterOpNumThreads(1)
        if (nnapi) {
            val flags = if (nnapiFp16) EnumSet.of(NNAPIFlags.USE_FP16) else EnumSet.noneOf(NNAPIFlags::class.java)
            options.addNnapi(flags)
        }
        runCatching { options.enableProfiling(File(folder.parentFile, ".ort-profile.json").absolutePath) }
        env.createSession(File(folder, modelFile).absolutePath, options)
    }
    private val meta = session.metadata.customMetadata
    private val shift = meta.getValue("neg_mean").split(',').map(String::toFloat).toFloatArray()
    private val scale = meta.getValue("inv_stddev").split(',').map(String::toFloat).toFloatArray()
    private val tokens = readTokens(File(folder, "tokens.txt"))
    private var profileSummary: String? = null

    override val backendInfo: String?
        get() = profileSummary ?: runCatching {
            val json = session.endProfiling()
            val nnapiEvents = Regex("\"NnapiExecutionProvider\"|NnapiExecutionProvider").findAll(json).count()
            profileSummary = "NNAPI=${if (nnapi) "requested" else "off"}; profile NNAPI events=$nnapiEvents"
            profileSummary
        }.getOrNull()

    override fun recognize(samples: FloatArray): String {
        val frames = features(samples, shift, scale)
        val packed = FloatArray(frames.size * 560)
        frames.forEachIndexed { i, row -> row.copyInto(packed, i * 560) }
        val inputs = linkedMapOf<String, OnnxTensor>()
        try {
            inputs["x"] = OnnxTensor.createTensor(env, FloatBuffer.wrap(packed), longArrayOf(1, frames.size.toLong(), 560))
            fun scalar(name: String, value: Int) { inputs[name] = OnnxTensor.createTensor(env, IntBuffer.wrap(intArrayOf(value)), longArrayOf(1)) }
            scalar("x_length", frames.size)
            scalar("language", meta.getValue("lang_en").toInt())
            scalar("text_norm", meta.getValue("without_itn").toInt())
            session.run(inputs).use { output ->
                @Suppress("UNCHECKED_CAST")
                val logits = output[0].value as Array<Array<FloatArray>>
                return decode(logits[0], frames.size + 4, tokens)
            }
        } finally { inputs.values.forEach { it.close() } }
    }
    override fun close() = session.close()
}

private class LiteRtEngine(
    folder: File,
    threads: Int,
    modelFile: String,
) : Engine {
    private val interpreter = Interpreter(File(folder, modelFile), Interpreter.Options().apply {
        setNumThreads(threads)
        setUseXNNPACK(true)
    })
    private val cmvn = JSONObject(File(folder, "cmvn.json").readText())
    private val shift = cmvn.getJSONArray("shift").let { values -> FloatArray(values.length()) { values.getDouble(it).toFloat() } }
    private val scale = cmvn.getJSONArray("scale").let { values -> FloatArray(values.length()) { values.getDouble(it).toFloat() } }
    private val tokens = readTokens(File(folder, "tokens.txt"))

    override val backendInfo: String? get() = "CPU/XNNPACK"

    override fun recognize(samples: FloatArray): String {
        val frames = features(samples, shift, scale)
        val bucket = liteRtBucket(frames.size)
        val signature = "sv_$bucket"
        val data = arrayOf(Array(bucket) { row -> if (row < frames.size) frames[row] else FloatArray(560) })
        val inputs = mapOf<String, Any>("args_0" to data, "args_1" to intArrayOf(frames.size), "args_2" to intArrayOf(4), "args_3" to intArrayOf(15))
        val output = arrayOf(Array(bucket + 4) { FloatArray(25055) })
        val names = interpreter.getSignatureOutputs(signature)
        require(names.size == 1) { "Unexpected LiteRT output contract" }
        interpreter.runSignature(inputs, mapOf(names[0] to output), signature)
        return decode(output[0], frames.size + 4, tokens)
    }

    override fun close() = interpreter.close()
}

/** LiteRT 2.x GPU path: CompiledModel + Accelerator.GPU, no legacy delegate. */
private class LiteRtCompiledEngine(
    folder: File,
    modelFile: String,
    private val fp16: Boolean,
) : Engine {
    private val cacheDir = File(folder, ".gpu_cache").apply { mkdirs() }
    private val model = CompiledModel.create(
        File(folder, modelFile).absolutePath,
        CompiledModel.Options(Accelerator.GPU).apply {
            gpuOptions = CompiledModel.GpuOptions(
                precision = if (fp16) CompiledModel.GpuOptions.Precision.FP16
                else CompiledModel.GpuOptions.Precision.FP32,
                serializationDir = cacheDir.absolutePath,
                modelCacheKey = "sensevoice-${if (fp16) "fp16" else "fp32"}",
                serializeProgramCache = true,
            )
        },
    )
    private val cmvn = JSONObject(File(folder, "cmvn.json").readText())
    private val shift = cmvn.getJSONArray("shift").let { values -> FloatArray(values.length()) { values.getDouble(it).toFloat() } }
    private val scale = cmvn.getJSONArray("scale").let { values -> FloatArray(values.length()) { values.getDouble(it).toFloat() } }
    private val tokens = readTokens(File(folder, "tokens.txt"))

    override val backendInfo: String?
        get() = "GPU delegate (CompiledModel), precision=${if (fp16) "FP16" else "FP32"}"

    override fun recognize(samples: FloatArray): String {
        val frames = features(samples, shift, scale)
        val bucket = liteRtBucket(frames.size)
        val signature = "sv_$bucket"
        val packed = FloatArray(bucket * 560)
        frames.forEachIndexed { index, row -> row.copyInto(packed, index * 560) }
        val inputs = model.createInputBuffers(signature)
        val outputs = model.createOutputBuffers(signature)
        try {
            require(inputs.size == 4 && outputs.size == 1) { "Unexpected LiteRT signature contract" }
            inputs[0].writeFloat(packed)
            inputs[1].writeInt(intArrayOf(frames.size))
            inputs[2].writeInt(intArrayOf(4))
            inputs[3].writeInt(intArrayOf(15))
            model.run(inputs, outputs, signature)
            val flat = outputs[0].readFloat()
            val rows = Array(bucket + 4) { row ->
                FloatArray(25055) { column -> flat[row * 25055 + column] }
            }
            return decode(rows, frames.size + 4, tokens)
        } finally {
            inputs.forEach { it.close() }
            outputs.forEach { it.close() }
        }
    }

    override fun close() = model.close()
}
