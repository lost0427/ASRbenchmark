package org.sensevoice.benchmark

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import org.json.JSONObject
import org.tensorflow.lite.Interpreter
import java.io.Closeable
import java.io.File
import java.nio.FloatBuffer
import java.nio.IntBuffer

object SherpaNative {
    init { System.loadLibrary("benchmark_sherpa") }
    external fun create(model: String, tokens: String, threads: Int): Long
    external fun recognize(handle: Long, samples: FloatArray): String
    external fun destroy(handle: Long)
}
object MnnNative {
    init { System.loadLibrary("benchmark_mnn") }
    external fun create(model: String, tokens: String, threads: Int): Long
    external fun recognize(handle: Long, samples: FloatArray): String
    external fun destroy(handle: Long)
}
object NcnnNative {
    init { System.loadLibrary("benchmark_ncnn") }
    external fun create(model: String, tokens: String, threads: Int): Long
    external fun recognize(handle: Long, samples: FloatArray): String
    external fun destroy(handle: Long)
}
object FeatureNative {
    init { System.loadLibrary("benchmark_features") }
    external fun fbank(samples: FloatArray): FloatArray
}

data class EngineSpec(val id: String, val title: String, val folder: String, val precision: String, val runtime: String)
val engineSpecs = listOf(
    EngineSpec("sherpa", "sherpa-onnx", "onnx", "Dynamic INT8 (MatMul weights UINT8)", "sherpa-onnx 1.13.8 / ORT 1.23.2 CPU"),
    EngineSpec("ort", "ONNX Runtime", "onnx", "Dynamic INT8 (MatMul weights UINT8)", "ONNX Runtime 1.23.2 CPU EP"),
    EngineSpec("mnn", "MNN", "mnn", "Q8 weights / floating activations", "MNN 3.6.1 / sherpa-mnn CPU"),
    EngineSpec("ncnn", "ncnn", "ncnn", "INT8 mixed model", "ncnn c4193aa / sherpa-ncnn c61e50d CPU"),
    EngineSpec("litert", "LiteRT", "litert", "Dynamic W8 / FP32 activations", "LiteRT 2.1.6 / CPU, XNNPACK requested"),
)

interface Engine : Closeable { fun recognize(samples: FloatArray): String }

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
    return when (spec.id) {
        "sherpa" -> NativeEngine(SherpaNative.create(File(folder, "model.int8.onnx").absolutePath, tokens, threads), SherpaNative::recognize, SherpaNative::destroy)
        "mnn" -> NativeEngine(MnnNative.create(File(folder, "model.mnn").absolutePath, tokens, threads), MnnNative::recognize, MnnNative::destroy)
        "ncnn" -> NativeEngine(NcnnNative.create(folder.absolutePath, tokens, threads), NcnnNative::recognize, NcnnNative::destroy)
        "ort" -> OrtEngine(folder, threads)
        "litert" -> LiteRtEngine(folder, threads)
        else -> error("Unknown engine: ${spec.id}")
    }
}

private class NativeEngine(private var handle: Long, private val run: (Long, FloatArray) -> String, private val release: (Long) -> Unit) : Engine {
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

private class OrtEngine(folder: File, threads: Int) : Engine {
    private val env = OrtEnvironment.getEnvironment()
    private val session = OrtSession.SessionOptions().use { options ->
        options.setIntraOpNumThreads(threads)
        options.setInterOpNumThreads(1)
        env.createSession(File(folder, "model.int8.onnx").absolutePath, options)
    }
    private val meta = session.metadata.customMetadata
    private val shift = meta.getValue("neg_mean").split(',').map(String::toFloat).toFloatArray()
    private val scale = meta.getValue("inv_stddev").split(',').map(String::toFloat).toFloatArray()
    private val tokens = readTokens(File(folder, "tokens.txt"))

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

private class LiteRtEngine(folder: File, threads: Int) : Engine {
    private val interpreter = Interpreter(File(folder, "sensevoice_small_q8.tflite"), Interpreter.Options().setNumThreads(threads).setUseXNNPACK(true))
    private val cmvn = JSONObject(File(folder, "cmvn.json").readText())
    private val shift = cmvn.getJSONArray("shift").let { values -> FloatArray(values.length()) { values.getDouble(it).toFloat() } }
    private val scale = cmvn.getJSONArray("scale").let { values -> FloatArray(values.length()) { values.getDouble(it).toFloat() } }
    private val tokens = readTokens(File(folder, "tokens.txt"))

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
