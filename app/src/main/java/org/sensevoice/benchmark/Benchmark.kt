package org.sensevoice.benchmark

import android.app.Application
import android.os.Build
import android.os.Debug
import android.os.SystemClock
import android.os.PowerManager
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest
import java.util.Locale
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong
import kotlin.math.ceil

data class BenchState(
    val busy: Boolean = false, val status: String = "Import CPU models to begin.",
    val installed: Set<String> = emptySet(), val report: String? = null, val running: Boolean = false,
)
data class RunConfig(val threads: Int, val warmup: Int, val repetitions: Int)

class BenchmarkViewModel(application: Application) : AndroidViewModel(application) {
    private val store = ModelStore(application)
    private val mutableState = MutableStateFlow(BenchState(installed = installed(), report = lastReport()))
    val state = mutableState.asStateFlow()
    private val cancel = AtomicBoolean(false)
    private fun installed() = engineSpecs.filter(store::available).map { it.id }.toSet()
    private fun lastReport() = File(getApplication<Application>().filesDir, "last-report.json").takeIf { it.exists() }?.readText()

    fun importModels(uri: android.net.Uri) {
        if (state.value.busy) return
        mutableState.value = state.value.copy(busy = true, status = "Reading model manifest…")
        viewModelScope.launch {
            val outcome = runCatching {
                withContext(Dispatchers.IO) {
                    store.importDirectory(uri) { message -> mutableState.value = state.value.copy(status = message) }
                }
            }
            mutableState.value = state.value.copy(busy = false, installed = installed(), status = outcome.fold({ "Models imported and SHA-256 verified." }, { "Import failed: ${it.message}" }))
        }
    }

    fun cancelRun() { cancel.set(true); mutableState.value = state.value.copy(status = "Stopping after the current inference…") }

    fun run(ids: Set<String>, config: RunConfig) {
        if (state.value.busy || ids.isEmpty()) return
        require(config.threads in 1..16 && config.warmup in 0..20 && config.repetitions in 1..100)
        cancel.set(false)
        mutableState.value = state.value.copy(busy = true, running = true, status = "Preparing fixed audio…", report = null)
        viewModelScope.launch {
            val result = runCatching { withContext(Dispatchers.IO) { benchmark(ids, config) } }
            mutableState.value = state.value.copy(busy = false, running = false, report = result.getOrNull(), status = result.fold({ if (cancel.get()) "Run cancelled. Partial results saved." else "Benchmark complete. Results saved." }, { "Benchmark failed: ${it.message}" }))
        }
    }

    override fun onCleared() { cancel.set(true); super.onCleared() }

    private fun benchmark(ids: Set<String>, config: RunConfig): String {
        val context = getApplication<Application>()
        val audioMeta = JSONObject(context.assets.open("audio/sample.json").bufferedReader().use { it.readText() })
        val wav = context.assets.open("audio/sample.wav").use { it.readBytes() }
        val actual = MessageDigest.getInstance("SHA-256").digest(wav).joinToString("") { "%02x".format(it) }
        check(actual == audioMeta.getString("sha256")) { "Fixed audio checksum mismatch" }
        val samples = readWave(wav)
        val duration = samples.size / 16000.0
        val battery = context.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
        val power = context.getSystemService(PowerManager::class.java)
        fun environment() = JSONObject().put("thermalStatus", if (Build.VERSION.SDK_INT >= 29) power.currentThermalStatus else JSONObject.NULL)
            .put("batteryTemperatureTenthsC", context.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))?.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, -1))
        val report = JSONObject().put("schemaVersion", 2).put("createdAt", java.time.Instant.now().toString())
            .put("appVersion", BuildConfig.VERSION_NAME).put("buildVariant", BuildConfig.FLAVOR)
            .put("ortDistribution", "Official Maven 1.30.0").put("backend", "CPU").put("audio", audioMeta)
            .put("device", JSONObject().put("manufacturer", Build.MANUFACTURER).put("model", Build.MODEL).put("android", Build.VERSION.RELEASE)
                .put("api", Build.VERSION.SDK_INT).put("abis", JSONArray(Build.SUPPORTED_ABIS.toList()))
                .put("soc", if (Build.VERSION.SDK_INT >= 31) Build.SOC_MODEL else JSONObject.NULL)
                .put("chargingStatus", battery?.getIntExtra(BatteryManager.EXTRA_STATUS, -1)))
            .put("config", JSONObject().put("threads", config.threads).put("warmup", config.warmup).put("repetitions", config.repetitions)
                .put("executionOrder", JSONArray(engineSpecs.filter { it.id in ids }.map { it.id }))
                .put("language", "en").put("itn", false).put("timing", "PCM to text; includes feature extraction, tensor copies and decoding")
                .put("coreInferenceMs", JSONObject.NULL).put("memorySamplingIntervalMs", 100)
                .put("normalization", "Uppercase ASCII letters/digits, punctuation removed, whitespace collapsed"))
        val rows = JSONArray()
        report.put("results", rows)
        for (spec in engineSpecs.filter { it.id in ids }) {
            if (cancel.get()) break
            val row = JSONObject().put("engine", spec.title).put("id", spec.id).put("precision", spec.precision)
                .put("engineId", spec.engineId).put("modelVariant", spec.variant)
                .put("runtime", spec.runtime).put("backend", "CPU").put("startEnvironment", environment())
            rows.put(row)
            if (!store.available(spec)) { row.put("status", "missing_model"); continue }
            row.put("model", store.manifest!!.getJSONObject("engines").getJSONObject(spec.folder))
            val modelFiles = row.getJSONObject("model").getJSONArray("files")
            row.put("modelBundleBytes", (0 until modelFiles.length()).sumOf { modelFiles.getJSONObject(it).getLong("sizeBytes") })
            val baseline = pssKb()
            val peak = AtomicLong(baseline)
            val sampler = Executors.newSingleThreadScheduledExecutor()
            sampler.scheduleWithFixedDelay({ runCatching { peak.accumulateAndGet(pssKb(), ::maxOf) } }, 0, 100, TimeUnit.MILLISECONDS)
            var engine: Engine? = null
            val times = mutableListOf<Double>()
            val texts = JSONArray()
            try {
                mutableState.value = state.value.copy(status = "Loading ${spec.label}…")
                val loadStart = SystemClock.elapsedRealtimeNanos()
                engine = createEngine(spec, store.root, config.threads)
                row.put("loadMs", elapsed(loadStart)).put("pssBaselineKb", baseline).put("pssLoadedKb", pssKb())
                val firstStart = SystemClock.elapsedRealtimeNanos()
                val firstText = engine.recognize(samples)
                row.put("firstInferenceMs", elapsed(firstStart)).put("firstText", firstText)
                for (i in 0 until config.warmup) {
                    if (cancel.get()) break
                    mutableState.value = state.value.copy(status = "${spec.label}: warmup ${i + 1}/${config.warmup}")
                    engine.recognize(samples)
                }
                for (i in 0 until config.repetitions) {
                    if (cancel.get()) break
                    mutableState.value = state.value.copy(status = "${spec.label}: run ${i + 1}/${config.repetitions}")
                    val start = SystemClock.elapsedRealtimeNanos()
                    val text = engine.recognize(samples)
                    times += elapsed(start)
                    texts.put(text)
                }
                val text = if (texts.length() > 0) texts.getString(texts.length() - 1) else firstText
                row.put("status", if (cancel.get()) "cancelled" else if (text.isBlank()) "empty_output" else "ok")
                    .put("text", text).put("wer", errorRate(audioMeta.getString("reference"), text, words = true))
                    .put("cer", errorRate(audioMeta.getString("reference"), text, words = false))
                    .put("computedLfrFrames", if (spec.engineId == "litert") liteRtBucket(lfrFrameCount(samples.size)) else JSONObject.NULL)
                if (times.isNotEmpty()) {
                    val sorted = times.sorted()
                    val median = if (sorted.size % 2 == 0) (sorted[sorted.size / 2 - 1] + sorted[sorted.size / 2]) / 2 else sorted[sorted.size / 2]
                    row.put("meanMs", times.average()).put("medianMs", median).put("p90Ms", sorted[ceil(sorted.size * 0.9).toInt() - 1])
                        .put("minMs", sorted.first()).put("maxMs", sorted.last()).put("rtf", times.average() / (duration * 1000))
                }
            } catch (e: Exception) { row.put("status", "error").put("error", "${e.javaClass.simpleName}: ${e.message}")
            } catch (e: LinkageError) { row.put("status", "runtime_unavailable").put("error", e.message)
            } finally {
                runCatching { engine?.close() }.onFailure { row.put("cleanupError", it.message) }
                sampler.shutdown()
                sampler.awaitTermination(1, TimeUnit.SECONDS)
                row.put("sampledPeakPssKb", peak.get()).put("samplesMs", JSONArray(times)).put("texts", texts).put("endEnvironment", environment())
                report.put("cancelled", cancel.get())
                File(context.filesDir, "last-report.json").writeText(report.toString(2))
            }
        }
        report.put("cancelled", cancel.get()).put("completedAt", java.time.Instant.now().toString())
        return report.toString(2).also { File(context.filesDir, "last-report.json").writeText(it) }
    }
}

private fun elapsed(start: Long) = (SystemClock.elapsedRealtimeNanos() - start) / 1_000_000.0
private fun pssKb() = Debug.MemoryInfo().also { Debug.getMemoryInfo(it) }.totalPss.toLong()

fun readWave(bytes: ByteArray): FloatArray {
    val b = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
    fun tag(offset: Int) = String(bytes, offset, 4, Charsets.US_ASCII)
    require(bytes.size >= 12 && tag(0) == "RIFF" && tag(8) == "WAVE")
    var pos = 12
    var validFormat = false
    while (pos + 8 <= bytes.size) {
        val size = b.getInt(pos + 4)
        require(size >= 0 && pos.toLong() + 8 + size <= bytes.size)
        when (tag(pos)) {
            "fmt " -> {
                require(size >= 16)
                validFormat = b.getShort(pos + 8).toInt() == 1 && b.getShort(pos + 10).toInt() == 1 && b.getInt(pos + 12) == 16000 && b.getShort(pos + 22).toInt() == 16
            }
            "data" -> {
                require(validFormat && size % 2 == 0) { "Expected 16 kHz mono PCM16 WAV" }
                return FloatArray(size / 2) { b.getShort(pos + 8 + it * 2).toFloat() / 32768f }
            }
        }
        pos += 8 + size + size % 2
    }
    error("WAV has no audio data")
}

private fun normalized(text: String) = text.uppercase(Locale.ROOT).replace(Regex("[^A-Z0-9 ]"), " ").trim().replace(Regex("\\s+"), " ")
fun errorRate(reference: String, text: String, words: Boolean): Double {
    fun units(value: String): List<String> {
        val clean = normalized(value)
        return if (clean.isBlank()) emptyList() else if (words) clean.split(' ') else clean.replace(" ", "").map(Char::toString)
    }
    val a = units(reference); val b = units(text)
    var previous = IntArray(b.size + 1) { it }
    for (i in a.indices) {
        val current = IntArray(b.size + 1); current[0] = i + 1
        for (j in b.indices) current[j + 1] = minOf(current[j] + 1, previous[j + 1] + 1, previous[j] + if (a[i] == b[j]) 0 else 1)
        previous = current
    }
    return previous.last().toDouble() / a.size.coerceAtLeast(1)
}

fun reportCsv(json: String): String {
    val report = JSONObject(json)
    val columns = listOf("id", "engine", "engineId", "modelVariant", "precision", "runtime", "status", "modelBundleBytes", "loadMs", "firstInferenceMs", "meanMs", "medianMs", "p90Ms", "rtf", "sampledPeakPssKb", "wer", "cer", "text", "error")
    fun escape(value: Any?) = "\"" + (value?.takeUnless { it == JSONObject.NULL }?.toString() ?: "").replace("\"", "\"\"") + "\""
    val rows = report.getJSONArray("results")
    return buildString {
        appendLine((columns + listOf("audioId", "audioSha256", "device", "threads", "warmup", "repetitions")).joinToString(",", transform = ::escape))
        for (i in 0 until rows.length()) {
            val row = rows.getJSONObject(i)
            val values = columns.map { row.opt(it) } + listOf(report.getJSONObject("audio").getString("utterance"), report.getJSONObject("audio").getString("sha256"), report.getJSONObject("device").getString("model"), report.getJSONObject("config").getInt("threads"), report.getJSONObject("config").getInt("warmup"), report.getJSONObject("config").getInt("repetitions"))
            appendLine(values.joinToString(",", transform = ::escape))
        }
    }
}
