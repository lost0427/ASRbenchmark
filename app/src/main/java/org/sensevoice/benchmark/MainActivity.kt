package org.sensevoice.benchmark

import android.os.Bundle
import android.view.WindowManager
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import org.json.JSONObject
import java.util.Locale

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            MaterialTheme {
                val vm: BenchmarkViewModel = viewModel()
                val state by vm.state.collectAsState()
                val audio = remember { JSONObject(assets.open("audio/sample.json").bufferedReader().use { it.readText() }) }
                val audioDuration = String.format(Locale.US, "%.2f", audio.getDouble("durationSeconds"))
                val audioBucket = liteRtBucket(lfrFrameCount(audio.getInt("frames")))
                DisposableEffect(state.busy) {
                    if (state.busy) window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
                    else window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
                    onDispose { window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON) }
                }
                val importModels = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocumentTree()) { uri -> uri?.let(vm::importModels) }
                var exportText by rememberSaveable { mutableStateOf("") }
                var exportStatus by remember { mutableStateOf("") }
                val saveJson = rememberLauncherForActivityResult(ActivityResultContracts.CreateDocument("application/json")) { uri ->
                    if (uri != null) exportStatus = runCatching { contentResolver.openOutputStream(uri)!!.bufferedWriter().use { it.write(exportText) }; "Report exported." }.getOrElse { "Export failed: ${it.message}" }
                }
                val saveCsv = rememberLauncherForActivityResult(ActivityResultContracts.CreateDocument("text/csv")) { uri ->
                    if (uri != null) exportStatus = runCatching { contentResolver.openOutputStream(uri)!!.bufferedWriter().use { it.write(exportText) }; "Report exported." }.getOrElse { "Export failed: ${it.message}" }
                }
                var selected by rememberSaveable { mutableStateOf(engineSpecs.map { it.id }) }
                var threads by rememberSaveable { mutableStateOf("2") }
                var warmup by rememberSaveable { mutableStateOf("3") }
                var repeats by rememberSaveable { mutableStateOf("10") }
                val config = RunConfig(threads.toIntOrNull() ?: 0, warmup.toIntOrNull() ?: -1, repeats.toIntOrNull() ?: 0)
                val valid = config.threads in 1..16 && config.warmup in 0..20 && config.repetitions in 1..100
                Surface(Modifier.fillMaxSize()) {
                    Column(Modifier.safeDrawingPadding().verticalScroll(rememberScrollState()).padding(20.dp), verticalArrangement = Arrangement.spacedBy(14.dp)) {
                        Text("SenseVoice Benchmark", style = MaterialTheme.typography.headlineMedium)
                        Text("CPU · Fixed LibriSpeech sample", style = MaterialTheme.typography.titleMedium)
                        Text("${audio.getString("utterance")} · $audioDuration s · 16 kHz mono\n${audio.getString("reference")}", style = MaterialTheme.typography.bodyMedium)
                        OutlinedButton(onClick = { importModels.launch(null) }, enabled = !state.busy) { Text("Import model folder") }
                        Text("Prepare the folder with tools/prepare_models.py. Models run offline after import.", style = MaterialTheme.typography.bodySmall)
                        engineSpecs.forEach { engine ->
                            Row(Modifier.fillMaxWidth()) {
                                Checkbox(checked = engine.id in selected, enabled = !state.busy, onCheckedChange = { checked -> selected = if (checked) selected + engine.id else selected - engine.id })
                                Column(Modifier.padding(top = 8.dp)) {
                                    Text(engine.title, style = MaterialTheme.typography.titleMedium)
                                    Text(engine.precision, style = MaterialTheme.typography.bodySmall)
                                    Text(if (engine.id in state.installed) "Ready" else "Model not imported", color = if (engine.id in state.installed) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.error)
                                }
                            }
                        }
                        Text("sherpa-onnx and ONNX Runtime share the same model and ORT version. LiteRT pads to $audioBucket LFR frames.", style = MaterialTheme.typography.bodySmall)
                        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                            OutlinedTextField(threads, { threads = it }, label = { Text("Threads") }, singleLine = true, enabled = !state.busy, keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number), modifier = Modifier.weight(1f))
                            OutlinedTextField(warmup, { warmup = it }, label = { Text("Warmup") }, singleLine = true, enabled = !state.busy, keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number), modifier = Modifier.weight(1f))
                            OutlinedTextField(repeats, { repeats = it }, label = { Text("Repeats") }, singleLine = true, enabled = !state.busy, keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number), modifier = Modifier.weight(1f))
                        }
                        if (!valid) Text("Threads: 1–16 · Warmup: 0–20 · Repeats: 1–100", color = MaterialTheme.colorScheme.error)
                        Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                            Button(onClick = { vm.run(selected.toSet(), config) }, enabled = !state.busy && valid && selected.any { it in state.installed }) { Text("Run benchmark") }
                            OutlinedButton(onClick = vm::cancelRun, enabled = state.running) { Text("Stop") }
                        }
                        if (state.busy) LinearProgressIndicator(Modifier.fillMaxWidth())
                        Text(state.status)
                        state.report?.let { json ->
                            val results = JSONObject(json).getJSONArray("results")
                            Text("Results", style = MaterialTheme.typography.headlineSmall)
                            Text("PCM to text latency · Sampled process PSS\nWER on one short clip is a sanity check, not a corpus accuracy score.", style = MaterialTheme.typography.bodySmall)
                            for (i in 0 until results.length()) {
                                val row = results.getJSONObject(i)
                                ElevatedCard(Modifier.fillMaxWidth()) {
                                    Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(5.dp)) {
                                        Text(row.getString("engine"), style = MaterialTheme.typography.titleMedium)
                                        Text(row.getString("status"))
                                        fun number(key: String, format: String = "%.2f") = if (row.has(key)) String.format(Locale.US, format, row.getDouble(key)) else "—"
                                        if (row.has("meanMs")) {
                                            Text("Mean ${number("meanMs")} ms · RTF ${number("rtf", "%.3f")}")
                                            Text("Median ${number("medianMs")} ms · P90 ${number("p90Ms")} ms")
                                            Text("Load ${number("loadMs")} ms · First ${number("firstInferenceMs")} ms")
                                            Text("Peak PSS ${number("sampledPeakPssKb", "%.0f")} KiB · WER ${number("wer", "%.3f")}")
                                        }
                                        Text(row.optString("text", row.optString("error", "")))
                                    }
                                }
                            }
                            Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                                OutlinedButton(onClick = { exportText = json; saveJson.launch("sensevoice-results.json") }, enabled = !state.busy) { Text("Export JSON") }
                                OutlinedButton(onClick = { exportText = reportCsv(json); saveCsv.launch("sensevoice-results.csv") }, enabled = !state.busy) { Text("Export CSV") }
                            }
                        }
                        if (exportStatus.isNotBlank()) Text(exportStatus)
                        Text("Audio: LibriSpeech test-clean · CC BY 4.0 · OpenSLR 12", style = MaterialTheme.typography.bodySmall)
                    }
                }
            }
        }
    }
}
