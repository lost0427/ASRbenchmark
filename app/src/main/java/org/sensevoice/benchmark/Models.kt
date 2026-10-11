package org.sensevoice.benchmark

import android.content.Context
import android.net.Uri
import androidx.documentfile.provider.DocumentFile
import org.json.JSONObject
import java.io.File
import java.security.MessageDigest

class ModelStore(private val context: Context) {
    val root = File(context.filesDir, "models")
    val manifest: JSONObject? get() = runCatching { JSONObject(File(root, "manifest.json").readText()) }.getOrNull()

    fun available(spec: EngineSpec): Boolean {
        val entries = manifest?.optJSONObject("engines") ?: return false
        return entries.has(spec.folder) && entries.getJSONObject(spec.folder).getJSONArray("files").let { files ->
            val paths = (0 until files.length()).map { files.getJSONObject(it).getString("path") }
            spec.requiredFiles.all { "${spec.folder}/$it" in paths } && paths.all { File(root, it).isFile }
        }
    }

    fun importDirectory(uri: Uri, progress: (String) -> Unit) {
        val directory = DocumentFile.fromTreeUri(context, uri) ?: error("Cannot open model directory")
        val sourceManifest = directory.findFile("manifest.json") ?: error("Select the folder containing manifest.json. Prepare it with tools/prepare_models.py.")
        val text = context.contentResolver.openInputStream(sourceManifest.uri)!!.bufferedReader().use { it.readText() }
        val data = JSONObject(text)
        require(data.getInt("schemaVersion") in 1..2) { "Unsupported model manifest" }
        val entries = data.getJSONObject("engines")
        require(entries.length() > 0) { "Manifest has no models" }
        val staging = File(context.filesDir, "models-import")
        staging.deleteRecursively()
        staging.mkdirs()
        try {
            for (name in entries.keys()) {
                require(name in engineSpecs.map { it.folder }) { "Unknown model folder: $name" }
                val files = entries.getJSONObject(name).getJSONArray("files")
                require(files.length() in 1..10)
                val declared = (0 until files.length()).map { files.getJSONObject(it).getString("path") }
                engineSpecs.filter { it.folder == name }.forEach { spec ->
                    require(spec.requiredFiles.all { "$name/$it" in declared }) { "Incomplete model: $name" }
                }
                for (i in 0 until files.length()) {
                    val entry = files.getJSONObject(i)
                    val relative = entry.getString("path")
                    require(relative.matches(Regex("$name/[A-Za-z0-9_.-]+"))) { "Invalid file path" }
                    val parts = relative.split('/')
                    val source = directory.findFile(parts[0])?.findFile(parts[1]) ?: error("Missing $relative")
                    val target = File(staging, relative)
                    target.parentFile!!.mkdirs()
                    val digest = MessageDigest.getInstance("SHA-256")
                    var size = 0L
                    progress("Importing $relative")
                    context.contentResolver.openInputStream(source.uri)!!.use { input ->
                        target.outputStream().use { output ->
                            val buffer = ByteArray(1024 * 1024)
                            while (true) {
                                val count = input.read(buffer)
                                if (count < 0) break
                                size += count
                                require(size <= entry.getLong("sizeBytes")) { "File larger than declared: $relative" }
                                digest.update(buffer, 0, count)
                                output.write(buffer, 0, count)
                            }
                        }
                    }
                    require(size == entry.getLong("sizeBytes")) { "File size mismatch: $relative" }
                    val actual = digest.digest().joinToString("") { "%02x".format(it) }
                    require(actual == entry.getString("sha256")) { "SHA-256 mismatch: $relative" }
                }
            }
            File(staging, "manifest.json").writeText(text)
            // Preserve the previous working set if the filesystem rename fails.
            val backup = File(context.filesDir, "models-previous")
            backup.deleteRecursively()
            if (root.exists()) check(root.renameTo(backup)) { "Cannot move previous models" }
            if (!staging.renameTo(root)) {
                backup.renameTo(root)
                error("Cannot activate imported models")
            }
            backup.deleteRecursively()
        } catch (e: Exception) { staging.deleteRecursively(); throw e }
    }
}
