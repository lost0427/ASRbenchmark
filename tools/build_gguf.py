"""Build the pinned SenseVoice.cpp CPU bridge; ONNX Runtime stays official."""
import argparse
import os
from pathlib import Path
import shutil
import tarfile

from build_native import ROOT, WORK, PINS, source, fetch, attach, run

GGUF_REV = "c78e6919351ac83255e96de46169f518097f1ef3"
GGML_REV = "a5960e80d3e65ce6ff18f90315ab96f63cf9c4cc"


def prepare_source():
    PINS["gguf"] = ("lovemefan/SenseVoice.cpp", GGUF_REV)
    src = source("gguf")
    ggml = src / "sense-voice/csrc/third-party/ggml"
    if not (ggml / "CMakeLists.txt").exists():
        archive = WORK / "lowprecision-source/ggml.tar.gz"
        fetch(f"https://codeload.github.com/ggerganov/ggml/tar.gz/{GGML_REV}", archive)
        ggml.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive, "r:gz") as tar:
            for member in tar:
                parts = Path(member.name).parts[1:]
                if not parts: continue
                destination = (ggml / Path(*parts)).resolve()
                if not destination.is_relative_to(ggml.resolve()): raise ValueError("Invalid archive path")
                if member.isdir(): destination.mkdir(parents=True, exist_ok=True)
                elif member.isfile():
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with tar.extractfile(member) as inp, destination.open("wb") as out: shutil.copyfileobj(inp, out)
    # Fixed-utterance benchmark runs the whole clip; no VAD graph is needed.
    # The published GGUFs also predate v1.4's optional embedded Silero weights.
    implementation = src / "sense-voice/csrc/sense-voice.cc"
    text = implementation.read_text(encoding="utf-8")
    if "// vad allocator" in text:
        start = text.index("    // vad allocator")
        end = text.index("    // encoder allocator", start)
        text = text[:start] + "    // Benchmark: no VAD; recognize the complete fixed utterance.\n\n" + text[end:]
    text = text.replace("n_threads, true, cmvn, state->feature", "n_threads, debug, cmvn, state->feature")
    implementation.write_text(text, encoding="utf-8")
    return src


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk", default=os.environ.get("ANDROID_HOME") or os.environ.get("ANDROID_SDK_ROOT"))
    parser.add_argument("--cmake", default="cmake")
    parser.add_argument("--jobs", default="2")
    parser.add_argument("--host", action="store_true", help="Build desktop recognizer and quantizer")
    args = parser.parse_args()
    src = prepare_source()
    common = ["-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release", "-DBUILD_SHARED_LIBS=OFF",
              "-DCMAKE_POSITION_INDEPENDENT_CODE=ON", "-DGGML_NATIVE=OFF", "-DGGML_OPENMP=OFF",
              "-DGGML_CPU_ALL_VARIANTS=OFF", "-DGGML_BLAS=OFF", "-DGGML_CUDA=OFF", "-DGGML_VULKAN=OFF"]
    if args.host:
        build = WORK / "gguf-host-build"
        cmake = src / "CMakeLists.txt"
        text = cmake.read_text(encoding="utf-8")
        if "# Benchmark desktop smoke" not in text:
            text += (f'\n# Benchmark desktop smoke\nif(NOT ANDROID)\n'
                     f'  add_executable(benchmark_gguf_smoke "{(ROOT / "native/gguf/smoke.cc").as_posix()}")\n'
                     '  target_include_directories(benchmark_gguf_smoke PRIVATE ${CMAKE_SOURCE_DIR}/sense-voice/csrc)\n'
                     '  target_link_libraries(benchmark_gguf_smoke PRIVATE sense-voice-core ggml)\nendif()\n')
            cmake.write_text(text, encoding="utf-8")
        run(args.cmake, "-S", src, "-B", build, *common, "-DSENSE_VOICE_BUILD_EXAMPLES=OFF")
        run(args.cmake, "--build", build, "--target", "benchmark_gguf_smoke", "--parallel", args.jobs)
        return
    if not args.sdk: parser.error("Pass --sdk or set ANDROID_HOME")
    ndk = Path(args.sdk).resolve() / "ndk/28.2.13676358"
    attach(src, "gguf")
    build = WORK / "gguf-android-build"
    run(args.cmake, "-S", src, "-B", build, *common,
        f"-DCMAKE_TOOLCHAIN_FILE={(ndk / 'build/cmake/android.toolchain.cmake').as_posix()}",
        "-DANDROID_ABI=arm64-v8a", "-DANDROID_PLATFORM=android-26", "-DANDROID_STL=c++_shared",
        "-DCMAKE_CXX_VISIBILITY_PRESET=hidden", "-DCMAKE_C_VISIBILITY_PRESET=hidden",
        "-DCMAKE_VISIBILITY_INLINES_HIDDEN=ON", "-DSENSE_VOICE_BUILD_EXAMPLES=OFF",
        "-DCMAKE_SHARED_LINKER_FLAGS=-Wl,-z,max-page-size=16384")
    run(args.cmake, "--build", build, "--target", "benchmark_gguf", "--parallel", args.jobs)
    library = build / "lib/libbenchmark_gguf.so"
    for suffix in ["", "-aggressive"]:
        target = ROOT / f"app/src/main/jniLibs{suffix}/arm64-v8a"
        target.mkdir(parents=True, exist_ok=True)
        shutil.copy2(library, target / library.name)
    notices = ROOT / "app/src/main/assets/runtime-licenses"
    notices.mkdir(parents=True, exist_ok=True)
    for name, directory in [("SenseVoice.cpp", src), ("GGML", src / "sense-voice/csrc/third-party/ggml")]:
        shutil.copy2(directory / "LICENSE", notices / f"{name}-LICENSE.txt")
    print("GGUF bridge staged for both APK flavors; official ORT is unchanged.", flush=True)


if __name__ == "__main__": main()
