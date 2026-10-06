"""Build CPU JNI libraries from pinned upstreams on Windows or Linux."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import requests
import time
import zipfile
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / ".native"
PINS = {
    "onnx": ("k2-fsa/sherpa-onnx", "v1.13.8"),
    "mnn": ("alibaba/MNN", "d407447ed56c4121a11ccbd266dc184ca1ead0c2"),
    "ncnn": ("k2-fsa/sherpa-ncnn", "c61e50d61e9fbed5972afa4d95bc560e168affe2"),
}


def run(*args, env=None):
    subprocess.run([str(a) for a in args], check=True, env=env)


def fetch(url, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    if not output.exists():
        print("Download", url, flush=True)
        temporary = output.with_suffix(output.suffix + ".part")
        for attempt in range(5):
            offset = temporary.stat().st_size if temporary.exists() else 0
            try:
                with requests.get(url, stream=True, headers={"Range": f"bytes={offset}-"} if offset else {}, timeout=(20, 45)) as response:
                    response.raise_for_status()
                    mode = "ab" if offset and response.status_code == 206 else "wb"
                    with temporary.open(mode) as dst:
                        for chunk in response.iter_content(1024 * 256):
                            dst.write(chunk)
                temporary.replace(output)
                break
            except requests.RequestException:
                if attempt == 4:
                    raise
                time.sleep(2)


def source(name):
    repo, revision = PINS[name]
    dst = WORK / (name + "-archive-source")
    if not (dst / "CMakeLists.txt").exists():
        archive = WORK / (name + "-" + revision + ".tar.gz")
        fetch(f"https://codeload.github.com/{repo}/tar.gz/{revision}", archive)
        dst.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive, "r:gz") as tar:
            for member in tar:
                parts = Path(member.name).parts
                if len(parts) < 2:
                    continue
                relative = Path(*parts[1:])
                target = (dst / relative).resolve()
                if not target.is_relative_to(dst.resolve()):
                    raise ValueError("Archive path outside source directory")
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                elif member.isfile():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with tar.extractfile(member) as src, target.open("wb") as out:
                        shutil.copyfileobj(src, out)
    return dst


def attach(src, name):
    cmake = src / "CMakeLists.txt"
    text = cmake.read_text(encoding="utf-8")
    marker = "# SenseVoice benchmark bridge"
    if marker not in text:
        text += f'\n{marker}\nadd_subdirectory("{(ROOT / "native" / name).as_posix()}" "${{CMAKE_BINARY_DIR}}/benchmark")\n'
        cmake.write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sdk", default=os.environ.get("ANDROID_HOME") or os.environ.get("ANDROID_SDK_ROOT"))
    parser.add_argument("--ndk", default="28.2.13676358")
    parser.add_argument("--cmake", default="cmake")
    parser.add_argument("--jobs", default="2")
    args = parser.parse_args()
    # Python honors the Windows system proxy; CMake's curl also needs env vars.
    for scheme, proxy in urllib.request.getproxies().items():
        if scheme in ("http", "https"):
            os.environ.setdefault(scheme.upper() + "_PROXY", proxy)
    os.environ["GIT_CEILING_DIRECTORIES"] = str(WORK)
    if os.name == "nt" and not shutil.which("sed"):
        git = shutil.which("git")
        git_tools = Path(git).resolve().parents[1] / "usr/bin" if git else None
        if git_tools and (git_tools / "sed.exe").exists():
            os.environ["PATH"] = str(git_tools) + os.pathsep + os.environ["PATH"]
        else:
            raise RuntimeError("Windows native builds need Git for Windows' usr/bin (sed.exe) on PATH")
    if not args.sdk:
        parser.error("Set ANDROID_HOME or pass --sdk")
    ndk = Path(args.sdk).resolve() / "ndk" / args.ndk
    toolchain = ndk / "build/cmake/android.toolchain.cmake"
    if not toolchain.exists():
        raise FileNotFoundError(toolchain)
    common = [
        "-G", "Ninja", f"-DCMAKE_TOOLCHAIN_FILE={toolchain.as_posix()}",
        "-DANDROID_ABI=arm64-v8a", "-DANDROID_PLATFORM=android-26", "-DANDROID_STL=c++_shared",
        "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_POSITION_INDEPENDENT_CODE=ON",
        "-DCMAKE_CXX_VISIBILITY_PRESET=hidden", "-DCMAKE_C_VISIBILITY_PRESET=hidden",
        "-DCMAKE_VISIBILITY_INLINES_HIDDEN=ON", "-DCMAKE_POLICY_VERSION_MINIMUM=3.5",
        "-DCMAKE_SHARED_LINKER_FLAGS=-Wl,-z,max-page-size=16384",
        "-DBUILD_SHARED_LIBS=OFF",
    ]

    def build(src, name, flags, targets, env=None):
        dst = WORK / (name + "-build")
        run(args.cmake, "-S", src, "-B", dst, *common, *flags, env=env)
        run(args.cmake, "--build", dst, "--target", *targets, "--parallel", args.jobs, env=env)
        return dst

    # sherpa and direct ORT use the same shared runtime from the Maven AAR.
    ort = WORK / "ort"
    fetch("https://repo.maven.apache.org/maven2/com/microsoft/onnxruntime/onnxruntime-android/1.23.2/onnxruntime-android-1.23.2.aar", ort / "runtime.aar")
    with zipfile.ZipFile(ort / "runtime.aar") as z:
        for name in ["jni/arm64-v8a/libonnxruntime.so"]:
            z.extract(name, ort)
    for header in ["onnxruntime_c_api.h", "onnxruntime_ep_c_api.h", "onnxruntime_cxx_api.h", "onnxruntime_cxx_inline.h", "onnxruntime_float16.h"]:
        fetch("https://raw.githubusercontent.com/microsoft/onnxruntime/v1.23.2/include/onnxruntime/core/session/" + header, ort / "include" / header)
    fetch("https://raw.githubusercontent.com/microsoft/onnxruntime/v1.23.2/include/onnxruntime/core/providers/nnapi/nnapi_provider_factory.h", ort / "include/nnapi_provider_factory.h")
    onnx = source("onnx")
    attach(onnx, "onnx")
    env = os.environ.copy()
    env["SHERPA_ONNXRUNTIME_INCLUDE_DIR"] = str(ort / "include")
    env["SHERPA_ONNXRUNTIME_LIB_DIR"] = str(ort / "jni/arm64-v8a")
    onnx_build = build(onnx, "onnx", [
        "-DSHERPA_ONNX_ENABLE_C_API=ON", "-DSHERPA_ONNX_ENABLE_BINARY=OFF",
        "-DSHERPA_ONNX_BUILD_C_API_EXAMPLES=OFF", "-DSHERPA_ONNX_ENABLE_JNI=OFF",
        "-DSHERPA_ONNX_ENABLE_PORTAUDIO=OFF", "-DSHERPA_ONNX_ENABLE_WEBSOCKET=OFF",
        "-DSHERPA_ONNX_ENABLE_TTS=OFF", "-DSHERPA_ONNX_ENABLE_SPEAKER_DIARIZATION=OFF",
        "-DSHERPA_ONNX_ENABLE_GPU=OFF", "-DSHERPA_ONNX_ENABLE_TESTS=OFF",
        "-DSHERPA_ONNX_LINK_LIBSTDCPP_STATICALLY=OFF",
    ], ["benchmark_sherpa"], env)

    mnn = source("mnn")
    install = WORK / "mnn-install"
    mnn_build = build(mnn, "mnn-runtime", [
        "-DMNN_BUILD_SHARED_LIBS=OFF", "-DMNN_SEP_BUILD=OFF", "-DMNN_LOW_MEMORY=ON",
        "-DMNN_BUILD_CONVERTER=OFF", "-DMNN_BUILD_TRAIN=OFF", "-DMNN_BUILD_TEST=OFF",
        "-DMNN_BUILD_DEMO=OFF", "-DMNN_BUILD_TOOLS=OFF", "-DMNN_BUILD_PROTOBUFFER=OFF",
        "-DMNN_BUILD_LLM=OFF", "-DMNN_BUILD_AUDIO=OFF", "-DMNN_BUILD_OPENCV=OFF",
        "-DMNN_OPENCL=OFF", "-DMNN_VULKAN=OFF", "-DMNN_OPENGL=OFF", "-DMNN_NNAPI=OFF",
        "-DMNN_BUILD_FOR_ANDROID_COMMAND=ON",
        "-DMNN_JNI=OFF", f"-DCMAKE_INSTALL_PREFIX={install.as_posix()}",
    ], ["MNN"])
    run(args.cmake, "--install", mnn_build)
    sherpa_mnn = mnn / "apps/frameworks/sherpa-mnn"
    attach(sherpa_mnn, "mnn")
    mnn_api_build = build(sherpa_mnn, "mnn-api", [
        f"-DMNN_LIB_DIR={install.as_posix()}", "-DSHERPA_MNN_ENABLE_C_API=ON",
        "-DSHERPA_MNN_ENABLE_BINARY=OFF", "-DSHERPA_MNN_BUILD_C_API_EXAMPLES=OFF",
        "-DSHERPA_MNN_ENABLE_PORTAUDIO=OFF", "-DSHERPA_MNN_ENABLE_WEBSOCKET=OFF",
        "-DSHERPA_MNN_ENABLE_TTS=OFF", "-DSHERPA_MNN_ENABLE_SPEAKER_DIARIZATION=OFF",
        "-DSHERPA_MNN_ENABLE_GPU=OFF", "-DSHERPA_MNN_ENABLE_TESTS=OFF",
    ], ["benchmark_mnn"])

    ncnn = source("ncnn")
    attach(ncnn, "ncnn")
    ncnn_build = build(ncnn, "ncnn", [
        "-DSHERPA_NCNN_ENABLE_C_API=OFF", "-DSHERPA_NCNN_ENABLE_JNI=OFF",
        "-DSHERPA_NCNN_ENABLE_BINARY=OFF", "-DSHERPA_NCNN_ENABLE_PORTAUDIO=OFF",
        "-DSHERPA_NCNN_ENABLE_GENERATE_INT8_SCALE_TABLE=OFF",
        "-DNCNN_VULKAN=OFF", "-DNCNN_INT8=ON",
        "-DNCNN_ARM82=OFF", "-DNCNN_ARM82DOT=OFF", "-DNCNN_ARM82FP16FML=OFF",
    ], ["benchmark_ncnn", "benchmark_features"])

    out = ROOT / "app/src/main/jniLibs/arm64-v8a"
    out.mkdir(parents=True, exist_ok=True)
    for directory, lib in [(onnx_build, "benchmark_sherpa"), (mnn_api_build, "benchmark_mnn"),
                           (ncnn_build, "benchmark_ncnn"), (ncnn_build, "benchmark_features")]:
        paths = list(directory.rglob("lib" + lib + ".so"))
        if len(paths) != 1:
            raise RuntimeError(f"Expected one {lib}: {paths}")
        shutil.copy2(paths[0], out / paths[0].name)
    host = "windows-x86_64" if os.name == "nt" else "linux-x86_64"
    shutil.copy2(ndk / "toolchains/llvm/prebuilt" / host / "sysroot/usr/lib/aarch64-linux-android/libc++_shared.so", out)
    notices = ROOT / "app/src/main/assets/runtime-licenses"
    notices.mkdir(parents=True, exist_ok=True)
    for name, directory in [("sherpa-onnx", onnx), ("MNN", mnn), ("sherpa-mnn", sherpa_mnn), ("sherpa-ncnn", ncnn)]:
        for file in directory.iterdir():
            if file.is_file() and file.name.upper().startswith(("LICENSE", "COPYING", "NOTICE")):
                shutil.copy2(file, notices / (name + "-" + file.name + ".txt"))
    for build_dir in [onnx_build, mnn_api_build, ncnn_build]:
        deps = build_dir / "_deps"
        if not deps.exists():
            continue
        for dependency in deps.glob("*-src"):
            for file in dependency.iterdir():
                if file.is_file() and file.name.upper().startswith(("LICENSE", "COPYING", "NOTICE")):
                    shutil.copy2(file, notices / (dependency.name + "-" + file.name + ".txt"))
    for vendor in (mnn / "3rd_party").iterdir():
        if vendor.is_dir():
            for file in vendor.iterdir():
                if file.is_file() and file.name.upper().startswith(("LICENSE", "COPYING", "NOTICE")):
                    shutil.copy2(file, notices / ("MNN-" + vendor.name + "-" + file.name + ".txt"))
    print("CPU JNI libraries staged:", out, flush=True)


if __name__ == "__main__":
    main()
