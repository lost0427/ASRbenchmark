"""Verify signed arm64 APKs, native page alignment and official ORT packaging."""
import argparse
import hashlib
import os
from pathlib import Path
import struct
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def sha(data): return hashlib.sha256(data).hexdigest()


def check_apk(apk, build_tools, official):
    signer = build_tools / ("apksigner.bat" if os.name == "nt" else "apksigner")
    subprocess.run([str(signer), "verify", str(apk)], check=True)
    aligner = build_tools / ("zipalign.exe" if os.name == "nt" else "zipalign")
    subprocess.run([str(aligner), "-c", "-P", "16", "4", str(apk)], check=True)
    with zipfile.ZipFile(apk) as archive:
        names = [name for name in archive.namelist() if name.startswith("lib/") and name.endswith(".so")]
        assert names and all(name.startswith("lib/arm64-v8a/") for name in names), "APK must be arm64 only"
        for required in ["benchmark_sherpa", "benchmark_mnn", "benchmark_ncnn", "benchmark_features", "benchmark_gguf", "c++_shared", "onnxruntime", "onnxruntime4j_jni"]:
            assert f"lib/arm64-v8a/lib{required}.so" in names, f"Missing native runtime: {required}"
        for name in names:
            data = archive.read(name)
            assert data[:6] == b"\x7fELF\x02\x01", f"Expected little-endian ELF64: {name}"
            assert struct.unpack_from("<H", data, 18)[0] == 183, f"Expected AArch64: {name}"
            offset = struct.unpack_from("<Q", data, 32)[0]
            size, count = struct.unpack_from("<HH", data, 54)
            for i in range(count):
                header = offset + i * size
                if struct.unpack_from("<I", data, header)[0] == 1:  # PT_LOAD
                    alignment = struct.unpack_from("<Q", data, header + 48)[0]
                    assert alignment >= 16384, f"Native load segment lacks 16 KB alignment: {name}"
            assert archive.getinfo(name).compress_type == zipfile.ZIP_STORED, f"Native library is compressed: {name}"
        assert sha(archive.read("lib/arm64-v8a/libonnxruntime.so")) == sha(official), "APK does not contain the official ORT runtime"
    # Catch the previously observed unstripped 300 MB APK regression.
    assert apk.stat().st_size < 100_000_000, "APK exceeds the expected stripped-runtime size"
    print(f"Verified: {apk.name}, {apk.stat().st_size / 1e6:.1f} MB, SHA-256 {sha(apk.read_bytes())}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk", type=Path, default=os.environ.get("ANDROID_HOME") or os.environ.get("ANDROID_SDK_ROOT"))
    parser.add_argument("--build-tools", default="35.0.0")
    parser.add_argument("--ndk", default="28.2.13676358")
    parser.add_argument("apks", nargs="*", type=Path)
    args = parser.parse_args()
    if args.sdk is None: parser.error("Pass --sdk or set ANDROID_HOME")
    with zipfile.ZipFile(ROOT / ".native/ort/runtime.aar") as archive:
        official = archive.read("jni/arm64-v8a/libonnxruntime.so")
    # AGP runs --strip-unneeded even on the Maven binary, changing section-name
    # metadata. Apply the same operation before checking byte-for-byte identity.
    reference = ROOT / ".native/ort/packaging-reference.so"
    reference.write_bytes(official)
    stripped = reference.with_name("packaging-reference-stripped.so")
    host = "windows-x86_64" if os.name == "nt" else "linux-x86_64"
    strip = args.sdk / "ndk" / args.ndk / "toolchains/llvm/prebuilt" / host / "bin" / ("llvm-strip.exe" if os.name == "nt" else "llvm-strip")
    subprocess.run([str(strip), "--strip-unneeded", "-o", str(stripped), str(reference)], check=True)
    official = stripped.read_bytes()
    apks = args.apks or sorted((ROOT / "app/build/outputs/apk").glob("*/debug/*.apk"))
    assert apks, "No APKs found"
    for apk in apks: check_apk(apk.resolve(), args.sdk / "build-tools" / args.build_tools, official)


if __name__ == "__main__": main()
