"""Download CPU models into one SAF-importable folder; never add weights to Git."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import requests
from build_native import fetch
from motrix_download import MotrixDownloader

ONNX_REV = "2365baeacb507f821a0c8120fcee3d484dba7a07"
LITERT_REV = "8d1dfd033bafc5d40e03cb1dc714134386eb6895"
ONNX = "csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"
MNN_URL = "https://github.com/lost0427/PocketASR/releases/download/model-sherpa-mnn-sensevoice-q8-v1/"
NCNN_URL = "https://github.com/k2-fsa/sherpa-ncnn/releases/download/asr-models/sherpa-ncnn-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17.tar.bz2"


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def checked(url, output, expected=None, *, download=fetch):
    download(url, output)
    value = sha(output)
    if expected and value != expected:
        raise ValueError(f"SHA-256 mismatch: {output}. Remove this file and retry.")
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("models/cpu-models"))
    parser.add_argument("--engines", nargs="+", choices=["onnx", "mnn", "ncnn", "litert"], default=["onnx", "mnn", "ncnn", "litert"])
    parser.add_argument("--downloader", choices=["motrix", "requests"], default="motrix")
    parser.add_argument("--motrix-cli", default="motrix", help="Motrix CLI executable or motrix.js path")
    parser.add_argument("--connections", type=int, default=8, help="Motrix connections per server (1-128)")
    parser.add_argument("--proxy", help="Proxy URL passed to Motrix, for example http://127.0.0.1:7897")
    parser.add_argument("--download-timeout", type=float, default=0, help="Seconds to wait per Motrix file; 0 means unlimited")
    args = parser.parse_args()
    if not 1 <= args.connections <= 128 or args.download_timeout < 0:
        parser.error("--connections must be 1-128 and --download-timeout must be nonnegative")
    download = MotrixDownloader(args.motrix_cli, args.connections, args.proxy, args.download_timeout).fetch if args.downloader == "motrix" else fetch
    def get(url, output, expected=None):
        return checked(url, output, expected, download=download)
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    manifest = {"schemaVersion": 1, "engines": {}}
    source = {
        "onnx": (f"https://huggingface.co/{ONNX}/tree/{ONNX_REV}", "Dynamic UINT8 MatMul weight quantization"),
        "mnn": (MNN_URL, "Q8 weight-only, block 64, MNN 3.6.1"),
        "ncnn": (NCNN_URL, "INT8 mixed model, upstream ncnn2int8 export"),
        "litert": (f"https://huggingface.co/Luigi/sensevoice-litert/tree/{LITERT_REV}", "dynamic_wi8_afp32"),
    }
    for engine in args.engines:
        folder = root / engine
        folder.mkdir(exist_ok=True)
        if engine == "onnx":
            get(f"https://huggingface.co/{ONNX}/resolve/{ONNX_REV}/model.int8.onnx", folder / "model.int8.onnx", hf_hash(ONNX, ONNX_REV, "model.int8.onnx", args.proxy))
            get(f"https://huggingface.co/{ONNX}/resolve/{ONNX_REV}/tokens.txt", folder / "tokens.txt", "f449eb28dc567533d7fa59be34e2abca8784f771850c78a47fb731a31429a1dc")
            get(f"https://huggingface.co/{ONNX}/resolve/{ONNX_REV}/LICENSE", folder / "LICENSE.txt")
            names = ["model.int8.onnx", "tokens.txt", "LICENSE.txt"]
        elif engine == "mnn":
            get(MNN_URL + "model.mnn", folder / "model.mnn", "28b954a62c9f8f8a9ccbe1079b1bb3b1d283fee0c84b311dadc7762cfcbe82bd")
            get(MNN_URL + "tokens.txt", folder / "tokens.txt", "f449eb28dc567533d7fa59be34e2abca8784f771850c78a47fb731a31429a1dc")
            get(MNN_URL + "FunASR-MODEL-LICENSE.txt", folder / "LICENSE.txt")
            get(MNN_URL + "provenance.json", folder / "provenance.json")
            names = ["model.mnn", "tokens.txt", "LICENSE.txt", "provenance.json"]
        elif engine == "ncnn":
            archive = root.parent / "ncnn-int8.tar.bz2"
            get(NCNN_URL, archive)
            names = ["model.ncnn.param", "model.ncnn.bin", "tokens.txt", "LICENSE.txt"]
            with tarfile.open(archive, "r:bz2") as tar:
                for name in names:
                    upstream = "LICENSE" if name == "LICENSE.txt" else name
                    members = [m for m in tar.getmembers() if m.isfile() and Path(m.name).name == upstream]
                    if len(members) != 1:
                        raise ValueError(f"Expected one {upstream} in ncnn archive")
                    with tar.extractfile(members[0]) as src, (folder / name).open("wb") as dst:
                        shutil.copyfileobj(src, dst)
            archive.unlink()
        else:
            names = ["sensevoice_small_q8.tflite", "tokens.txt", "cmvn.json"]
            for name in names:
                get(f"https://huggingface.co/Luigi/sensevoice-litert/resolve/{LITERT_REV}/{name}", folder / name, hf_hash("Luigi/sensevoice-litert", LITERT_REV, name, args.proxy))
            get(f"https://huggingface.co/{ONNX}/resolve/{ONNX_REV}/LICENSE", folder / "LICENSE.txt")
            names += ["LICENSE.txt"]
        files = [{"path": f"{engine}/{name}", "sizeBytes": (folder / name).stat().st_size, "sha256": sha(folder / name)} for name in names]
        manifest["engines"][engine] = {"source": source[engine][0], "quantization": source[engine][1], "files": files}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("Copy this folder to the phone, then import it:", root.resolve())


def hf_hash(repo, revision, filename, proxy=None):
    response = requests.get(f"https://huggingface.co/api/models/{repo}/revision/{revision}", params={"blobs": "true"}, timeout=45, proxies={"http": proxy, "https": proxy} if proxy else None)
    response.raise_for_status()
    item = next(x for x in response.json()["siblings"] if x["rfilename"] == filename)
    return item.get("lfs", {}).get("sha256")


if __name__ == "__main__":
    main()
