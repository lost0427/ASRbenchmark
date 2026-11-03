"""Download and unpack the Qwen3-ASR-1.7B benchmark models."""
import argparse
import os
import shutil
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / ".native/models/qwen3-asr"
TRANSCRIBE_REPO = "handy-computer/Qwen3-ASR-1.7B-gguf"
ORT_REPO = "tonythethompson/Qwen3-ASR-1.7B-ONNX"
ORT_REVISION = "6c07a3dfed80e1cc64e10cad108b00c2080ae19b"
SHERPA_REPO = "thieunv-asilla/sherpa-onnx-qwen3-asr-1.7B-int8"
SHERPA_REVISION = "69eb686fd94a4a865bb5340a3d6ac0d7f1fec0d5"
QWEN_REPO = "Qwen/Qwen3-ASR-1.7B"
QWEN_REVISION = "7278e1e70fe206f11671096ffdd38061171dd6e5"


def download(repo, revision, filename, destination, force):
    from huggingface_hub import hf_hub_download
    destination = Path(destination)
    if destination.is_file() and destination.stat().st_size > 0 and not force:
        print(f"skip {destination}")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(repo_id=repo, filename=filename, revision=revision,
                           local_dir=destination.parent)
    path = Path(path)
    if path.resolve() != destination.resolve():
        shutil.move(str(path), str(destination))
    print(f"download {destination} ({destination.stat().st_size} bytes)")
    return destination


def safe_extract(archive, destination, force):
    destination = Path(destination)
    if destination.exists() and any(destination.iterdir()) and not force:
        print(f"skip {destination}")
        return
    if destination.exists() and force:
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
        temporary = Path(temporary)
        with tarfile.open(archive, "r:*") as tar:
            for member in tar.getmembers():
                target = (temporary / member.name).resolve()
                if temporary.resolve() not in target.parents and target != temporary.resolve():
                    raise SystemExit(f"unsafe archive member: {member.name}")
            tar.extractall(temporary)
        entries = list(temporary.iterdir())
        source = entries[0] if len(entries) == 1 and entries[0].is_dir() else temporary
        shutil.move(str(source), str(destination))
    print(f"extract {archive} -> {destination}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--components", choices=["all", "transcribe", "ort", "sherpa"],
                        default="all")
    parser.add_argument("--endpoint", default="https://hf-mirror.com")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    os.environ["HF_ENDPOINT"] = args.endpoint
    root = args.root.resolve()

    if args.components in ("all", "transcribe"):
        for name in ("Qwen3-ASR-1.7B-Q8_0.gguf", "Qwen3-ASR-1.7B-F16.gguf",
                     "Qwen3-ASR-1.7B-BF16.gguf"):
            download(TRANSCRIBE_REPO, "main", name, root / "transcribe" / name, args.force)

    if args.components in ("all", "ort"):
        archive = download(ORT_REPO, ORT_REVISION, "qwen3-asr-1.7b-int4.tar.gz",
                           root / "ort-int4" / "qwen3-asr-1.7b-int4.tar.gz", args.force)
        safe_extract(archive, root / "ort-int4" / "extracted", args.force)
        archive = download(ORT_REPO, ORT_REVISION, "qwen3-asr-1.7b.tar.gz",
                           root / "ort-fp16" / "qwen3-asr-1.7b.tar.gz", args.force)
        temporary = root / "ort-fp16" / "unpacked"
        safe_extract(archive, temporary, args.force)
        if temporary.is_dir():
            for item in temporary.iterdir():
                target = root / "ort-fp16" / item.name
                if target.exists() and args.force:
                    if target.is_dir():
                        shutil.rmtree(target)
                    else:
                        target.unlink()
                if not target.exists():
                    shutil.move(str(item), str(target))
            shutil.rmtree(temporary)

    if args.components in ("all", "sherpa"):
        names = ("conv_frontend.onnx", "encoder.int8.onnx", "decoder.int8.onnx",
                 "tokenizer/merges.txt", "tokenizer/tokenizer_config.json",
                 "tokenizer/vocab.json")
        for name in names:
            download(SHERPA_REPO, SHERPA_REVISION, name, root / "sherpa" / name, args.force)

    download(QWEN_REPO, QWEN_REVISION, "chat_template.json", root / "chat_template.json",
             args.force)


if __name__ == "__main__":
    main()
