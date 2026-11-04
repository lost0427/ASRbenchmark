"""Build a LibriSpeech test-clean manifest from an extracted archive."""
import argparse
import json
from pathlib import Path

import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / ".native/librispeech"
DEFAULT_ARCHIVE = ROOT / "test-clean.tar.gz"


def parse_chapter(path):
    references = {}
    for line in path.read_text(encoding="utf-8", errors="strict").splitlines():
        if not line.strip():
            continue
        utterance, text = line.split(" ", 1)
        if utterance in references:
            raise SystemExit(f"Duplicate transcript {utterance} in {path}")
        references[utterance] = text
    return references


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    dataset = args.data / "LibriSpeech/test-clean"
    if not dataset.is_dir():
        raise SystemExit(f"Extract {args.archive} under {args.data} first")
    output = args.output or args.data / "manifest.jsonl"
    references = {}
    chapters = list(dataset.rglob("*.trans.txt"))
    for chapter in chapters:
        references.update(parse_chapter(chapter))
    flac_files = sorted(dataset.rglob("*.flac"))
    rows = []
    missing = []
    duplicates = set()
    for flac in flac_files:
        utterance = flac.stem
        if utterance in duplicates:
            raise SystemExit(f"Duplicate audio {utterance}")
        duplicates.add(utterance)
        reference = references.get(utterance)
        if reference is None:
            missing.append(utterance)
            continue
        info = sf.info(flac)
        if info.channels != 1:
            raise SystemExit(f"Not mono: {flac}")
        parts = utterance.split("-")
        rows.append({
            "id": utterance,
            "speaker": parts[0],
            "chapter": parts[1],
            "flac": str(flac.relative_to(args.data)).replace("\\", "/"),
            "reference": reference,
            "sampleRate": info.samplerate,
            "durationSeconds": info.frames / info.samplerate,
        })
    if missing:
        raise SystemExit(f"Missing transcripts: {missing[:5]} ({len(missing)})")
    extra = set(references) - {row["id"] for row in rows}
    if extra:
        raise SystemExit(f"Transcripts without audio: {sorted(extra)[:5]} ({len(extra)})")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "dataset": "LibriSpeech test-clean",
        "utterances": len(rows),
        "chapters": len(chapters),
        "speakers": len({row["speaker"] for row in rows}),
        "durationSeconds": sum(row["durationSeconds"] for row in rows),
        "sampleRate": sorted({row["sampleRate"] for row in rows}),
        "manifest": str(output),
    }
    (output.with_suffix(".summary.json")).write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
