"""Extract one fixed LibriSpeech utterance, without unpacking the dataset."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile
import wave

UTTERANCE = "3729-6852-0008"
PREFIX = "LibriSpeech/test-clean/3729/6852"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path, default=Path("app/src/main/assets/audio"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    wanted = {
        f"{PREFIX}/{UTTERANCE}.flac": "audio.flac",
        f"{PREFIX}/3729-6852.trans.txt": "transcript.txt",
        "LibriSpeech/LICENSE.TXT": "LICENSE.txt",
    }
    with tempfile.TemporaryDirectory() as tmp:
        found = set()
        with tarfile.open(args.archive, "r|gz") as archive:
            for member in archive:
                if member.name not in wanted:
                    continue
                if not member.isfile():
                    raise ValueError(f"Expected regular file: {member.name}")
                data = archive.extractfile(member).read()
                Path(tmp, wanted[member.name]).write_bytes(data)
                found.add(member.name)
                if found == set(wanted):
                    break
        if found != set(wanted):
            raise ValueError(f"Missing archive entries: {set(wanted) - found}")
        reference = next(
            line.split(" ", 1)[1]
            for line in Path(tmp, "transcript.txt").read_text().splitlines()
            if line.startswith(UTTERANCE + " ")
        )
        output = args.output / "sample.wav"
        subprocess.run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(Path(tmp, "audio.flac")), "-map_metadata", "-1",
            "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", "-bitexact", str(output),
        ], check=True)
        with wave.open(str(output)) as audio:
            assert audio.getnchannels() == 1 and audio.getsampwidth() == 2
            metadata = {
                "dataset": "LibriSpeech test-clean", "utterance": UTTERANCE,
                "sourceEntry": f"{PREFIX}/{UTTERANCE}.flac",
                "reference": reference, "sampleRate": audio.getframerate(),
                "frames": audio.getnframes(),
                "durationSeconds": audio.getnframes() / audio.getframerate(),
                "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                "license": "CC BY 4.0", "source": "https://www.openslr.org/12/",
                "selection": "Complete fixed utterance; no trimming, denoising or normalization",
            }
        (args.output / "sample.json").write_text(json.dumps(metadata, indent=2) + "\n")
        (args.output / "LICENSE.txt").write_bytes(Path(tmp, "LICENSE.txt").read_bytes())
        print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
