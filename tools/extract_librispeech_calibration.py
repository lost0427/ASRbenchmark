"""Extract calibration chapters from LibriSpeech train-clean-100."""
import argparse
import random
import tarfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--speakers", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()
    if not args.archive.is_file() or args.speakers < 1:
        parser.error("archive must exist and speakers must be positive")

    prefix = "LibriSpeech/train-clean-100/"
    with tarfile.open(args.archive, "r:gz") as archive:
        chapters = {}
        for member in archive:
            name = member.name.replace("\\", "/")
            if not name.startswith(prefix) or not name.endswith(".trans.txt"):
                continue
            relative = name[len(prefix):]
            parts = relative.split("/")
            if len(parts) == 3:
                chapters.setdefault(parts[0], []).append((name, "/".join(parts[:2])))
        speakers = sorted(chapters)
        if len(speakers) < args.speakers:
            raise RuntimeError(f"Archive contains only {len(speakers)} speakers")
        rng = random.Random(args.seed)
        selected = []
        for speaker in sorted(rng.sample(speakers, args.speakers)):
            selected.append(rng.choice(chapters[speaker]))

        selected_dirs = {directory for _, directory in selected}
        selected_keys = {tuple(directory.split("/")) for directory in selected_dirs}
        targets = []
        for member in archive.getmembers():
            name = member.name.replace("\\", "/")
            if not name.startswith(prefix):
                continue
            relative = name[len(prefix):]
            parts = relative.split("/")
            if (
                len(parts) == 3
                and (parts[0], parts[1]) in selected_keys
                and member.isfile()
                and (name.endswith(".flac") or name.endswith(".trans.txt"))
            ):
                targets.append(member)

        root = args.output.resolve()
        extracted = 0
        for member in targets:
            relative = member.name[len(prefix):]
            destination = (root / relative).resolve()
            if not destination.is_relative_to(root):
                raise RuntimeError(f"Unsafe archive path: {member.name}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise RuntimeError(f"Could not read archive member: {member.name}")
            with source, destination.open("wb") as output:
                while chunk := source.read(1024 * 1024):
                    output.write(chunk)
            extracted += 1

    print(f"speakers={args.speakers} chapters={len(selected_dirs)} files={extracted} output={root}")


if __name__ == "__main__":
    main()
