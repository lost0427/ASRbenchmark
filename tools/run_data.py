"""Locate benchmark run outputs for the plotting scripts.

The raw run directories live in the git-ignored ``.native/librispeech`` tree on
the machine that produced them. A snapshot of the directories the plotting
scripts read is committed under ``data/runs`` so the figures can be regenerated
from a fresh clone.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / ".native" / "librispeech"
SNAPSHOT = ROOT / "data" / "runs"


def resolve_runs() -> Path:
    """Return the run-output root: the live tree if present, else the snapshot."""
    return NATIVE if NATIVE.is_dir() else SNAPSHOT


def resolve_manifest() -> Path:
    """Return the manifest used for sampling: live tree first, else the snapshot."""
    live = NATIVE / "manifest.jsonl"
    if live.is_file():
        return live
    return ROOT / "data" / "manifest.jsonl"
