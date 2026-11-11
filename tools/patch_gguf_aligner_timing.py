"""Add finer stage timers to the vendored Qwen3-ASR-GGUF aligner.

The upstream ``qwen_asr_gguf/inference/aligner.py`` times only ``ctx.decode``.
The embedding-table gather, the ``LlamaBatch`` construction, the KV-cache clear
and the per-timestamp argmax then fall outside every reported stage and end up in
the residual "preprocessing" bucket. This script rewrites that file so the
decoder figure covers the same work as the TensorRT backend's decoder window:

    decoder_time = embed_time + decode_time

and exposes ``embed_time``, ``decode_time`` and ``argmax_time`` separately.

Usage:
    python tools/patch_gguf_aligner_timing.py .native/qwen3-asr-gguf

The patch is idempotent: running it twice is a no-op. It only touches a clone
under ``.native``; the upstream repository is unchanged.
"""
import argparse
import sys
from pathlib import Path

MARKER = "t_embed_start = time.time()"

PATCHES = [
    (
        "        audio_embd, t_enc = self.encoder.encode(audio)\n",
        "        audio_embd, t_enc = self.encoder.encode(audio)\n"
        "        t_embed_start = time.time()\n",
    ),
    (
        "        full_embd[len(pre_ids)+audio_embd.shape[0]:] = self.embedding_table[post_ids]\n",
        "        full_embd[len(pre_ids)+audio_embd.shape[0]:] = self.embedding_table[post_ids]\n"
        "        t_embed = time.time() - t_embed_start\n",
    ),
    (
        "        t_dec = time.time() - t_dec_start\n",
        "        t_dec = time.time() - t_dec_start\n"
        "        t_argmax_start = time.time()\n",
    ),
    (
        "        del batch\n",
        "        t_argmax = time.time() - t_argmax_start\n\n        del batch\n",
    ),
    (
        """            performance={
                "encoder_time": t_enc,
                "decoder_time": t_dec,
                "total_time": t_total
            }""",
        """            performance={
                "encoder_time": t_enc,
                "decoder_time": t_embed + t_dec,
                "embed_time": t_embed,
                "decode_time": t_dec,
                "argmax_time": t_argmax,
                "total_time": t_total
            }""",
    ),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", type=Path, help="path to the qwen3-asr-gguf checkout")
    args = parser.parse_args()

    target = args.repo / "qwen_asr_gguf" / "inference" / "aligner.py"
    if not target.is_file():
        print(f"aligner.py not found: {target}", file=sys.stderr)
        return 1

    text = target.read_text(encoding="utf-8")
    if MARKER in text:
        print(f"already patched: {target}")
        return 0

    for old, new in PATCHES:
        if text.count(old) != 1:
            print(f"anchor not unique or missing:\n{old!r}", file=sys.stderr)
            return 1
        text = text.replace(old, new)

    target.write_text(text, encoding="utf-8", newline="")
    print(f"patched: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
