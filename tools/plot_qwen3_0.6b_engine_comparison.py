"""Plot the full test-clean Qwen3-ASR 0.6B engine comparison."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
RUNS = [
    ("GGUF DML adapter 1", ROOT / ".native/librispeech/qwen3-asr-0.6b-dml-device-1-full-test-clean/gguf.summary.json"),
    ("TensorRT BF16 + EXL3 2.5 bpw, train-clean-100", ROOT / ".native/librispeech/qwen3-asr-memory-full-test-clean/0.6b/trt-bf16-exl3-2.5bpw/results/tensorrt-exllama.summary.json"),
    ("TensorRT BF16 + EXL3 3 bpw", ROOT / ".native/librispeech/qwen3-asr-memory-full-test-clean/0.6b/trt-bf16-exl3-3bpw/results/tensorrt-exllama.summary.json"),
    ("TensorRT BF16 + EXL3 4 bpw", ROOT / ".native/librispeech/qwen3-asr-memory-full-test-clean/0.6b/trt-bf16-exl3-4bpw/results/tensorrt-exllama.summary.json"),
]
COLORS = ["#3b6ea5", "#d98b3a", "#c44e52", "#4c956c"]


def main():
    summaries = []
    reference_ids = None
    reference_texts = None
    for _, summary_path in RUNS:
        summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))
        result_path = summary_path.with_name(summary_path.name.replace(".summary.json", ".jsonl"))
        rows = [json.loads(line) for line in result_path.read_text(encoding="utf-8").splitlines()]
        ids = [row["id"] for row in rows]
        references = [row["reference"] for row in rows]
        if len(ids) != 2620 or len(set(ids)) != 2620:
            raise ValueError(f"Expected 2,620 unique rows in {result_path}")
        if reference_ids is None:
            reference_ids, reference_texts = ids, references
        elif ids != reference_ids or references != reference_texts:
            raise ValueError(f"Run does not match the shared sample: {result_path}")
    if any(item["utterances"] != 2620 for item in summaries):
        raise ValueError("Expected full LibriSpeech test-clean summaries")

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.edgecolor": "#c9ced6", "axes.linewidth": 0.8,
        "figure.facecolor": "#ffffff", "svg.fonttype": "none",
    })
    labels = [item[0] for item in RUNS]
    wer = np.array([item["corpusWER"] * 100 for item in summaries])
    median = np.array([item["medianMs"] for item in summaries])
    p90 = np.array([item["p90Ms"] for item in summaries])
    elapsed = np.array([item["elapsedSeconds"] for item in summaries])

    fig, ax = plt.subplots(figsize=(8.8, 6.2))
    fig.subplots_adjust(left=0.14, right=0.96, top=0.83, bottom=0.15)
    for label, x, y, color, offset in zip(
        ["GGUF DML 1", "EXL3 2.5 cal.", "EXL3 3 bpw", "EXL3 4 bpw"],
        median, wer, COLORS, [(8, 8), (8, 8), (8, 8), (8, -15)],
    ):
        ax.scatter(x, y, s=115, color=color, edgecolor="white", linewidth=1, zorder=3)
        ax.annotate(label, (x, y), xytext=offset, textcoords="offset points", fontsize=8)
    spread = wer.max() - wer.min()
    ax.set_ylim(wer.min() - spread * 0.12, wer.max() + spread * 0.12)
    ax.set_xlabel("Median end-to-end latency (ms) · lower is better")
    ax.set_ylabel("Corpus WER (%) · lower is better")
    ax.grid(color="#e4e8ee", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.02, 0.95, "Qwen3-ASR 0.6B · speed and accuracy", fontsize=16,
             fontweight="bold", color="#1d2733")
    fig.text(0.02, 0.90, "LibriSpeech test-clean · 2,620 utterances · lower-left is better",
             fontsize=9.5, color="#5b6675")
    for ext, kwargs in (("png", {"dpi": 220}), ("svg", {})):
        fig.savefig(DOCS / f"windows-librispeech-qwen3-0.6b-speed-accuracy.{ext}",
                    bbox_inches="tight", **kwargs)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 5.8))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.78, bottom=0.27, wspace=0.32)
    for ax, values, title in zip(
        axes, (median, p90, wer), ("Median latency (ms)", "P90 latency (ms)", "Corpus WER (%)")
    ):
        bars = ax.bar(np.arange(len(labels)), values, color=COLORS, width=0.68)
        ax.set_title(title, fontsize=10)
        ax.set_xticks(np.arange(len(labels)), ["GGUF\nDML 1", "EXL3\n2.5 cal.", "EXL3\n3 bpw", "EXL3\n4 bpw"], fontsize=8)
        ax.grid(axis="y", color="#e4e8ee", linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.2f}",
                    ha="center", va="bottom", fontsize=8)
    fig.text(0.02, 0.94, "Qwen3-ASR 0.6B · engine comparison", fontsize=16,
             fontweight="bold", color="#1d2733")
    fig.text(0.02, 0.89, f"LibriSpeech test-clean · 2,620 utterances · full runs: "
             + " · ".join(f"{value:.0f}s" for value in elapsed),
             fontsize=9, color="#5b6675")
    for ext, kwargs in (("png", {"dpi": 220}), ("svg", {})):
        fig.savefig(DOCS / f"windows-librispeech-qwen3-0.6b-comparison.{ext}",
                    bbox_inches="tight", **kwargs)
    plt.close(fig)


if __name__ == "__main__":
    main()
