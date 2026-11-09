"""Summarize full-corpus Qwen3-ASR engine runs and draw comparison charts."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from windows_benchmark import normalize

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / ".native/librispeech/qwen3-asr-1.7b-full-test-clean"
MEMORY_RUNS = ROOT / ".native/librispeech/qwen3-asr-memory-full-test-clean/1.7b"
DOCS = ROOT / "docs"
CONFIGS = [
    ("GGUF INT4 + Q4_K, DML adapter 1", ROOT / ".native/librispeech/qwen3-asr-1.7b-dml-device-1-full-test-clean", "gguf"),
    ("TensorRT BF16 + EXL3 2.5 bpw, train-clean-100",
     MEMORY_RUNS / "trt-bf16-exl3-2.5bpw-train100/results", "tensorrt-exllama"),
    ("TensorRT BF16 + EXL3 2 bpw, train-clean-100",
     MEMORY_RUNS / "trt-bf16-exl3-2bpw-train100", "tensorrt-exllama"),
    ("TensorRT BF16 + EXL3 3 bpw", MEMORY_RUNS / "trt-bf16-exl3-3bpw/results", "tensorrt-exllama"),
    ("TensorRT BF16 + EXL3 3 bpw, train-clean-100",
     MEMORY_RUNS / "trt-bf16-exl3-3bpw-train100/results", "tensorrt-exllama"),
    ("TensorRT BF16 + EXL3 4 bpw", MEMORY_RUNS / "trt-bf16-exl3-4bpw/results", "tensorrt-exllama"),
]
COLORS = ["#3b6ea5", "#d98b3a", "#c0a020", "#c44e52", "#b279a2", "#4c956c"]


def main():
    sample = [json.loads(line) for line in
              (MEMORY_RUNS / "gguf-int4-q4k/results/sample.jsonl").read_text(encoding="utf-8").splitlines()]
    ids = [row["id"] for row in sample]
    if len(ids) != 2620 or len(set(ids)) != len(ids):
        raise ValueError(f"Expected 2,620 unique utterances, got {len(ids)} rows")

    summaries = {}
    runs = {}
    for label, directory, stem in CONFIGS:
        summary = json.loads((directory / f"{stem}.summary.json").read_text(encoding="utf-8"))
        rows = [json.loads(line) for line in (directory / f"{stem}.jsonl").read_text(encoding="utf-8").splitlines()]
        if [row["id"] for row in rows] != ids:
            raise ValueError(f"{label} IDs do not exactly match the shared sample")
        if any(row["reference"] != ref["reference"] for row, ref in zip(rows, sample)):
            raise ValueError(f"{label} references do not match the shared sample")
        summaries[label] = summary
        runs[label] = rows

    baseline = next(iter(runs.values()))
    pairwise = {}
    for label, rows in list(runs.items())[1:]:
        pairwise[label] = {
            "editWinsTiesLossesVsGGUF": [
                sum(row["edits"] < base["edits"] for row, base in zip(rows, baseline)),
                sum(row["edits"] == base["edits"] for row, base in zip(rows, baseline)),
                sum(row["edits"] > base["edits"] for row, base in zip(rows, baseline)),
            ],
            "exactSameNormalizedText": sum(
                normalize(row["hypothesis"]) == normalize(base["hypothesis"])
                for row, base in zip(rows, baseline)
            ),
        }
    three = runs["TensorRT BF16 + EXL3 3 bpw"]
    calibrated_three = runs["TensorRT BF16 + EXL3 3 bpw, train-clean-100"]
    four = runs["TensorRT BF16 + EXL3 4 bpw"]
    pairwise["TensorRT BF16 + EXL3 3 bpw vs 4 bpw"] = {
        "editWinsTiesLosses": [
            sum(a["edits"] < b["edits"] for a, b in zip(three, four)),
            sum(a["edits"] == b["edits"] for a, b in zip(three, four)),
            sum(a["edits"] > b["edits"] for a, b in zip(three, four)),
        ],
        "exactSameNormalizedText": sum(
            normalize(a["hypothesis"]) == normalize(b["hypothesis"])
            for a, b in zip(three, four)
        ),
    }
    pairwise["TensorRT BF16 + EXL3 3 bpw, train-clean-100 vs legacy 3 bpw"] = {
        "editWinsTiesLosses": [
            sum(a["edits"] < b["edits"] for a, b in zip(calibrated_three, three)),
            sum(a["edits"] == b["edits"] for a, b in zip(calibrated_three, three)),
            sum(a["edits"] > b["edits"] for a, b in zip(calibrated_three, three)),
        ],
        "exactSameNormalizedText": sum(
            normalize(a["hypothesis"]) == normalize(b["hypothesis"])
            for a, b in zip(calibrated_three, three)
        ),
    }

    result = {
        "dataset": "LibriSpeech test-clean",
        "model": "Qwen3-ASR-1.7B",
        "utterances": len(sample),
        "referenceWords": summaries[next(iter(summaries))]["referenceWords"],
        "idAlignment": "all IDs and references match the shared sample",
        "engines": summaries,
        "pairwise": pairwise,
    }
    (RUNS / "comparison.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.edgecolor": "#c9ced6", "axes.linewidth": 0.8,
        "figure.facecolor": "#ffffff", "svg.fonttype": "none",
    })
    labels = list(summaries)
    wer = np.array([summaries[label]["corpusWER"] * 100 for label in labels])
    median = np.array([summaries[label]["medianMs"] for label in labels])
    p90 = np.array([summaries[label]["p90Ms"] for label in labels])
    full = np.array([summaries[label]["elapsedSeconds"] for label in labels])
    colors = COLORS[:len(labels)]

    fig, ax = plt.subplots(figsize=(8.8, 6.2))
    fig.subplots_adjust(left=0.14, right=0.96, top=0.83, bottom=0.15)
    plot_labels = ["GGUF DML 1", "EXL3 2.5 cal.", "EXL3 2 cal.",
                   "EXL3 3 bpw", "EXL3 3 bpw cal.", "EXL3 4 bpw"]
    annotation_offsets = [(8, 8), (8, 8), (8, 8), (-48, -17), (8, 8), (8, -17)]
    for label, x, y, color, plot_label, offset in zip(
        labels, median, wer, colors, plot_labels, annotation_offsets
    ):
        ax.scatter(x, y, s=115, color=color, edgecolor="white", linewidth=1.0, zorder=3)
        ax.annotate(plot_label, (x, y), xytext=offset, textcoords="offset points", fontsize=8)
    spread = wer.max() - wer.min()
    ax.set_ylim(wer.min() - spread * 0.12, wer.max() + spread * 0.12)
    ax.set_xlabel("Median end-to-end latency (ms) · lower is better")
    ax.set_ylabel("Corpus WER (%) · lower is better")
    ax.grid(color="#e4e8ee", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.02, 0.95, "Qwen3-ASR 1.7B · speed and accuracy", fontsize=16,
             fontweight="bold", color="#1d2733")
    fig.text(0.02, 0.90, "LibriSpeech test-clean · 2,620 utterances · lower-left is better",
             fontsize=9.5, color="#5b6675")
    fig.savefig(DOCS / "windows-librispeech-qwen3-1.7b-speed-accuracy.png", dpi=220, bbox_inches="tight")
    fig.savefig(DOCS / "windows-librispeech-qwen3-1.7b-speed-accuracy.svg", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.8))
    fig.subplots_adjust(left=0.07, right=0.99, top=0.79, bottom=0.30, wspace=0.32)
    panels = [(median, "Median latency (ms)"), (p90, "P90 latency (ms)"), (wer, "Corpus WER (%)")]
    for ax, (values, title) in zip(axes, panels):
        bars = ax.bar(np.arange(len(labels)), values, color=colors, width=0.68)
        ax.set_title(title, fontsize=10)
        ax.set_xticks(np.arange(len(labels)), [
            "GGUF", "2.5\nbpw cal.", "2\nbpw cal.", "3 bpw", "3 bpw\ncal.", "4 bpw"
        ], fontsize=8)
        ax.grid(axis="y", color="#e4e8ee", linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.2f}",
                    ha="center", va="bottom", fontsize=8)
    fig.text(0.02, 0.94, "Qwen3-ASR 1.7B · engine comparison", fontsize=16,
             fontweight="bold", color="#1d2733")
    fig.text(0.02, 0.89, f"LibriSpeech test-clean · {len(sample):,} utterances · full runs: "
             + " · ".join(f"{value:.0f}s" for value in full), fontsize=9, color="#5b6675")
    fig.savefig(DOCS / "windows-librispeech-qwen3-1.7b-comparison.png", dpi=220, bbox_inches="tight")
    fig.savefig(DOCS / "windows-librispeech-qwen3-1.7b-comparison.svg", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
