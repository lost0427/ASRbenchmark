"""Compare the official and TensorRT+EXL3 Qwen3-ForcedAligner backends."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / ".native/librispeech"
DOCS = ROOT / "docs"
OFFICIAL = RUNS / "qwen3-asr-aligner-official-full-test-clean"
TENSORRT = RUNS / "qwen3-asr-aligner-tensorrt-full-test-clean"


def load(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def summary(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    official = load(OFFICIAL / "official.jsonl")
    tensorrt = load(TENSORRT / "tensorrt.jsonl")
    differences = []
    exact = 0
    mismatched = 0
    for left, right in zip(official, tensorrt):
        if left["id"] != right["id"] or [i["text"] for i in left["items"]] != [i["text"] for i in right["items"]]:
            mismatched += 1
            continue
        for a, b in zip(left["items"], right["items"]):
            for key in ("start_time", "end_time"):
                value = abs(a[key] - b[key])
                differences.append(value)
                exact += value < 1e-9
    summary_official = summary(OFFICIAL / "official.summary.json")
    summary_tensorrt = summary(TENSORRT / "tensorrt.summary.json")

    report = {
        "benchmark": "LibriSpeech test-clean",
        "model": "Qwen3-ForcedAligner-0.6B",
        "conditions": {
            "utterances": len(official),
            "items": sum(len(row["items"]) for row in official),
            "seed": 1337,
            "warmupUtterances": 2,
        },
        "backends": {
            "official": summary_official,
            "tensorrt": summary_tensorrt,
        },
        "agreement": {
            "tokenMismatchUtterances": mismatched,
            "boundaryMaeSeconds": float(np.mean(differences)),
            "boundaryP90Seconds": float(np.percentile(differences, 90)),
            "boundaryMaxSeconds": float(np.max(differences)),
            "exactBoundaryFraction": exact / max(1, len(differences)),
        },
    }
    (DOCS / "qwen3-aligner-comparison.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.edgecolor": "#c9ced6", "axes.linewidth": 0.8,
        "figure.facecolor": "#ffffff", "svg.fonttype": "none",
    })
    labels = ["Official\nTransformers", "TensorRT BF16 +\nEXL3 4 bpw"]
    colors = ["#8c8c8c", "#3b6ea5"]
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.8))
    fig.subplots_adjust(left=0.07, right=0.985, top=0.76, bottom=0.26, wspace=0.34)
    metrics = [
        ("Median latency (ms)", [summary_official["medianMs"], summary_tensorrt["medianMs"]], ".2f"),
        ("P90 latency (ms)", [summary_official["p90Ms"], summary_tensorrt["p90Ms"]], ".2f"),
        ("Boundary MAE (ms)", [0.0, report["agreement"]["boundaryMaeSeconds"] * 1000], ".2f"),
    ]
    for ax, (title, values, number_format) in zip(axes, metrics):
        bars = ax.bar(np.arange(2), values, color=colors, width=0.6)
        ax.set_title(title, fontsize=10)
        ax.set_xticks(np.arange(2), labels, fontsize=8)
        ax.grid(axis="y", color="#e4e8ee", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(0, max(max(values) * 1.25, 0.1))
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.025,
                    format(value, number_format), ha="center", va="bottom", fontsize=8)
    fig.text(0.02, 0.95, "Qwen3-ForcedAligner 0.6B · official vs TensorRT + EXL3",
             fontsize=15, fontweight="bold", color="#1d2733")
    fig.text(0.02, 0.89,
             "LibriSpeech test-clean · 2,620 utterances · boundary MAE measured against the official backend",
             fontsize=8.8, color="#5b6675")
    fig.text(0.02, 0.05,
             f"Token mismatch: {mismatched} utterances · exact boundaries: "
             f"{report['agreement']['exactBoundaryFraction'] * 100:.1f}%",
             fontsize=8.0, color="#6b7685")
    for ext, options in (("png", {"dpi": 220}), ("svg", {})):
        path = DOCS / f"windows-librispeech-qwen3-aligner-comparison.{ext}"
        fig.savefig(path, bbox_inches="tight", **options)
        if ext == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8")
    plt.close(fig)

    stages = summary_tensorrt["medianStagesMs"]
    figure, ax = plt.subplots(figsize=(7.8, 5.6))
    figure.subplots_adjust(left=0.13, right=0.96, top=0.78, bottom=0.2)
    total = summary_tensorrt["medianMs"]
    pieces = [
        ("Preprocess", stages["preprocessingMs"], "#8b9aa8"),
        ("Encoder", stages["encoderMs"], "#3b6ea5"),
        ("Decoder", stages["decoderMs"], "#c44e52"),
        ("Other", max(0.0, total - stages["preprocessingMs"] - stages["encoderMs"] - stages["decoderMs"]), "#bdc3c7"),
    ]
    bottom = 0.0
    for name, value, color in pieces:
        ax.bar([0], [value], bottom=bottom, color=color, width=0.55, label=name)
        if value > 1:
            ax.text(0, bottom + value / 2, f"{value:.1f}", ha="center", va="center",
                    fontsize=8, color="white" if name in ("Encoder", "Decoder") else "#26313d")
        bottom += value
    ax.axhline(summary_official["medianMs"], color="#8c8c8c", linestyle="--", linewidth=1.2,
               label="Official total")
    ax.text(0.32, summary_official["medianMs"], f"official {summary_official['medianMs']:.1f} ms",
            va="bottom", fontsize=8, color="#5b6675")
    ax.set_xticks([0], ["TensorRT +\nEXL3 4 bpw"])
    ax.set_ylabel("Median latency (ms)")
    ax.set_ylim(0, summary_official["medianMs"] * 1.22)
    ax.grid(axis="y", color="#e4e8ee", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.set_title(f"TensorRT + EXL3 total: {total:.1f} ms", fontsize=10)
    figure.text(0.02, 0.95, "Qwen3-ForcedAligner 0.6B · median latency by stage",
                fontsize=15, fontweight="bold", color="#1d2733")
    figure.text(0.02, 0.90, "LibriSpeech test-clean · 2,620 utterances · TensorRT encoder + EXL3 decoder",
                fontsize=8.8, color="#5b6675")
    for ext, options in (("png", {"dpi": 220}), ("svg", {})):
        path = DOCS / f"windows-librispeech-qwen3-aligner-stage-latency.{ext}"
        figure.savefig(path, bbox_inches="tight", **options)
        if ext == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8")
    plt.close(figure)
    print(json.dumps(report["agreement"], indent=2))


if __name__ == "__main__":
    main()
