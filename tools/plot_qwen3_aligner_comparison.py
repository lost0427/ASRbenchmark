"""Compare the official, GGUF, and TensorRT+EXL3 Qwen3-ForcedAligner backends."""
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / ".native/librispeech"
DOCS = ROOT / "docs"
RUN_DIRS = {
    "Official Transformers": RUNS / "qwen3-asr-aligner-official-full-test-clean",
    "GGUF INT4 + Q4_K": RUNS / "qwen3-asr-aligner-gguf-full-test-clean",
    "TensorRT BF16 + EXL3 4 bpw": RUNS / "qwen3-asr-aligner-tensorrt-full-test-clean",
}
FILES = {
    "Official Transformers": ("official.jsonl", "official.summary.json"),
    "GGUF INT4 + Q4_K": ("gguf.jsonl", "gguf.summary.json"),
    "TensorRT BF16 + EXL3 4 bpw": ("tensorrt.jsonl", "tensorrt.summary.json"),
}


def load(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def normalize(text):
    return re.sub(r"[^A-Z0-9' ]", " ", text.upper()).split()


def flatten(items):
    words = []
    for item in items:
        tokens = normalize(item["text"])
        if not tokens:
            continue
        if len(tokens) == 1:
            words.append((tokens[0], item["start_time"], item["end_time"]))
        else:
            step = (item["end_time"] - item["start_time"]) / len(tokens)
            for index, token in enumerate(tokens):
                words.append((token, item["start_time"] + step * index,
                              item["start_time"] + step * (index + 1)))
    return words


def main():
    data = {}
    summaries = {}
    for name, directory in RUN_DIRS.items():
        jsonl, summary_name = FILES[name]
        data[name] = load(directory / jsonl)
        summaries[name] = json.loads((directory / summary_name).read_text(encoding="utf-8"))

    reference = data["Official Transformers"]
    agreement = {}
    for name in ("GGUF INT4 + Q4_K", "TensorRT BF16 + EXL3 4 bpw"):
        differences = []
        exact = 0
        mismatched = 0
        for left, right in zip(reference, data[name]):
            left_words = flatten(left["items"])
            right_words = flatten(right["items"])
            if [word[0] for word in left_words] != [word[0] for word in right_words]:
                mismatched += 1
                continue
            for a, b in zip(left_words, right_words):
                for offset in (1, 2):
                    value = abs(a[offset] - b[offset])
                    differences.append(value)
                    exact += value < 1e-9
        agreement[name] = {
            "tokenMismatchUtterances": mismatched,
            "boundaryMaeSeconds": float(np.mean(differences)),
            "boundaryP90Seconds": float(np.percentile(differences, 90)),
            "boundaryMaxSeconds": float(np.max(differences)),
            "exactBoundaryFraction": exact / max(1, len(differences)),
        }
    agreement["Official Transformers"] = {
        "tokenMismatchUtterances": 0,
        "boundaryMaeSeconds": 0.0,
        "boundaryP90Seconds": 0.0,
        "boundaryMaxSeconds": 0.0,
        "exactBoundaryFraction": 1.0,
    }

    report = {
        "benchmark": "LibriSpeech test-clean",
        "model": "Qwen3-ForcedAligner-0.6B",
        "conditions": {
            "utterances": len(reference),
            "items": sum(len(row["items"]) for row in reference),
            "seed": 1337,
            "warmupUtterances": 2,
        },
        "backends": summaries,
        "agreement": agreement,
    }
    (DOCS / "qwen3-aligner-comparison.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.edgecolor": "#c9ced6", "axes.linewidth": 0.8,
        "figure.facecolor": "#ffffff", "svg.fonttype": "none",
    })
    labels = ["Official\nTransformers", "GGUF\nINT4 + Q4_K", "TensorRT BF16\n+ EXL3 4 bpw"]
    colors = ["#8c8c8c", "#d98b3a", "#3b6ea5"]
    names = list(RUN_DIRS)
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 5.0))
    fig.subplots_adjust(left=0.07, right=0.985, top=0.76, bottom=0.27, wspace=0.34)
    metrics = [
        ("Median latency (ms)", [summaries[name]["medianMs"] for name in names], ".2f"),
        ("P90 latency (ms)", [summaries[name]["p90Ms"] for name in names], ".2f"),
        ("Boundary MAE (ms)", [agreement[name]["boundaryMaeSeconds"] * 1000 for name in names], ".2f"),
    ]
    for ax, (title, values, number_format) in zip(axes, metrics):
        bars = ax.bar(np.arange(3), values, color=colors, width=0.62)
        ax.set_title(title, fontsize=10)
        ax.set_xticks(np.arange(3), labels, fontsize=8)
        ax.grid(axis="y", color="#e4e8ee", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(0, max(max(values) * 1.25, 0.1))
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.025,
                    format(value, number_format), ha="center", va="bottom", fontsize=8)
    fig.text(0.02, 0.95, "Qwen3-ForcedAligner 0.6B · official vs GGUF vs TensorRT + EXL3",
             fontsize=15, fontweight="bold", color="#1d2733")
    fig.text(0.02, 0.89,
             "LibriSpeech test-clean · 2,620 utterances · boundary MAE measured against the official backend",
             fontsize=8.8, color="#5b6675")
    fig.text(0.02, 0.05,
             f"Token mismatch: GGUF {agreement['GGUF INT4 + Q4_K']['tokenMismatchUtterances']} · "
             f"TensorRT {agreement['TensorRT BF16 + EXL3 4 bpw']['tokenMismatchUtterances']} utterances · "
             f"exact boundaries: GGUF {agreement['GGUF INT4 + Q4_K']['exactBoundaryFraction'] * 100:.1f}%, "
             f"TensorRT {agreement['TensorRT BF16 + EXL3 4 bpw']['exactBoundaryFraction'] * 100:.1f}%",
             fontsize=8.0, color="#6b7685")
    for ext, options in (("png", {"dpi": 220}), ("svg", {})):
        path = DOCS / f"windows-librispeech-qwen3-aligner-comparison.{ext}"
        fig.savefig(path, bbox_inches="tight", **options)
        if ext == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8")
    plt.close(fig)

    figure, ax = plt.subplots(figsize=(8.6, 5.8))
    figure.subplots_adjust(left=0.12, right=0.97, top=0.76, bottom=0.24)
    x = np.arange(2)
    stack_names = ["GGUF INT4 + Q4_K", "TensorRT BF16 + EXL3 4 bpw"]
    bottoms = np.zeros(2)
    stage_colors = {"Preprocess": "#8b9aa8", "Encoder": "#3b6ea5", "Decoder": "#c44e52", "Other": "#bdc3c7"}
    for stage in ("Preprocess", "Encoder", "Decoder", "Other"):
        values = []
        for name in stack_names:
            summary = summaries[name]
            stages = summary["medianStagesMs"]
            total = summary["medianMs"]
            if stage == "Preprocess":
                value = stages["preprocessingMs"]
            elif stage == "Encoder":
                value = stages["encoderMs"]
            elif stage == "Decoder":
                value = stages["decoderMs"]
            else:
                value = max(0.0, total - stages["preprocessingMs"] - stages["encoderMs"] - stages["decoderMs"])
            values.append(value)
        ax.bar(x, values, bottom=bottoms, color=stage_colors[stage], width=0.55, label=stage)
        for position, (value, bottom) in enumerate(zip(values, bottoms)):
            if value > 2:
                ax.text(position, bottom + value / 2, f"{value:.1f}", ha="center", va="center",
                        fontsize=8, color="white" if stage in ("Encoder", "Decoder") else "#26313d")
        bottoms += values
    official_total = summaries["Official Transformers"]["medianMs"]
    max_total = max(summaries[name]["medianMs"] for name in stack_names)
    ax.axhline(official_total, color="#8c8c8c", linestyle="--", linewidth=1.2,
               label="Official total")
    ax.text(1.3, official_total, f"official {official_total:.1f} ms",
            va="bottom", fontsize=8, color="#5b6675")
    ax.set_xticks(x, ["GGUF\nINT4 + Q4_K", "TensorRT BF16\n+ EXL3 4 bpw"], fontsize=8)
    ax.set_ylabel("Median latency (ms)")
    ax.set_ylim(0, max_total * 1.18)
    ax.grid(axis="y", color="#e4e8ee", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    figure.text(0.02, 0.95, "Qwen3-ForcedAligner 0.6B · median latency by stage",
                fontsize=15, fontweight="bold", color="#1d2733")
    figure.text(0.02, 0.90, "LibriSpeech test-clean · 2,620 utterances · official backend has no stage profile",
                fontsize=8.8, color="#5b6675")
    for ext, options in (("png", {"dpi": 220}), ("svg", {})):
        path = DOCS / f"windows-librispeech-qwen3-aligner-stage-latency.{ext}"
        figure.savefig(path, bbox_inches="tight", **options)
        if ext == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8")
    plt.close(figure)
    print(json.dumps(agreement, indent=2))


if __name__ == "__main__":
    main()
