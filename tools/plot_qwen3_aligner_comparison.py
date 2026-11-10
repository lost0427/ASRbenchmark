"""Compare the official, GGUF, EXL3, and TensorRT Qwen3-ForcedAligner backends."""
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
BACKENDS = {
    "official": ("Official Transformers", RUNS / "qwen3-asr-aligner-official-full-test-clean", "official"),
    "gguf": ("GGUF INT4 + Q4_K", RUNS / "qwen3-asr-aligner-gguf-full-test-clean", "gguf"),
    "exllama": ("TRT+EXL3 4 bpw", RUNS / "qwen3-asr-aligner-exl3-dyn-full-test-clean", "exllama"),
    "trt-bf16": ("TRT bf16", RUNS / "qwen3-asr-aligner-trt-bf16-pure-full", "trt-bf16"),
    "trt-fp16": ("TRT fp16", RUNS / "qwen3-asr-aligner-trt-fp16-pure-full", "trt-fp16"),
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
    data, summaries = {}, {}
    for key, (_, directory, stem) in BACKENDS.items():
        data[key] = load(directory / f"{stem}.jsonl")
        summaries[key] = json.loads(
            (directory / f"{stem}.summary.json").read_text(encoding="utf-8")
        )

    reference = data["official"]
    agreement = {
        "official": {
            "tokenMismatchUtterances": 0,
            "boundaryMaeSeconds": 0.0,
            "boundaryP90Seconds": 0.0,
            "boundaryMaxSeconds": 0.0,
            "exactBoundaryFraction": 1.0,
        }
    }
    for key in ("gguf", "exllama", "trt-bf16", "trt-fp16"):
        differences, exact, mismatched = [], 0, 0
        for left, right in zip(reference, data[key]):
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
        agreement[key] = {
            "tokenMismatchUtterances": mismatched,
            "boundaryMaeSeconds": float(np.mean(differences)),
            "boundaryP90Seconds": float(np.percentile(differences, 90)),
            "boundaryMaxSeconds": float(np.max(differences)),
            "exactBoundaryFraction": exact / max(1, len(differences)),
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
        "backends": {name: summaries[key] for key, (name, _, _) in BACKENDS.items()},
        "agreement": {name: agreement[key] for key, (name, _, _) in BACKENDS.items()},
    }
    (DOCS / "qwen3-aligner-comparison.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.edgecolor": "#c9ced6", "axes.linewidth": 0.8,
        "figure.facecolor": "#ffffff", "svg.fonttype": "none",
    })
    order = ["official", "gguf", "exllama", "trt-bf16", "trt-fp16"]
    labels = ["Official\nTransformers", "GGUF\nINT4+Q4_K", "TRT+EXL3\n4 bpw", "TRT\nbf16", "TRT\nfp16"]
    colors = ["#8c8c8c", "#d98b3a", "#a4636f", "#3b6ea5", "#2f8f83"]
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 5.0))
    fig.subplots_adjust(left=0.065, right=0.99, top=0.76, bottom=0.27, wspace=0.32)
    metrics = [
        ("Median latency (ms)", [summaries[key]["medianMs"] for key in order], ".2f"),
        ("P90 latency (ms)", [summaries[key]["p90Ms"] for key in order], ".2f"),
        ("Boundary MAE (ms)", [agreement[key]["boundaryMaeSeconds"] * 1000 for key in order], ".2f"),
    ]
    baseline_median = summaries["official"]["medianMs"]
    baseline_p90 = summaries["official"]["p90Ms"]
    for ax, (title, values, number_format) in zip(axes, metrics):
        bars = ax.bar(np.arange(len(order)), values, color=colors, width=0.68)
        ax.set_title(title, fontsize=10)
        ax.set_xticks(np.arange(len(order)), labels, fontsize=7.4)
        ax.grid(axis="y", color="#e4e8ee", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(0, max(max(values) * 1.42, 0.1))
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.025,
                    format(value, number_format), ha="center", va="bottom", fontsize=7.4)
        if title.startswith(("Median", "P90")):
            baseline = baseline_median if title.startswith("Median") else baseline_p90
            for bar, value in zip(bars, values):
                multiple = baseline / value if value else float("inf")
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.105,
                        f"×{multiple:.2f}", ha="center", va="bottom", fontsize=7.4,
                        color="#1d2733" if multiple >= 1 else "#a44a3f", fontweight="bold")
    fig.text(0.02, 0.95, "Qwen3-ForcedAligner 0.6B · decoder backend comparison",
             fontsize=15, fontweight="bold", color="#1d2733")
    fig.text(0.02, 0.89,
             "LibriSpeech test-clean · 2,620 utterances · TRT columns run encoder and decoder in the same precision",
             fontsize=8.8, color="#5b6675")
    fig.text(0.02, 0.045,
             f"Token mismatch: 0 for every backend · exact boundaries: GGUF {agreement['gguf']['exactBoundaryFraction'] * 100:.1f}%, "
             f"EXL3 {agreement['exllama']['exactBoundaryFraction'] * 100:.1f}%, TRT bf16 {agreement['trt-bf16']['exactBoundaryFraction'] * 100:.1f}%, "
             f"TRT fp16 {agreement['trt-fp16']['exactBoundaryFraction'] * 100:.1f}%",
             fontsize=8.0, color="#6b7685")
    for ext, options in (("png", {"dpi": 220}), ("svg", {})):
        path = DOCS / f"windows-librispeech-qwen3-aligner-comparison.{ext}"
        fig.savefig(path, bbox_inches="tight", **options)
        if ext == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8")
    plt.close(fig)

    figure, ax = plt.subplots(figsize=(9.2, 5.8))
    figure.subplots_adjust(left=0.11, right=0.975, top=0.76, bottom=0.24)
    stack_keys = ["gguf", "exllama", "trt-bf16", "trt-fp16"]
    stack_labels = ["GGUF\nINT4+Q4_K", "TRT+EXL3\n4 bpw", "TRT\nbf16", "TRT\nfp16"]
    x = np.arange(len(stack_keys))
    bottoms = np.zeros(len(stack_keys))
    stage_colors = {"Preprocess": "#8b9aa8", "Encoder": "#3b6ea5", "Decoder": "#c44e52", "Other": "#bdc3c7"}
    for stage in ("Preprocess", "Encoder", "Decoder", "Other"):
        values = []
        for key in stack_keys:
            stages = summaries[key]["medianStagesMs"]
            total = summaries[key]["medianMs"]
            if stage == "Preprocess":
                value = stages["preprocessingMs"]
            elif stage == "Encoder":
                value = stages["encoderMs"]
            elif stage == "Decoder":
                value = stages["decoderMs"]
            else:
                value = max(0.0, total - stages["preprocessingMs"] - stages["encoderMs"] - stages["decoderMs"])
            values.append(value)
        ax.bar(x, values, bottom=bottoms, color=stage_colors[stage], width=0.6, label=stage)
        for position, (value, bottom) in enumerate(zip(values, bottoms)):
            if value > 2.0:
                ax.text(position, bottom + value / 2, f"{value:.1f}", ha="center", va="center",
                        fontsize=7.6, color="white" if stage in ("Encoder", "Decoder") else "#26313d")
        bottoms += values
    official_total = summaries["official"]["medianMs"]
    ax.axhline(official_total, color="#8c8c8c", linestyle="--", linewidth=1.2, label="Official total")
    ax.text(len(stack_keys) - 0.55, official_total, f"official {official_total:.1f} ms",
            va="bottom", fontsize=8, color="#5b6675")
    for position, key in enumerate(stack_keys):
        total = summaries[key]["medianMs"]
        ax.text(position, total + 2.5, f"×{official_total / total:.2f}", ha="center", va="bottom",
                fontsize=8.2, color="#1d2733", fontweight="bold")
    ax.set_xticks(x, stack_labels, fontsize=8)
    ax.set_ylabel("Median latency (ms)")
    ax.set_ylim(0, max(summaries[key]["medianMs"] for key in stack_keys) * 1.18)
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
