"""Plot Qwen3-ASR and SenseVoice comparison charts."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
QWEN = [
    ("transcribe.cpp Q8_0", "CUDA", 1199, 5.80),
    ("transcribe.cpp BF16", "CUDA", 1760, 5.80),
    ("transcribe.cpp F16", "CUDA", 1846, 5.80),
    ("ONNX Runtime int4", "CUDA", 4282, 7.25),
    ("ONNX Runtime FP16", "CUDA", 7931, 5.80),
    ("transcribe.cpp Q8_0", "CPU", 9397, 5.80),
    ("ONNX Runtime int4", "CPU", 10061, 7.25),
    ("sherpa-onnx int8", "CPU", 10280, 7.25),
    ("transcribe.cpp F16", "CPU", 12335, 5.80),
    ("transcribe.cpp BF16", "CPU", 14183, 5.80),
]
COLORS = {"CPU": "#3b6ea5", "CUDA": "#c44e52"}


def setup():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.edgecolor": "#c9ced6",
        "axes.linewidth": 0.8,
        "axes.facecolor": "#ffffff",
        "figure.facecolor": "#ffffff",
        "svg.fonttype": "none",
    })


def first_chart():
    rows = sorted(QWEN, key=lambda row: row[2])
    labels = [f"{name}  ·  {provider}" for name, provider, _, _ in rows]
    values = [row[2] for row in rows]
    colors = [COLORS[row[1]] for row in rows]
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    fig.subplots_adjust(left=0.34, right=0.80, top=0.80, bottom=0.12)
    y = np.arange(len(rows))
    ax.barh(y, values, color=colors, height=0.66, zorder=2)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 17200)
    ax.set_xlabel("End-to-end latency (ms)")
    ax.grid(axis="x", color="#e4e8ee", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for position, (name, provider, value, wer) in zip(y, rows):
        ax.text(value + 180, float(position), f"{value:,} ms   WER {wer:.2f}%",
                va="center", ha="left", fontsize=8.6, color="#26313d")
    ax.spines[["top", "right"]].set_visible(False)
    handles = [Patch(color=COLORS[key]) for key in ("CPU", "CUDA")]
    ax.legend(handles, ["CPU", "CUDA"], frameon=False, loc="lower right", ncol=2,
              bbox_to_anchor=(1.0, -0.17), fontsize=9)
    fig.text(0.02, 0.945, "Qwen3-ASR-1.7B on Windows", fontsize=16, fontweight="bold",
             color="#1d2733")
    fig.text(0.02, 0.902, "23.86 s clip · RTX 4060 Laptop · 12 CPU threads · lower is better",
             fontsize=9.5, color="#5b6675")
    fig.savefig(DOCS / "windows-qwen3-engines.png", dpi=220, bbox_inches="tight")
    fig.savefig(DOCS / "windows-qwen3-engines.svg", bbox_inches="tight")
    plt.close(fig)


def second_chart():
    labels = ["SenseVoice\nQ8 · CPU", "Qwen3-ASR\nQ8 · CPU",
              "SenseVoice\nQ8 · CUDA", "Qwen3-ASR\nQ8 · CUDA"]
    values = [1255, 9397, 125, 1199]
    colors = ["#4c956c", "#d98b3a", "#4c956c", "#d98b3a"]
    x = np.array([0, 1, 3, 4])
    fig, ax = plt.subplots(figsize=(9.2, 5.5))
    fig.subplots_adjust(left=0.11, right=0.97, top=0.78, bottom=0.22)
    bars = ax.bar(x, values, color=colors, width=0.62, zorder=2)
    ax.set_yscale("log")
    ax.set_ylim(90, 17000)
    ax.set_xticks(x, labels)
    ax.set_ylabel("End-to-end latency (ms, log scale)")
    ax.grid(axis="y", color="#e4e8ee", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value * 1.14, f"{value:,} ms",
                ha="center", va="bottom", fontsize=9.5, color="#26313d")
    ax.text(0.5, 0.035, "CPU", transform=ax.transAxes, ha="center", color="#5b6675",
            fontsize=9, fontweight="bold")
    ax.text(0.88, 0.035, "CUDA", transform=ax.transAxes, ha="center", color="#5b6675",
            fontsize=9, fontweight="bold")
    fig.text(0.02, 0.935, "SenseVoice vs Qwen3-ASR · same clip, same engine",
             fontsize=15.5, fontweight="bold", color="#1d2733")
    fig.text(0.02, 0.885, "transcribe.cpp Q8_0 · WER 5.80% for both · Qwen3 is 7.5x slower on CPU and 9.6x on CUDA",
             fontsize=9.5, color="#5b6675")
    fig.text(0.02, 0.035, "SenseVoiceSmall ~234M parameters  ·  Qwen3-ASR 1.7B parameters  ·  one clean English utterance",
             fontsize=8.8, color="#6b7685")
    fig.savefig(DOCS / "windows-sensevoice-vs-qwen3.png", dpi=220, bbox_inches="tight")
    fig.savefig(DOCS / "windows-sensevoice-vs-qwen3.svg", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    setup()
    first_chart()
    second_chart()
