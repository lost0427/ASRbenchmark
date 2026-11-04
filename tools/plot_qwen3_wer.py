"""Plot Qwen3 and SenseVoice WER on the same 1,000-utterance subset."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
ROWS = [
    ("Qwen3 transcribe.cpp Q8", "Qwen3", 1.78250, 1.53991, 2.04563),
    ("Qwen3-ASR-GGUF q4_k", "Qwen3", 2.02283, 1.75341, 2.32330),
    ("Qwen3 ONNX Runtime int4", "Qwen3", 2.03285, 1.77721, 2.32146),
    ("Qwen3 sherpa-onnx int8", "Qwen3", 3.08933, 2.48130, 3.84441),
    ("SenseVoice transcribe.cpp Q8", "SenseVoice", 3.28460, 2.95147, 3.64924),
]
COLORS = {"Qwen3": "#c44e52", "SenseVoice": "#4c956c"}


def main():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.edgecolor": "#c9ced6",
        "axes.linewidth": 0.8,
        "figure.facecolor": "#ffffff",
        "svg.fonttype": "none",
    })
    rows = sorted(ROWS, key=lambda row: row[2])
    labels = [row[0] for row in rows]
    values = np.array([row[2] for row in rows])
    lower = values - np.array([row[3] for row in rows])
    upper = np.array([row[4] for row in rows]) - values
    colors = [COLORS[row[1]] for row in rows]
    fig, ax = plt.subplots(figsize=(10.4, 4.8))
    fig.subplots_adjust(left=0.31, right=0.88, top=0.78, bottom=0.21)
    y = np.arange(len(rows))
    ax.barh(y, values, color=colors, height=0.62, xerr=[lower, upper],
            error_kw={"ecolor": "#52606d", "elinewidth": 1.2, "capsize": 3}, zorder=2)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 4.6)
    ax.set_xlabel("Corpus WER (%) · lower is better")
    ax.grid(axis="x", color="#e4e8ee", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    for position, value, high in zip(y, values, np.array([row[4] for row in rows])):
        ax.text(high + 0.07, float(position), f"{value:.2f}%", va="center",
                ha="left", fontsize=9.2, color="#26313d")
    handles = [Patch(color=COLORS[key]) for key in ("Qwen3", "SenseVoice")]
    ax.legend(handles, ["Qwen3", "SenseVoice"], frameon=False, loc="lower right",
              fontsize=9)
    fig.text(0.02, 0.94, "Qwen3-ASR vs SenseVoice · 1,000 utterances",
             fontsize=16, fontweight="bold", color="#1d2733")
    fig.text(0.02, 0.89,
             "Same manifest, same audio, 19,972 reference words · 95% bootstrap CI",
             fontsize=9.4, color="#5b6675")
    fig.text(0.02, 0.035,
             "Qwen3 transcribe.cpp is 1.50 percentage points better than "
             "SenseVoice transcribe.cpp on this sample.",
             fontsize=8.8, color="#6b7685")
    fig.savefig(DOCS / "windows-librispeech-qwen3-wer.png", dpi=220,
                bbox_inches="tight")
    fig.savefig(DOCS / "windows-librispeech-qwen3-wer.svg", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
