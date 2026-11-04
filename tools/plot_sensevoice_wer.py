"""Plot SenseVoice corpus WER on LibriSpeech test-clean."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
ROWS = [
    ("LiteRT Q8", "LiteRT", 3.29255, 3.08354, 3.50525),
    ("transcribe.cpp Q8", "transcribe.cpp", 3.29443, 3.07701, 3.50663),
    ("MNN Q8", "MNN", 3.31514, 3.11785, 3.54200),
    ("ONNX Runtime int8", "ORT", 3.34902, 3.14456, 3.56012),
    ("sherpa-onnx int8", "sherpa-onnx", 3.37538, 3.17373, 3.59189),
    ("ncnn INT8", "ncnn", 12.40776, 12.01948, 12.81837),
]
COLORS = {
    "LiteRT": "#4c956c",
    "transcribe.cpp": "#3b6ea5",
    "MNN": "#8a6d3b",
    "ORT": "#6a51a3",
    "sherpa-onnx": "#c44e52",
    "ncnn": "#7f7f7f",
}


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
    fig, ax = plt.subplots(figsize=(10, 4.9))
    fig.subplots_adjust(left=0.27, right=0.91, top=0.80, bottom=0.18)
    y = np.arange(len(rows))
    ax.barh(y, values, color=colors, height=0.62, xerr=[lower, upper],
            error_kw={"ecolor": "#52606d", "elinewidth": 1.2, "capsize": 3}, zorder=2)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 14.5)
    ax.set_xlabel("Corpus WER (%) · lower is better")
    ax.grid(axis="x", color="#e4e8ee", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    for position, value, high in zip(y, values, np.array([row[4] for row in rows])):
        ax.text(high + 0.25, float(position), f"{value:.2f}%", va="center",
                ha="left", fontsize=9.2, color="#26313d")
    handles = [Patch(color=COLORS[row[1]]) for row in rows]
    ax.legend(handles, labels, frameon=False, loc="lower right", fontsize=8.2,
              ncol=2, bbox_to_anchor=(1.0, 0.02))
    fig.text(0.02, 0.945, "SenseVoice · LibriSpeech test-clean",
             fontsize=16, fontweight="bold", color="#1d2733")
    fig.text(0.02, 0.895,
             "2,620 utterances · 5.40 h · 53,120 reference words · 95% bootstrap CI",
             fontsize=9.4, color="#5b6675")
    fig.text(0.02, 0.035,
             "Same 16 kHz audio, English, ITN off, punctuation/case-normalized WER. "
             "The five Q8/int8 results are statistically indistinguishable.",
             fontsize=8.4, color="#6b7685")
    fig.savefig(DOCS / "windows-librispeech-sensevoice-wer.png", dpi=220,
                bbox_inches="tight")
    fig.savefig(DOCS / "windows-librispeech-sensevoice-wer.svg", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
