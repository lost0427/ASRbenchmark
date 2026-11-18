"""Compare the Python and native C++ Qwen3-ASR runtimes on the same corpus."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from run_data import resolve_runs

ROOT = Path(__file__).resolve().parents[1]
RUNS = resolve_runs()
DOCS = ROOT / "docs"

PAIRS = {
    "0.6B": (
        RUNS / "qwen3-asr-0.6b-w4a16-20261118" / "w4a16.summary.json",
        RUNS / "qwen3-asr-0.6b-native-cpp-full-test-clean" / "native-cpp.summary.json",
    ),
    "1.7B": (
        RUNS / "qwen3-asr-1.7b-w4a16-20261118" / "w4a16.summary.json",
        RUNS / "qwen3-asr-1.7b-native-cpp-full-test-clean" / "native-cpp.summary.json",
    ),
}

SEGMENTS = [
    ("preprocessingMs", "mel + I/O", "#3b6ea5"),
    ("encoderMs", "audio encoder", "#d98b3a"),
    ("decoderPrefillMs", "prefill", "#c0a020"),
    ("decoderGenerationMs", "decode", "#c44e52"),
]


def main():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.edgecolor": "#c9ced6",
            "axes.linewidth": 0.8,
            "figure.facecolor": "#ffffff",
            "svg.fonttype": "none",
        }
    )

    summaries = {}
    for model, (python_path, native_path) in PAIRS.items():
        summaries[model] = {
            "Python (TensorRT W4A16)": json.loads(python_path.read_text(encoding="utf-8")),
            "C++ native": json.loads(native_path.read_text(encoding="utf-8")),
        }

    figure, axes = plt.subplots(1, 2, figsize=(9.4, 4.9), sharey=True)
    figure.subplots_adjust(left=0.09, right=0.98, top=0.82, bottom=0.2, wspace=0.14)

    for axis, (model, group) in zip(axes, summaries.items()):
        labels = list(group)
        positions = np.arange(len(labels))
        bottoms = np.zeros(len(labels))
        for key, name, color in SEGMENTS:
            values = np.array([group[label]["medianStagesMs"].get(key, 0.0) for label in labels])
            axis.bar(positions, values, 0.55, bottom=bottoms, color=color, edgecolor="white",
                     linewidth=0.8, label=name)
            bottoms += values
        for index, label in enumerate(labels):
            summary = group[label]
            axis.text(index, bottoms[index] + 2.0,
                      f"{summary['medianMs']:.1f} ms\np90 {summary['p90Ms']:.0f} ms\n"
                      f"WER {summary['corpusWER'] * 100:.2f}%",
                      ha="center", va="bottom", fontsize=8.5, color="#3c4350")
        axis.set_xticks(positions)
        axis.set_xticklabels(labels)
        axis.set_title(model, fontsize=11, color="#222831")
        axis.set_ylim(0, bottoms.max() * 1.35)
        axis.grid(axis="y", color="#e6e9ee", linewidth=0.7)
        axis.set_axisbelow(True)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)

    axes[0].set_ylabel("median per utterance (ms)")
    handles, names = axes[0].get_legend_handles_labels()
    figure.legend(handles, names, loc="upper center", ncol=4, frameon=False,
                  bbox_to_anchor=(0.5, 0.99), fontsize=9)
    figure.suptitle("Qwen3-ASR: Python vs native C++ runtime, same session (LibriSpeech test-clean, 2,620 utterances)",
                    fontsize=11.5, y=0.93)

    target = DOCS / "qwen3-asr-py-vs-cpp.png"
    figure.savefig(target, dpi=200)
    print(f"wrote {target}")

    for model, group in summaries.items():
        for label, summary in group.items():
            stages = summary["medianStagesMs"]
            print(
                f"{model:5s} {label:24s} median {summary['medianMs']:7.1f} ms  "
                f"p90 {summary['p90Ms']:7.1f}  WER {summary['corpusWER'] * 100:5.2f}%  "
                f"mel {stages.get('preprocessingMs', 0):4.1f} enc {stages.get('encoderMs', 0):4.1f} "
                f"prefill {stages.get('decoderPrefillMs', 0):5.1f} decode {stages.get('decoderGenerationMs', 0):5.1f}"
            )


if __name__ == "__main__":
    main()
