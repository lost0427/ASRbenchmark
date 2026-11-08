"""Validate and plot the 0.6B EXL3 calibration comparison."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / ".native/librispeech/qwen3-asr-0.6b-3bpw-calibration-compare-20261108"
OUTPUT = ROOT / "docs"
VARIANTS = [
    {
        "name": "Legacy EXL3 3 bpw",
        "decoder": "decoder-exl3-3bpw",
        "directory": "old-250rows",
        "calibration": {
            "rows": 250,
            "columns": 2048,
            "source": "not recorded in quantization metadata",
        },
    },
    {
        "name": "EXL3 3 bpw, train-clean-100",
        "decoder": "decoder-exl3-3bpw-train-clean-100",
        "directory": "train-clean-100-200rows",
        "calibration": {
            "dataset": "LibriSpeech train-clean-100",
            "pairedAudioRequests": 1603,
            "rows": 200,
            "columns": 2048,
        },
    },
]
COLORS = ["#3b6ea5", "#4c956c"]


def load_run(variant, expected_ids=None, expected_references=None):
    result_dir = RUNS / variant["directory"]
    summary = json.loads(
        (result_dir / "tensorrt-exllama.summary.json").read_text(encoding="utf-8")
    )
    rows = [
        json.loads(line)
        for line in (result_dir / "tensorrt-exllama.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    ids = [row["id"] for row in rows]
    references = [row["reference"] for row in rows]
    if len(rows) != 2620 or len(set(ids)) != 2620 or summary["utterances"] != 2620:
        raise ValueError(f"{variant['name']}: expected 2,620 unique test-clean rows")
    if summary["emptyOutputs"] != 0:
        raise ValueError(f"{variant['name']}: found empty transcriptions")
    if expected_ids is not None and (ids != expected_ids or references != expected_references):
        raise ValueError(f"{variant['name']}: utterance IDs or references differ")
    if sum(row["edits"] for row in rows) != summary["edits"]:
        raise ValueError(f"{variant['name']}: per-utterance edits differ from summary")
    if sum(row["referenceWords"] for row in rows) != summary["referenceWords"]:
        raise ValueError(f"{variant['name']}: reference word count differs from summary")
    return summary, ids, references, rows


def main():
    summaries = []
    loaded = []
    reference_ids = None
    reference_texts = None
    for variant in VARIANTS:
        summary, ids, references, rows = load_run(
            variant, reference_ids, reference_texts
        )
        if reference_ids is None:
            reference_ids, reference_texts = ids, references
        summaries.append(summary)
        loaded.append(rows)

    better = sum(b["edits"] < a["edits"] for a, b in zip(*loaded))
    equal = sum(b["edits"] == a["edits"] for a, b in zip(*loaded))
    worse = sum(b["edits"] > a["edits"] for a, b in zip(*loaded))
    report = {
        "benchmark": "LibriSpeech test-clean",
        "conditions": {
            "utterances": 2620,
            "referenceWords": 53120,
            "encoder": "TensorRT BF16",
            "decoderRuntime": "ExLlamaV3",
            "warmupUtterances": 2,
            "runsPerVariant": 1,
        },
        "variants": [
            {
                **variant,
                "summary": {
                    key: summary[key]
                    for key in (
                        "referenceWords",
                        "edits",
                        "corpusWER",
                        "medianMs",
                        "p90Ms",
                        "elapsedSeconds",
                    )
                },
            }
            for variant, summary in zip(VARIANTS, summaries)
        ],
        "pairedUtteranceEditCounts": {
            "fewerWithNewCalibration": better,
            "same": equal,
            "moreWithNewCalibration": worse,
        },
    }
    (OUTPUT / "qwen3-0.6b-calibration-comparison.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.edgecolor": "#c9ced6",
            "axes.linewidth": 0.8,
            "figure.facecolor": "#ffffff",
            "svg.fonttype": "none",
        }
    )
    metrics = [
        ("Corpus WER (%)", [s["corpusWER"] * 100 for s in summaries], ".3f"),
        ("Median latency (ms)", [s["medianMs"] for s in summaries], ".2f"),
        ("Full run (s)", [s["elapsedSeconds"] for s in summaries], ".2f"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 4.4))
    fig.subplots_adjust(left=0.07, right=0.985, top=0.73, bottom=0.25, wspace=0.34)
    labels = ["Legacy\n250 × 2,048", "train-clean-100\n200 × 2,048"]
    for ax, (title, values, number_format) in zip(axes, metrics):
        bars = ax.bar(np.arange(2), values, color=COLORS, width=0.62)
        ax.set_title(title, fontsize=10)
        ax.set_xticks(np.arange(2), labels, fontsize=8)
        ax.grid(axis="y", color="#e4e8ee", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(0, max(values) * 1.2)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(values) * 0.025,
                format(value, number_format),
                ha="center",
                va="bottom",
                fontsize=8,
            )
    fig.text(
        0.02,
        0.95,
        "Qwen3-ASR 0.6B · decoder calibration comparison",
        fontsize=15,
        fontweight="bold",
        color="#1d2733",
    )
    fig.text(
        0.02,
        0.89,
        "LibriSpeech test-clean · 2,620 utterances · TensorRT BF16 encoder + ExLlamaV3 decoder",
        fontsize=8.8,
        color="#5b6675",
    )
    fig.text(
        0.02,
        0.06,
        "One full-corpus run per variant; latency is not a repeated-run average.",
        fontsize=8.3,
        color="#6b7685",
    )
    for extension, options in (("png", {"dpi": 220}), ("svg", {})):
        output_path = OUTPUT / f"windows-librispeech-qwen3-0.6b-calibration-comparison.{extension}"
        fig.savefig(
            output_path,
            bbox_inches="tight",
            **options,
        )
        if extension == "svg":
            output_path.write_text(
                "\n".join(line.rstrip() for line in output_path.read_text(encoding="utf-8").splitlines()) + "\n",
                encoding="utf-8",
            )
    plt.close(fig)


if __name__ == "__main__":
    main()
