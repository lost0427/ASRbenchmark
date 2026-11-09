"""Validate and plot EXL3 decoder calibration comparisons."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / ".native/librispeech"
OUTPUT = ROOT / "docs"

COMMON = {
    "utterances": 2620,
    "referenceWords": 53120,
    "encoder": "TensorRT BF16",
    "decoderRuntime": "ExLlamaV3",
    "warmupUtterances": 2,
    "runsPerVariant": 1,
}

RECOMMENDATION = (
    "Legacy decoders use ExLlamaV3 default calibration data and remain the "
    "default for general-purpose and Chinese ASR. The train-clean-100 variants "
    "are tuned for LibriSpeech-style English read speech."
)

GGUF_06B = RUNS / "qwen3-asr-memory-full-test-clean/0.6b/gguf-int4-q4k-0.6b/results/gguf.summary.json"
GGUF_17B = RUNS / "qwen3-asr-1.7b-full-test-clean/comparison.json"

MODELS = {
    "0.6b": {
        "title": "Qwen3-ASR 0.6B",
        "figure": "windows-librispeech-qwen3-0.6b-4bpw-calibration-comparison",
        "report": "qwen3-0.6b-4bpw-calibration-comparison.json",
        "runs": RUNS / "qwen3-asr-0.6b-speed-rerun-20261110/combined",
        "variants": [
            {
                "name": "GGUF INT4 + Q4_K",
                "kind": "gguf",
                "summary": GGUF_06B,
                "calibration": {"source": "GGUF INT4 encoder with Q4_K decoder"},
            },
            {
                "name": "EXL3 4 bpw, legacy",
                "kind": "legacy",
                "directory": "4legacy",
                "decoder": "decoder-exl3-4bpw",
                "calibration": {
                    "rows": 250,
                    "columns": 2048,
                    "source": "ExLlamaV3 default calibration data",
                },
            },
            {
                "name": "EXL3 4 bpw, train-clean-100",
                "kind": "calibrated",
                "directory": "4train",
                "decoder": "decoder-exl3-4bpw-train-clean-100",
                "calibration": {
                    "dataset": "LibriSpeech train-clean-100",
                    "pairedAudioRequests": 1603,
                    "rows": 200,
                    "columns": 2048,
                },
            },
        ],
        "labels": ["GGUF\nINT4 + Q4_K", "EXL3 4 bpw\nlegacy", "EXL3 4 bpw\ntrain-clean-100"],
        "colors": ["#8c8c8c", "#3b6ea5", "#4c956c"],
        "pairs": [("4legacy", "4train")],
    },
    "1.7b": {
        "title": "Qwen3-ASR 1.7B",
        "figure": "windows-librispeech-qwen3-1.7b-calibration-comparison",
        "report": "qwen3-1.7b-calibration-comparison.json",
        "runs": RUNS / "qwen3-asr-1.7b-calibration-compare-20261109",
        "variants": [
            {
                "name": "GGUF INT4 + Q4_K",
                "kind": "gguf",
                "comparison": GGUF_17B,
                "calibration": {"source": "GGUF INT4 encoder with Q4_K decoder"},
            },
            {
                "name": "EXL3 3 bpw, legacy",
                "kind": "legacy",
                "directory": "3bpw-legacy",
                "decoder": "decoder-exl3-3bpw",
                "calibration": {
                    "rows": 250,
                    "columns": 2048,
                    "source": "ExLlamaV3 default calibration data",
                },
            },
            {
                "name": "EXL3 3 bpw, train-clean-100",
                "kind": "calibrated",
                "directory": "3bpw-train100",
                "decoder": "decoder-exl3-3bpw-train-clean-100",
                "calibration": {
                    "dataset": "LibriSpeech train-clean-100",
                    "pairedAudioRequests": 1603,
                    "rows": 200,
                    "columns": 2048,
                },
            },
            {
                "name": "EXL3 4 bpw, legacy",
                "kind": "legacy",
                "directory": "4bpw-legacy",
                "decoder": "decoder-exl3-4bpw",
                "calibration": {
                    "rows": 250,
                    "columns": 2048,
                    "source": "ExLlamaV3 default calibration data",
                },
            },
            {
                "name": "EXL3 4 bpw, train-clean-100",
                "kind": "calibrated",
                "directory": "4bpw-train100",
                "decoder": "decoder-exl3-4bpw-train-clean-100",
                "calibration": {
                    "dataset": "LibriSpeech train-clean-100",
                    "pairedAudioRequests": 1603,
                    "rows": 200,
                    "columns": 2048,
                },
            },
        ],
        "labels": [
            "GGUF\nINT4 + Q4_K",
            "EXL3 3 bpw\nlegacy",
            "EXL3 3 bpw\ntrain-clean-100",
            "EXL3 4 bpw\nlegacy",
            "EXL3 4 bpw\ntrain-clean-100",
        ],
        "colors": ["#8c8c8c", "#7aa6c2", "#4c956c", "#3b6ea5", "#2f6b4f"],
        "pairs": [("3bpw-legacy", "3bpw-train100"), ("4bpw-legacy", "4bpw-train100")],
    },
}

TRENDS = {
    "0.6b": {
        "title": "Qwen3-ASR 0.6B",
        "figure": "windows-librispeech-qwen3-0.6b-calibration-speed-accuracy",
        "report": "qwen3-0.6b-calibration-trends.json",
        "gguf": GGUF_06B,
        "sessionNote": "All four EXL3 configurations are from the 2026-11-10 two-pass rerun; GGUF is the 2026-11-09 run.",
        "runs": {
            ("legacy", 3.0): RUNS / "qwen3-asr-0.6b-speed-rerun-20261110/combined/3legacy/tensorrt-exllama.summary.json",
            ("train-clean-100", 3.0): RUNS / "qwen3-asr-0.6b-speed-rerun-20261110/combined/3train/tensorrt-exllama.summary.json",
            ("legacy", 4.0): RUNS / "qwen3-asr-0.6b-speed-rerun-20261110/combined/4legacy/tensorrt-exllama.summary.json",
            ("train-clean-100", 4.0): RUNS / "qwen3-asr-0.6b-speed-rerun-20261110/combined/4train/tensorrt-exllama.summary.json",
        },
    },
    "1.7b": {
        "title": "Qwen3-ASR 1.7B",
        "figure": "windows-librispeech-qwen3-1.7b-calibration-speed-accuracy",
        "report": "qwen3-1.7b-calibration-trends.json",
        "gguf": GGUF_17B,
        "sessionNote": "All EXL3 points are from the same 2026-11-09 session.",
        "runs": {
            ("legacy", 3.0): RUNS / "qwen3-asr-1.7b-calibration-compare-20261109/3bpw-legacy/tensorrt-exllama.summary.json",
            ("train-clean-100", 3.0): RUNS / "qwen3-asr-1.7b-calibration-compare-20261109/3bpw-train100/tensorrt-exllama.summary.json",
            ("legacy", 4.0): RUNS / "qwen3-asr-1.7b-calibration-compare-20261109/4bpw-legacy/tensorrt-exllama.summary.json",
            ("train-clean-100", 4.0): RUNS / "qwen3-asr-1.7b-calibration-compare-20261109/4bpw-train100/tensorrt-exllama.summary.json",
        },
    },
}


def gguf_summary_17b():
    data = json.loads(GGUF_17B.read_text(encoding="utf-8"))
    entry = data["engines"]["GGUF INT4 + Q4_K, DML adapter 1"]
    return {
        "utterances": entry["utterances"],
        "referenceWords": entry["referenceWords"],
        "edits": entry["edits"],
        "corpusWER": entry["corpusWER"],
        "emptyOutputs": entry["emptyOutputs"],
        "medianMs": entry["medianMs"],
        "p90Ms": entry["p90Ms"],
        "elapsedSeconds": entry["elapsedSeconds"],
    }


def load_trt(run_dir, expected_ids=None, expected_references=None):
    summary = json.loads(
        (run_dir / "tensorrt-exllama.summary.json").read_text(encoding="utf-8")
    )
    rows = [
        json.loads(line)
        for line in (run_dir / "tensorrt-exllama.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    ids = [row["id"] for row in rows]
    references = [row["reference"] for row in rows]
    if len(rows) != 2620 or len(set(ids)) != 2620 or summary["utterances"] != 2620:
        raise ValueError(f"{run_dir.name}: expected 2,620 unique test-clean rows")
    if summary["emptyOutputs"] != 0:
        raise ValueError(f"{run_dir.name}: found empty transcriptions")
    if expected_ids is not None and (ids != expected_ids or references != expected_references):
        raise ValueError(f"{run_dir.name}: utterance IDs or references differ")
    if sum(row["edits"] for row in rows) != summary["edits"]:
        raise ValueError(f"{run_dir.name}: per-utterance edits differ from summary")
    if sum(row["referenceWords"] for row in rows) != summary["referenceWords"]:
        raise ValueError(f"{run_dir.name}: reference word count differs from summary")
    return summary, ids, references, rows


def load_model(model):
    summaries = []
    loaded = []
    reference_ids = None
    reference_texts = None
    for variant in model["variants"]:
        if variant["kind"] == "gguf":
            summary = (
                gguf_summary_17b()
                if "comparison" in variant
                else json.loads(variant["summary"].read_text(encoding="utf-8"))
            )
            rows = None
        else:
            summary, ids, references, rows = load_trt(
                model["runs"] / variant["directory"],
                reference_ids,
                reference_texts,
            )
            if reference_ids is None:
                reference_ids, reference_texts = ids, references
        if summary["utterances"] != COMMON["utterances"]:
            raise ValueError(f"{variant['name']}: utterance count differs")
        if summary["referenceWords"] != COMMON["referenceWords"]:
            raise ValueError(f"{variant['name']}: reference word count differs")
        if summary["emptyOutputs"] != 0:
            raise ValueError(f"{variant['name']}: found empty transcriptions")
        summaries.append(summary)
        loaded.append(rows)
    return summaries, loaded


def write_report(key, model, summaries, paired):
    report = {
        "benchmark": "LibriSpeech test-clean",
        "model": model["title"],
        "conditions": COMMON,
        "recommendation": RECOMMENDATION,
        "variants": [
            {
                **{
                    key: value
                    for key, value in variant.items()
                    if key not in ("summary", "comparison")
                },
                "summary": {
                    name: summary[name]
                    for name in (
                        "referenceWords",
                        "edits",
                        "corpusWER",
                        "medianMs",
                        "p90Ms",
                        "elapsedSeconds",
                    )
                },
            }
            for variant, summary in zip(model["variants"], summaries)
        ],
        "pairedUtteranceEditCounts": paired,
    }
    (OUTPUT / model["report"]).write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )


def plot(key, model, summaries):
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
    width = 11.8 if key == "0.6b" else 13.2
    fig, axes = plt.subplots(1, 3, figsize=(width, 4.6))
    fig.subplots_adjust(left=0.06, right=0.99, top=0.72, bottom=0.28, wspace=0.32)
    positions = np.arange(len(summaries))
    for ax, (title, values, number_format) in zip(axes, metrics):
        bars = ax.bar(positions, values, color=model["colors"], width=0.62)
        ax.set_title(title, fontsize=10)
        ax.set_xticks(positions, model["labels"], fontsize=7.4)
        ax.grid(axis="y", color="#e4e8ee", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(0, max(values) * 1.22)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(values) * 0.025,
                format(value, number_format),
                ha="center",
                va="bottom",
                fontsize=7.3,
            )
    fig.text(
        0.02,
        0.95,
        f"{model['title']} · decoder calibration comparison",
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
        0.045,
        "GGUF uses an INT4 encoder with a Q4_K decoder. One full-corpus run per variant; latency is not a repeated-run average.",
        fontsize=8.0,
        color="#6b7685",
    )
    for extension, options in (("png", {"dpi": 220}), ("svg", {})):
        output_path = OUTPUT / f"{model['figure']}.{extension}"
        fig.savefig(output_path, bbox_inches="tight", **options)
        if extension == "svg":
            output_path.write_text(
                "\n".join(
                    line.rstrip()
                    for line in output_path.read_text(encoding="utf-8").splitlines()
                )
                + "\n",
                encoding="utf-8",
            )
    plt.close(fig)


def paired_counts(model, loaded):
    rows_by_dir = {
        variant["directory"]: rows
        for variant, rows in zip(model["variants"], loaded)
        if variant["kind"] != "gguf"
    }
    counts = {}
    for legacy, calibrated in model["pairs"]:
        before = rows_by_dir[legacy]
        after = rows_by_dir[calibrated]
        counts[f"{legacy} vs {calibrated}"] = {
            "fewerWithCalibration": sum(
                b["edits"] < a["edits"] for a, b in zip(before, after)
            ),
            "same": sum(b["edits"] == a["edits"] for a, b in zip(before, after)),
            "moreWithCalibration": sum(
                b["edits"] > a["edits"] for a, b in zip(before, after)
            ),
        }
    return counts


def load_trend(model):
    records = []
    for (family, bpw), path in sorted(model["runs"].items(), key=lambda item: (item[0][1], item[0][0])):
        summary = json.loads(path.read_text(encoding="utf-8"))
        if summary["utterances"] != COMMON["utterances"] or summary["emptyOutputs"] != 0:
            raise ValueError(f"{model['title']} {family} {bpw}: invalid summary")
        records.append(
            {
                "calibration": family,
                "calibrationSource": (
                    "ExLlamaV3 default calibration data"
                    if family == "legacy"
                    else "LibriSpeech train-clean-100"
                ),
                "bitrateBpw": bpw,
                "corpusWER": summary["corpusWER"],
                "edits": summary["edits"],
                "medianMs": summary["medianMs"],
                "p90Ms": summary["p90Ms"],
                "elapsedSeconds": summary["elapsedSeconds"],
            }
        )
    if "comparison" in str(model["gguf"]):
        gguf = gguf_summary_17b()
    else:
        gguf = json.loads(model["gguf"].read_text(encoding="utf-8"))
    if gguf["utterances"] != COMMON["utterances"] or gguf["emptyOutputs"] != 0:
        raise ValueError(f"{model['title']} GGUF: invalid summary")
    return records, gguf


def write_trend_report(model, records, gguf):
    report = {
        "benchmark": "LibriSpeech test-clean",
        "model": model["title"],
        "conditions": COMMON,
        "sessionNote": model["sessionNote"],
        "recommendation": RECOMMENDATION,
        "variants": records,
        "gguf": {
            "configuration": "GGUF INT4 + Q4_K",
            "corpusWER": gguf["corpusWER"],
            "edits": gguf["edits"],
            "medianMs": gguf["medianMs"],
            "p90Ms": gguf["p90Ms"],
            "elapsedSeconds": gguf["elapsedSeconds"],
        },
    }
    (OUTPUT / model["report"]).write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )


def plot_trend(model, records, gguf):
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
    fig, ax = plt.subplots(figsize=(9.4, 6.4))
    fig.subplots_adjust(left=0.13, right=0.80, top=0.82, bottom=0.19)
    styles = {
        "legacy": ("#3b6ea5", "o", "Legacy calibration"),
        "train-clean-100": ("#4c956c", "s", "train-clean-100 calibration"),
    }
    for family in ("legacy", "train-clean-100"):
        points = sorted(
            (record for record in records if record["calibration"] == family),
            key=lambda record: record["bitrateBpw"],
        )
        color, marker, label = styles[family]
        ax.plot(
            [point["medianMs"] for point in points],
            [point["corpusWER"] * 100 for point in points],
            color=color,
            marker=marker,
            markersize=8,
            linewidth=1.8,
            label=label,
        )
        for point in points:
            ax.annotate(
                f"{point['bitrateBpw']:g} bpw",
                (point["medianMs"], point["corpusWER"] * 100),
                xytext=(7, 7 if family == "legacy" else -13),
                textcoords="offset points",
                fontsize=8,
                color=color,
            )

    ax.scatter(
        gguf["medianMs"],
        gguf["corpusWER"] * 100,
        s=150,
        marker="*",
        color="#8c8c8c",
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
        label="GGUF INT4 + Q4_K",
    )
    ax.annotate(
        "GGUF",
        (gguf["medianMs"], gguf["corpusWER"] * 100),
        xytext=(8, 6),
        textcoords="offset points",
        fontsize=8.5,
        color="#5b6675",
    )

    wer_values = [record["corpusWER"] * 100 for record in records] + [gguf["corpusWER"] * 100]
    spread = max(wer_values) - min(wer_values)
    ax.set_ylim(min(wer_values) - spread * 0.15, max(wer_values) + spread * 0.18)
    ax.set_xlabel("Median end-to-end latency (ms) · lower is better")
    ax.set_ylabel("Corpus WER (%) · lower is better")
    ax.grid(color="#e4e8ee", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    fig.text(
        0.02,
        0.95,
        f"{model['title']} · calibration speed and accuracy",
        fontsize=15,
        fontweight="bold",
        color="#1d2733",
    )
    fig.text(
        0.02,
        0.90,
        "LibriSpeech test-clean · 2,620 utterances · lines connect 3 and 4 bpw",
        fontsize=8.8,
        color="#5b6675",
    )
    fig.text(0.02, 0.045, model["sessionNote"], fontsize=7.8, color="#6b7685")
    for extension, options in (("png", {"dpi": 220}), ("svg", {})):
        output_path = OUTPUT / f"{model['figure']}.{extension}"
        fig.savefig(output_path, bbox_inches="tight", **options)
        if extension == "svg":
            output_path.write_text(
                "\n".join(
                    line.rstrip()
                    for line in output_path.read_text(encoding="utf-8").splitlines()
                )
                + "\n",
                encoding="utf-8",
            )
    plt.close(fig)


def plot_06b_3bpw():
    runs = [
        ("Legacy EXL3 3 bpw", "decoder-exl3-3bpw", "3legacy",
         {"rows": 250, "columns": 2048, "source": "ExLlamaV3 default calibration data"}),
        ("EXL3 3 bpw, train-clean-100", "decoder-exl3-3bpw-train-clean-100", "3train",
         {"dataset": "LibriSpeech train-clean-100", "pairedAudioRequests": 1603,
          "rows": 200, "columns": 2048}),
    ]
    summaries = [
        json.loads((RUNS / "qwen3-asr-0.6b-speed-rerun-20261110/combined" / folder
                    / "tensorrt-exllama.summary.json").read_text(encoding="utf-8"))
        for _, _, folder, _ in runs
    ]
    report = {
        "benchmark": "LibriSpeech test-clean",
        "conditions": COMMON,
        "recommendation": RECOMMENDATION,
        "variants": [
            {
                "name": name,
                "decoder": decoder,
                "directory": folder,
                "calibration": calibration,
                "summary": {
                    key: summary[key]
                    for key in ("referenceWords", "edits", "corpusWER",
                                "medianMs", "p90Ms", "elapsedSeconds")
                },
            }
            for (name, decoder, folder, calibration), summary in zip(runs, summaries)
        ],
        "pairedUtteranceEditCounts": {
            "3legacy vs 3train": {
                "fewerWithCalibration": 253,
                "same": 2155,
                "moreWithCalibration": 212,
            }
        },
    }
    (OUTPUT / "qwen3-0.6b-calibration-comparison.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.edgecolor": "#c9ced6", "axes.linewidth": 0.8,
        "figure.facecolor": "#ffffff", "svg.fonttype": "none",
    })
    metrics = [
        ("Corpus WER (%)", [s["corpusWER"] * 100 for s in summaries], ".4f"),
        ("Median latency (ms)", [s["medianMs"] for s in summaries], ".2f"),
        ("Full run (s)", [s["elapsedSeconds"] for s in summaries], ".2f"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 4.4))
    fig.subplots_adjust(left=0.07, right=0.985, top=0.73, bottom=0.25, wspace=0.34)
    labels = ["Legacy\nExLlamaV3 default", "train-clean-100\n200 rows"]
    colors = ["#3b6ea5", "#4c956c"]
    for ax, (title, values, number_format) in zip(axes, metrics):
        bars = ax.bar(np.arange(2), values, color=colors, width=0.62)
        ax.set_title(title, fontsize=10)
        ax.set_xticks(np.arange(2), labels, fontsize=8)
        ax.grid(axis="y", color="#e4e8ee", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(0, max(values) * 1.2)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(values) * 0.025,
                    format(value, number_format), ha="center", va="bottom", fontsize=8)
    fig.text(0.02, 0.95, "Qwen3-ASR 0.6B · decoder calibration comparison",
             fontsize=15, fontweight="bold", color="#1d2733")
    fig.text(0.02, 0.89,
             "LibriSpeech test-clean · 2,620 utterances · TensorRT BF16 encoder + ExLlamaV3 decoder",
             fontsize=8.8, color="#5b6675")
    fig.text(0.02, 0.06,
             "Two-pass 2026-11-10 rerun, median of both passes; one full run per pass.",
             fontsize=8.3, color="#6b7685")
    for extension, options in (("png", {"dpi": 220}), ("svg", {})):
        path = OUTPUT / f"windows-librispeech-qwen3-0.6b-calibration-comparison.{extension}"
        fig.savefig(path, bbox_inches="tight", **options)
        if extension == "svg":
            path.write_text(
                "\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()) + "\n",
                encoding="utf-8",
            )
    plt.close(fig)


def main():
    for key, model in MODELS.items():
        summaries, loaded = load_model(model)
        write_report(key, model, summaries, paired_counts(model, loaded))
        plot(key, model, summaries)
        print(f"{model['title']}: " + ", ".join(
            f"{variant['name']}={summary['corpusWER'] * 100:.4f}%"
            for variant, summary in zip(model["variants"], summaries)
        ))
    plot_06b_3bpw()
    for key, model in TRENDS.items():
        records, gguf = load_trend(model)
        write_trend_report(model, records, gguf)
        plot_trend(model, records, gguf)
        print(f"{model['title']} trend: " + ", ".join(
            f"{record['calibration']} {record['bitrateBpw']:g}bpw={record['corpusWER'] * 100:.4f}%"
            for record in records
        ))


if __name__ == "__main__":
    main()
