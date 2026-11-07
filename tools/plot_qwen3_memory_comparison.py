"""Plot full-corpus latency, WER, stage timing, and GPU memory for Qwen3-ASR."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / ".native/librispeech/qwen3-asr-memory-full-test-clean"
DOCS = ROOT / "docs"
CONFIGS = [
    ("GGUF INT4 + Q4_K", "gguf-int4-q4k", "gguf", "gguf"),
    ("BF16 + EXL3 3 bpw", "trt-bf16-exl3-3bpw", "tensorrt-exllama", "tensorrt-exllama"),
    ("BF16 + EXL3 4 bpw", "trt-bf16-exl3-4bpw", "tensorrt-exllama", "tensorrt-exllama"),
    ("INT8 + EXL3 4 bpw", "trt-int8-exl3-4bpw", "tensorrt-exllama", "tensorrt-exllama"),
]
CALIBRATED_25BPW = (
    "BF16 + EXL3 2.5 bpw, train-clean-100",
    "trt-bf16-exl3-2.5bpw",
    "tensorrt-exllama",
    "tensorrt-exllama",
)
CALIBRATED_20BPW = (
    "BF16 + EXL3 2 bpw, train-clean-100",
    "trt-bf16-exl3-2bpw-train100",
    "tensorrt-exllama",
    "tensorrt-exllama",
)
INT4_ENCODER = (
    "INT4 encoder + EXL3 4 bpw",
    "trt-int4-exl3-4bpw",
    "tensorrt-exllama",
    "tensorrt-exllama",
)
COLORS = {
    "GGUF INT4 + Q4_K": "#3b6ea5",
    "BF16 + EXL3 3 bpw": "#c44e52",
    "BF16 + EXL3 4 bpw": "#4c956c",
    "INT4 encoder + EXL3 4 bpw": "#8e6eaa",
    "BF16 + EXL3 2.5 bpw, train-clean-100": "#d98b3a",
    "BF16 + EXL3 2 bpw, train-clean-100": "#c0a020",
    "INT8 + EXL3 4 bpw": "#55717d",
}


def load_model(model):
    records = []
    expected = None
    configs = (
        CONFIGS[:3] + [INT4_ENCODER, CALIBRATED_25BPW, CALIBRATED_20BPW, CONFIGS[3]]
        if model == "0.6b"
        else CONFIGS[:3] + [INT4_ENCODER, CONFIGS[3]]
    )
    for label, folder, stem, engine in configs:
        directory = RUNS / model / folder
        if model == "0.6b" and folder == "gguf-int4-q4k":
            directory = RUNS / model / "gguf-int4-q4k-0.6b"
        elif folder == "trt-int4-exl3-4bpw":
            directory = (
                ROOT / ".native/librispeech/qwen3-asr-0.6b-int4-encoder-memory-full-test-clean"
                if model == "0.6b"
                else directory
            )
        result_dir = (
            directory
            if folder in ("trt-int4-exl3-4bpw", "trt-bf16-exl3-2bpw-train100")
            else directory / "results"
        )
        summary = json.loads((result_dir / f"{stem}.summary.json").read_text(encoding="utf-8"))
        if summary["utterances"] != 2620 or summary["emptyOutputs"] != 0:
            raise ValueError(f"{model} {label}: unexpected utterance or empty-output count")
        rows = [json.loads(line) for line in (result_dir / f"{stem}.jsonl").read_text(encoding="utf-8").splitlines()]
        ids = [row["id"] for row in rows]
        if len(ids) != 2620 or len(set(ids)) != 2620:
            raise ValueError(f"{model} {label}: expected 2,620 unique rows, got {len(ids)}")
        if expected is not None and ids != expected[0]:
            raise ValueError(f"{model} {label}: utterance IDs differ")
        if expected is not None and [row["reference"] for row in rows] != expected[1]:
            raise ValueError(f"{model} {label}: references differ")
        edits = sum(row["edits"] for row in rows)
        words = sum(row["referenceWords"] for row in rows)
        if edits != summary["edits"] or words != summary["referenceWords"]:
            raise ValueError(f"{model} {label}: JSONL totals differ from summary")
        expected = (ids, [row["reference"] for row in rows])
        memory = json.loads((directory / "memory.json").read_text(encoding="utf-8"))
        if memory["exitCode"] != 0:
            raise ValueError(f"{model} {label}: memory run exited {memory['exitCode']}")
        command = " ".join(memory["command"]).lower()
        if model == "0.6b" and folder == "gguf-int4-q4k":
            if "--gguf-model" not in command or "model-0.6b" not in command:
                raise ValueError("0.6b GGUF run did not explicitly select model-0.6b")
        elif "int8" in folder and "encoder-window.int8.engine" not in command:
            raise ValueError(f"{model} {label}: memory command did not select INT8 encoder")
        elif folder == "trt-int4-exl3-4bpw" and "encoder-window.int4.engine" not in command:
            raise ValueError(f"{model} {label}: memory command did not select INT4 encoder")
        elif "2bpw" in folder and "decoder-exl3-2bpw-train100" not in command:
            raise ValueError(f"{model} {label}: memory command did not select calibrated 2 bpw decoder")
        elif "2.5bpw" in folder and "decoder-exl3-2.5bpw-train100" not in command:
            raise ValueError(f"{model} {label}: memory command did not select calibrated 2.5 bpw decoder")
        records.append({"label": label, "summary": summary, "memory": memory})
    return records


def plot(model, records):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.edgecolor": "#c9ced6", "axes.linewidth": 0.8,
                         "figure.facecolor": "#ffffff", "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    fig.subplots_adjust(left=0.075, right=0.985, top=0.80, bottom=0.25, wspace=0.43)
    labels = [r["label"] for r in records]
    colors = [COLORS[label] for label in labels]

    ax = axes[0]
    point_labels = [
        "GGUF" if "GGUF" in record["label"] else
        "INT8 4 bpw" if "INT8" in record["label"] else
        "INT4 encoder" if "INT4 encoder" in record["label"] else
        "BF16 2.5 bpw" if "2.5 bpw" in record["label"] else
        "BF16 2 bpw" if "2 bpw" in record["label"] else
        "BF16 3 bpw" if "3 bpw" in record["label"] else "BF16 4 bpw"
        for record in records
    ]
    offsets_by_model = {
        "0.6b": {
            "GGUF": (7, 7), "BF16 3 bpw": (8, 48), "BF16 4 bpw": (8, 25),
            "INT4 encoder": (10, 10), "BF16 2.5 bpw": (8, 82),
            "BF16 2 bpw": (8, 0), "INT8 4 bpw": (12, -2),
        },
        "1.7b": {
            "GGUF": (7, 7), "BF16 3 bpw": (10, 26), "BF16 4 bpw": (10, 14),
            "INT4 encoder": (10, 42), "INT8 4 bpw": (10, -8),
        },
    }
    offsets = [offsets_by_model[model][label] for label in point_labels]
    for index, (record, label, color) in enumerate(zip(records, point_labels, colors)):
        summary = record["summary"]
        ax.scatter(summary["medianMs"], summary["corpusWER"] * 100,
                   s=90, color=color, edgecolor="white", linewidth=0.8)
        ax.annotate(label, (summary["medianMs"], summary["corpusWER"] * 100),
                    xytext=offsets[index], textcoords="offset points",
                    ha="left", fontsize=7.4)
    ax.set_xlabel("Median end-to-end latency (ms)")
    ax.set_ylabel("Corpus WER (%)")
    ax.set_title("Speed and accuracy")
    ax.grid(color="#e4e8ee", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    stage_names = ["preprocessingMs", "encoderMs", "decoderPrefillMs", "decoderGenerationMs", "otherMs"]
    stage_labels = ["Preprocess", "Encoder", "Decoder prefill", "Decoder generation", "Other"]
    bottoms = np.zeros(len(records))
    stage_colors = ["#8b9aa8", "#3b6ea5", "#4c956c", "#c44e52", "#bdc3c7"]
    for stage, stage_label, stage_color in zip(stage_names, stage_labels, stage_colors):
        values = []
        for record in records:
            stages = record["summary"].get("medianStagesMs", {})
            if stage == "decoderGenerationMs" and stage not in stages:
                stages = {**stages, "decoderGenerationMs": stages.get("decodeMs", 0)}
            if stage == "decoderPrefillMs" and stage not in stages:
                stages = {**stages, "decoderPrefillMs": stages.get("prefillMs", 0)}
            values.append(stages.get(stage, 0))
        ax.bar(np.arange(len(records)), values, bottom=bottoms, color=stage_color,
               width=0.68, label=stage_label)
        bottoms += values
    ax.set_xticks(
        np.arange(len(records)),
        [label.replace(" ", "\n", 1) for label in point_labels],
        fontsize=7,
    )
    ax.set_ylabel("Median stage latency (ms)")
    ax.set_title("Median latency by stage")
    ax.legend(frameon=False, fontsize=7, loc="upper center", ncol=3,
              bbox_to_anchor=(0.5, 1.0))
    ax.grid(axis="y", color="#e4e8ee", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[2]
    x = np.arange(len(records))
    delta = np.array([r["memory"]["peakDeltaOverIdleMiB"] for r in records])
    ax.bar(x, delta, color=colors, width=0.68)
    for pos, record in enumerate(records):
        memory = record["memory"]
        ax.text(pos, delta[pos] + 55, f"{delta[pos]:,} MiB",
                ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(
        x, [label.replace(" ", "\n", 1) for label in point_labels], fontsize=7
    )
    ax.set_ylabel("Incremental GPU memory (MiB)")
    ax.set_title("Peak memory use")
    ax.grid(axis="y", color="#e4e8ee", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(f"Qwen3-ASR {model} · LibriSpeech test-clean", x=0.02, y=0.97,
                 ha="left", fontsize=15, fontweight="bold", color="#1d2733")
    fig.text(0.02, 0.91, "2,620 utterances · same IDs and references · TensorRT encoder + ExLlamaV3 decoder · GPU-wide delta",
             fontsize=9, color="#5b6675")
    for ext, kwargs in (("png", {"dpi": 220}), ("svg", {})):
        fig.savefig(DOCS / f"windows-librispeech-qwen3-{model}-memory.{ext}",
                    bbox_inches="tight", **kwargs)
    plt.close(fig)


def main():
    result = {model: load_model(model) for model in ("0.6b", "1.7b")}
    output = {}
    for model, records in result.items():
        output[model] = [{
            "configuration": record["label"],
            "corpusWER": record["summary"]["corpusWER"],
            "medianMs": record["summary"]["medianMs"],
            "p90Ms": record["summary"]["p90Ms"],
            "medianStagesMs": record["summary"]["medianStagesMs"],
            "idleBaselineMedianMiB": record["memory"]["idleBaselineMedianMiB"],
            "peakUsedMiB": record["memory"]["peakUsedMiB"],
            "peakDeltaOverIdleMiB": record["memory"]["peakDeltaOverIdleMiB"],
            "gpu": record["memory"]["gpu"],
        } for record in records]
        plot(model, records)
    (DOCS / "qwen3-memory-comparison.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
