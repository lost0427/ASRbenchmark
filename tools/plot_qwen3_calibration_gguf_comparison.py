"""Plot calibration comparisons against the Qwen3-ASR-GGUF Q4_K reference."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / ".native/librispeech"
DOCS = ROOT / "docs"

STAGES = [
    ("preprocessingMs", "Preprocess", "#8b9aa8"),
    ("encoderMs", "Encoder", "#3b6ea5"),
    ("decoderPrefillMs", "Decoder prefill", "#4c956c"),
    ("decoderGenerationMs", "Decoder generation", "#c44e52"),
    ("otherMs", "Other", "#bdc3c7"),
]

MODELS = {
    "0.6b": {
        "flat": "qwen3-0.6b-calibration-vs-gguf",
        "stage": "qwen3-0.6b-stage-latency",
        "note": "All four EXL3 configurations are from the 2026-11-10 two-pass rerun; GGUF is the 2026-11-09 run.",
        "official": "qwen3-asr-0.6b-official-transformers-full-test-clean/official-transformers.summary.json",
        "runs": [
            ("3bpw legacy", "qwen3-asr-0.6b-speed-rerun-20261110/combined/3legacy/tensorrt-exllama.summary.json", "exl3"),
            ("3bpw train100", "qwen3-asr-0.6b-speed-rerun-20261110/combined/3train/tensorrt-exllama.summary.json", "exl3"),
            ("4bpw legacy", "qwen3-asr-0.6b-speed-rerun-20261110/combined/4legacy/tensorrt-exllama.summary.json", "exl3"),
            ("4bpw train100", "qwen3-asr-0.6b-speed-rerun-20261110/combined/4train/tensorrt-exllama.summary.json", "exl3"),
            ("GGUF Q4_K", "qwen3-asr-memory-full-test-clean/0.6b/gguf-int4-q4k-0.6b/results/gguf.summary.json", "gguf"),
        ],
    },
    "1.7b": {
        "flat": "qwen3-1.7b-calibration-vs-gguf",
        "stage": "qwen3-1.7b-stage-latency",
        "note": "All four EXL3 configurations are from the same 2026-11-09 session.",
        "official": "qwen3-asr-1.7b-official-transformers-full-test-clean/official-transformers.summary.json",
        "runs": [
            ("3bpw legacy", "qwen3-asr-1.7b-calibration-compare-20261109/3bpw-legacy/tensorrt-exllama.summary.json", "exl3"),
            ("3bpw train100", "qwen3-asr-1.7b-calibration-compare-20261109/3bpw-train100/tensorrt-exllama.summary.json", "exl3"),
            ("4bpw legacy", "qwen3-asr-1.7b-calibration-compare-20261109/4bpw-legacy/tensorrt-exllama.summary.json", "exl3"),
            ("4bpw train100", "qwen3-asr-1.7b-calibration-compare-20261109/4bpw-train100/tensorrt-exllama.summary.json", "exl3"),
            ("GGUF Q4_K", None, "gguf"),
        ],
    },
}


def load_summary(path):
    data = json.loads((RUNS / path).read_text(encoding="utf-8"))
    if data["utterances"] != 2620 or data["emptyOutputs"] != 0:
        raise ValueError(f"{path}: invalid summary")
    return data


def gguf_summary(model):
    if model == "0.6b":
        return load_summary(
            "qwen3-asr-memory-full-test-clean/0.6b/gguf-int4-q4k-0.6b/results/gguf.summary.json"
        )
    data = json.loads(
        (RUNS / "qwen3-asr-1.7b-full-test-clean/comparison.json").read_text(encoding="utf-8")
    )
    entry = data["engines"]["GGUF INT4 + Q4_K, DML adapter 1"]
    return {
        "corpusWER": entry["corpusWER"],
        "medianMs": entry["medianMs"],
        "medianStagesMs": entry["medianStagesMs"],
        "utterances": entry["utterances"],
        "edits": entry["edits"],
        "emptyOutputs": entry["emptyOutputs"],
    }


def gguf_stage(data):
    stages = data["medianStagesMs"]
    known = stages["encoderMs"] + stages["prefillMs"] + stages["decodeMs"]
    return {
        "preprocessingMs": 0.0,
        "encoderMs": stages["encoderMs"],
        "decoderPrefillMs": stages["prefillMs"],
        "decoderGenerationMs": stages["decodeMs"],
        "otherMs": data["medianMs"] - known,
    }


def collect(model):
    records = []
    for label, path, kind in MODELS[model]["runs"]:
        if kind == "gguf":
            data = gguf_summary(model)
            stage = gguf_stage(data)
        else:
            data = load_summary(path)
            stage = data["medianStagesMs"]
        records.append(
            {
                "name": label,
                "corpusWER": data["corpusWER"],
                "medianMs": data["medianMs"],
                "edits": data["edits"],
                "stages": stage,
            }
        )
    official = load_summary(MODELS[model]["official"])
    records.append(
        {
            "name": "Official Transformers",
            "corpusWER": official["corpusWER"],
            "medianMs": official["medianMs"],
            "edits": official["edits"],
            "stages": None,
        }
    )
    gguf_latency = records[-2]["medianMs"]
    official_latency = records[-1]["medianMs"]
    for record in records[:-2]:
        record["speedupVsGguf"] = gguf_latency / record["medianMs"]
        record["speedupVsOfficial"] = official_latency / record["medianMs"]
    return records


def setup():
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


def save(fig, stem):
    for extension, options in (("png", {"dpi": 220}), ("svg", {})):
        path = DOCS / f"{stem}.{extension}"
        fig.savefig(path, bbox_inches="tight", **options)
        if extension == "svg":
            path.write_text(
                "\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines())
                + "\n",
                encoding="utf-8",
            )
    plt.close(fig)


def plot_flat(model, records):
    setup()
    colors = ["#3b6ea5", "#4c956c", "#7aa6c2", "#2f6b4f", "#8c8c8c", "#8e6eaa"]
    offsets = [(8, 8), (8, -20), (8, 8), (8, -20), (8, 10), (-10, 12)]
    aligns = ["left", "left", "left", "left", "left", "right"]
    fig, ax = plt.subplots(figsize=(10.4, 6.4))
    fig.subplots_adjust(left=0.11, right=0.97, top=0.82, bottom=0.19)
    gguf_latency = records[-2]["medianMs"]
    official_latency = records[-1]["medianMs"]
    callouts = None
    if model == "0.6b":
        callouts = {
            "3bpw legacy": (155, 2.95),
            "3bpw train100": (155, 2.64),
            "4bpw legacy": (155, 2.43),
            "4bpw train100": (155, 2.22),
        }
    else:
        callouts = {
            "3bpw train100": (240, 2.14),
            "3bpw legacy": (240, 2.05),
            "4bpw legacy": (240, 1.96),
            "4bpw train100": (240, 1.87),
        }
    for record, color, offset, align in zip(records, colors, offsets, aligns):
        x = record["medianMs"]
        y = record["corpusWER"] * 100
        ax.scatter(x, y, s=55, color=color, edgecolor="white", linewidth=0.7, zorder=3)
        if record["name"].startswith("GGUF"):
            detail = (f"{y:.3f}% / {x:.1f} ms\n"
                      f"×{official_latency / x:.2f} vs official · reference")
        elif record["name"].startswith("Official"):
            detail = f"{y:.3f}% / {x:.1f} ms · official reference"
        else:
            detail = (f"{y:.3f}% / {x:.1f} ms\n"
                      f"×{gguf_latency / x:.2f} vs GGUF · ×{official_latency / x:.2f} vs official")
        text = f"{record['name']}\n{detail}"
        if callouts and record["name"] in callouts:
            label_x, label_y = callouts[record["name"]]
            ax.annotate(
                text,
                (x, y),
                xytext=(label_x, label_y),
                textcoords="data",
                fontsize=7.6,
                color=color,
                ha="left",
                va="center",
                arrowprops={"arrowstyle": "-", "color": color, "lw": 0.7,
                            "shrinkA": 0, "shrinkB": 3},
            )
        else:
            ax.annotate(
                text,
                (x, y),
                xytext=offset,
                textcoords="offset points",
                fontsize=7.6,
                color=color,
                ha=align,
                va="top" if offset[1] < 0 else "bottom",
                annotation_clip=False,
            )
    values = [record["corpusWER"] * 100 for record in records]
    spread = max(values) - min(values)
    ax.set_ylim(min(values) - spread * 0.35, max(values) + spread * 0.22)
    ax.set_xlabel("Median end-to-end latency (ms) · lower is better")
    ax.set_ylabel("Corpus WER (%) · lower is better")
    ax.grid(color="#e4e8ee", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.02, 0.95, f"Qwen3-ASR {model.upper()} · calibration vs GGUF and official backend",
             fontsize=15, fontweight="bold", color="#1d2733")
    fig.text(0.02, 0.90,
             "LibriSpeech test-clean · 2,620 utterances · train100 = train-clean-100 · ×N vs GGUF / official",
             fontsize=8.4, color="#5b6675")
    fig.text(0.02, 0.045, MODELS[model]["note"], fontsize=7.6, color="#6b7685")
    save(fig, f"windows-librispeech-{MODELS[model]['flat']}")


def plot_stage(model, records):
    setup()
    fig, ax = plt.subplots(figsize=(11.4, 6.4))
    fig.subplots_adjust(left=0.09, right=0.98, top=0.76, bottom=0.25)
    positions = np.arange(len(records))
    bottoms = np.zeros(len(records))
    for key, label, color in STAGES:
        values = np.array([record["stages"][key] for record in records])
        ax.bar(positions, values, bottom=bottoms, color=color, width=0.58,
               label=label, zorder=2)
        bottoms += values
    gguf_total = records[-1]["medianMs"]
    for index, record in enumerate(records):
        total = record["medianMs"]
        stage = record["stages"]
        if record["name"].startswith("GGUF"):
            multiple = "  reference"
        else:
            multiple = f"  ×{gguf_total / total:.2f} faster vs GGUF"
        ax.text(index, total + 5, f"{total:.1f} ms{multiple}", ha="center",
                va="bottom", fontsize=7.8, color="#26313d")
        if stage["decoderPrefillMs"] > 5:
            ax.text(index, stage["preprocessingMs"] + stage["encoderMs"] + stage["decoderPrefillMs"] / 2,
                    f"{stage['decoderPrefillMs']:.1f}", ha="center", va="center",
                    fontsize=6.8, color="#1f3b2d")
        if stage["decoderGenerationMs"] > 20:
            ax.text(index, total - stage["decoderGenerationMs"] / 2 - stage["otherMs"],
                    f"{stage['decoderGenerationMs']:.1f}", ha="center", va="center",
                    fontsize=7.4, color="white")
    ax.set_xticks(positions, [record["name"].replace(" ", "\n", 1) for record in records], fontsize=8)
    ax.set_ylabel("Median stage latency (ms)")
    ax.set_ylim(0, gguf_total * 1.22)
    ax.grid(axis="y", color="#e4e8ee", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=7.6, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.03))
    fig.text(0.02, 0.95, f"Qwen3-ASR {model.upper()} · median latency by stage",
             fontsize=15, fontweight="bold", color="#1d2733")
    fig.text(0.02, 0.90,
             "LibriSpeech test-clean · 2,620 utterances · labels show total ms and ×N faster than GGUF",
             fontsize=8.6, color="#5b6675")
    fig.text(0.02, 0.045, MODELS[model]["note"], fontsize=7.8, color="#6b7685")
    save(fig, f"windows-librispeech-{MODELS[model]['stage']}")


def main():
    report = {"benchmark": "LibriSpeech test-clean", "conditions": {"utterances": 2620}, "models": {}}
    for model in MODELS:
        records = collect(model)
        report["models"][model] = records
        plot_flat(model, records)
        plot_stage(model, [record for record in records if record["stages"] is not None])
        print(model, [f"{r['name']}={r['corpusWER'] * 100:.4f}%" for r in records])
    (DOCS / "qwen3-calibration-vs-gguf.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

