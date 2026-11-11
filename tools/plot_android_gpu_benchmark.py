"""Plot the Android GPU benchmark report: latency, CPU-vs-accelerator, Pareto."""
from pathlib import Path
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
_live = ROOT / "models/sensevoice-results-10-GPU.json"
REPORT = _live if _live.is_file() else ROOT / "data/android/sensevoice-results-10-GPU.json"

COLORS = {
    "CPU": "#3b6ea5",
    "GPU Vulkan": "#6a51a3",
    "GPU OpenCL": "#c47f2a",
    "NNAPI": "#2f8f6b",
    "GPU delegate": "#c44e52",
}

# (label, cpu id or None, [accelerator ids], note)
PAIRS = [
    ("ncnn INT8", "ncnn", ["ncnn-vk-int8"], ""),
    ("ncnn FP16", "ncnn-fp16", ["ncnn-vk-fp16"], ""),
    ("MNN Q8", "mnn", ["mnn-cl-q8"], ""),
    ("MNN FP32", "mnn-cpu-fp32", ["mnn-cl-fp16", "mnn-cl-fp32", "mnn-vk-fp16"], ""),
    ("ORT Q8", "ort", ["ort-nnapi-q8"], ""),
    ("ORT FP32", None, ["ort-nnapi", "ort-nnapi-fp16"], "no CPU FP32 control"),
    ("sherpa FP32", None, ["sherpa-nnapi"], "no CPU FP32 control"),
    ("LiteRT Q8", "litert", ["litert-gpu-q8"], ""),
    ("LiteRT FP32", "litert-cpu-fp32", ["litert-gpu-fp16", "litert-gpu-fp32"], ""),
]


def load():
    data = json.loads(REPORT.read_bytes().decode("utf-8"))
    return data, {row["id"]: row for row in data["results"]}


def valid(row):
    return (
        row is not None
        and row.get("status") == "ok"
        and row.get("wer") is not None
        and row.get("meanMs") is not None
        and row["wer"] <= 0.15
    )


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


def latency_chart(data, rows):
    picked = [row for row in rows.values() if valid(row)]
    picked.sort(key=lambda row: row["meanMs"])
    labels = [f"{row['id']}  ·  {row['accelerator']}" for row in picked]
    values = [row["meanMs"] for row in picked]
    colors = [COLORS.get(row["accelerator"], "#7a8798") for row in picked]
    fig, ax = plt.subplots(figsize=(10.5, 8.2))
    fig.subplots_adjust(left=0.34, right=0.82, top=0.86, bottom=0.12)
    y = np.arange(len(picked))
    ax.barh(y, values, color=colors, height=0.68, zorder=2)
    ax.set_yticks(y, labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, max(values) * 1.22)
    ax.set_xlabel("Mean latency, PCM to text (ms) — lower is better")
    ax.grid(axis="x", color="#e4e8ee", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for position, row in zip(y, picked):
        ax.text(row["meanMs"] + max(values) * 0.01, float(position),
                f"{row['meanMs']:,.0f} ms   WER {row['wer']:.2%}",
                va="center", ha="left", fontsize=8.4, color="#26313d")
    ax.spines[["top", "right"]].set_visible(False)
    present = [key for key in COLORS if any(row["accelerator"] == key for row in picked)]
    ax.legend([Patch(color=COLORS[key]) for key in present], present,
              frameon=False, loc="lower right", bbox_to_anchor=(1.0, -0.14), ncol=3, fontsize=9)
    fig.text(0.02, 0.955,
             "SenseVoice Android: CPU vs GPU, valid outputs only",
             fontsize=16, fontweight="bold", color="#1d2733")
    config = data["config"]
    fig.text(0.02, 0.917,
             f"{data['device']['model']} · {data['device']['soc']} · Android {data['device']['android']} · "
             f"{config['threads']} threads · {config['warmup']} warmups / {config['repetitions']} runs · "
             f"WER ≤ 15%",
             fontsize=9.2, color="#5b6675")
    fig.text(0.02, 0.893,
             "Empty-output and errored rows are excluded here; see the CPU-vs-accelerator chart.",
             fontsize=8.8, color="#8792a1", style="italic")
    fig.savefig(DOCS / "android-gpu-latency.png", dpi=220, bbox_inches="tight")
    fig.savefig(DOCS / "android-gpu-latency.svg", bbox_inches="tight")
    plt.close(fig)


def ratio_chart(rows):
    entries = []
    for label, cpu_id, accel_ids, note in PAIRS:
        cpu = rows.get(cpu_id) if cpu_id else None
        for accel_id in accel_ids:
            accel = rows.get(accel_id)
            if accel is None:
                continue
            ok = valid(accel) and cpu is not None and cpu.get("meanMs")
            ratio = accel["meanMs"] / cpu["meanMs"] if ok else np.nan
            entries.append((label, accel_id, accel, cpu, ratio, ok, note))
    # Sort: known ratios ascending, failures at the end.
    entries.sort(key=lambda item: (np.isnan(item[4]), item[4] if not np.isnan(item[4]) else 0))
    labels = [item[1] for item in entries]
    fig, ax = plt.subplots(figsize=(10.5, 7.4))
    fig.subplots_adjust(left=0.36, right=0.86, top=0.84, bottom=0.14)
    y = np.arange(len(entries))
    for position, (_, accel_id, accel, cpu, ratio, ok, note) in zip(y, entries):
        if not ok:
            ax.barh(position, 0.12, left=0, color="#e8d3d4", height=0.66, zorder=2, hatch="///",
                    edgecolor="#c44e52", linewidth=0.8)
            if accel.get("status") == "empty_output":
                status = "empty output"
            elif accel.get("status") == "error":
                status = "error: " + str(accel.get("error", ""))[:52]
            elif cpu is None:
                status = "no same-model CPU control"
            elif accel.get("wer") is not None:
                status = f"invalid text (WER {accel['wer']:.0%})"
            else:
                status = "not comparable"
            ax.text(0.18, float(position), status, va="center", ha="left", fontsize=8.4, color="#a33a3f")
        else:
            color = "#c44e52" if ratio > 1.0 else "#2f8f6b"
            ax.barh(position, ratio, color=color, height=0.66, zorder=2)
            ax.text(ratio + 0.06, float(position), f"{ratio:.2f}×  ({accel['meanMs']:,.0f} vs {cpu['meanMs']:,.0f} ms)",
                    va="center", ha="left", fontsize=8.4, color="#26313d")
    ax.axvline(1.0, color="#7a8798", linewidth=1.0, linestyle=(0, (4, 3)), zorder=3)
    ax.text(1.0, -0.9, "parity", ha="center", va="bottom", fontsize=8.4, color="#5b6675")
    ax.set_yticks(y, labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 3.2)
    ax.set_xlabel("Accelerator latency ÷ CPU baseline (same model, lower is better)")
    ax.grid(axis="x", color="#e4e8ee", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend([Patch(color="#2f8f6b"), Patch(color="#c44e52"), Patch(facecolor="#e8d3d4", hatch="///",
                                                                     edgecolor="#c44e52")],
              ["faster than CPU", "slower than CPU", "invalid output / not comparable"],
              frameon=False, loc="lower right", bbox_to_anchor=(1.0, -0.16), ncol=3, fontsize=9)
    fig.text(0.02, 0.955, "SenseVoice Android: what each accelerator buys over CPU",
             fontsize=16, fontweight="bold", color="#1d2733")
    fig.text(0.02, 0.918,
             "Ratio < 1 means the GPU/NNAPI path is faster. NNAPI rows use FP32 models, "
             "their CPU reference is the engine's Q8 row where noted.",
             fontsize=8.9, color="#5b6675")
    fig.savefig(DOCS / "android-gpu-vs-cpu.png", dpi=220, bbox_inches="tight")
    fig.savefig(DOCS / "android-gpu-vs-cpu.svg", bbox_inches="tight")
    plt.close(fig)


def pareto_chart(rows):
    points = [row for row in rows.values() if valid(row)]
    fig, ax = plt.subplots(figsize=(9.6, 6.4))
    fig.subplots_adjust(left=0.10, right=0.97, top=0.82, bottom=0.13)
    for row in points:
        ax.scatter(row["meanMs"], row["wer"] * 100, s=54,
                   color=COLORS.get(row["accelerator"], "#7a8798"),
                   edgecolor="white", linewidth=0.8, zorder=3)
    highlights = {"sherpa": "fastest overall", "ncnn-vk-int8": "fastest GPU path (valid text)",
                  "mnn-q6": "lowest WER"}
    offsets = {"sherpa": (10, 10), "ncnn-vk-int8": (10, -14), "mnn-q6": (10, 8)}
    for row_id, note in highlights.items():
        row = rows.get(row_id)
        if row is None or not valid(row):
            continue
        dx, dy = offsets[row_id]
        ax.annotate(f"{row_id} · {row['meanMs']:,.0f} ms · {row['wer']:.2%}\n{note}",
                    (row["meanMs"], row["wer"] * 100), textcoords="offset points",
                    xytext=(dx, dy), fontsize=8.6, color="#26313d")
    ax.set_xscale("log")
    ax.set_xlabel("Mean latency (ms, log scale) — lower is better")
    ax.set_ylabel("WER (%) — lower is better")
    ax.set_ylim(2, 17)
    ax.grid(color="#e4e8ee", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    present = [key for key in COLORS if any(row["accelerator"] == key for row in points)]
    ax.legend([Patch(color=COLORS[key]) for key in present], present, frameon=False,
              loc="upper right", fontsize=9)
    fig.text(0.02, 0.945, "SenseVoice Android: accuracy vs latency",
             fontsize=16, fontweight="bold", color="#1d2733")
    fig.text(0.02, 0.905, "One 23.86 s LibriSpeech clip · WER is a sanity check, not a corpus score",
             fontsize=9.2, color="#5b6675")
    fig.savefig(DOCS / "android-gpu-pareto.png", dpi=220, bbox_inches="tight")
    fig.savefig(DOCS / "android-gpu-pareto.svg", bbox_inches="tight")
    plt.close(fig)


def main():
    setup()
    data, rows = load()
    latency_chart(data, rows)
    ratio_chart(rows)
    pareto_chart(rows)
    print("wrote docs/android-gpu-latency.png/.svg, android-gpu-vs-cpu.png/.svg, android-gpu-pareto.png/.svg")


if __name__ == "__main__":
    main()
