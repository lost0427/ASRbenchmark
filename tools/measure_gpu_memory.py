"""Run one benchmark command while sampling whole-GPU memory with nvidia-smi."""
import argparse
import json
import statistics
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--interval-ms", type=int, default=500)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or args.interval_ms < 100:
        parser.error("provide a command after -- and an interval of at least 100 ms")

    sampler = subprocess.Popen(
        ["nvidia-smi", "--query-gpu=name,timestamp,memory.used,memory.total",
         "--format=csv,noheader,nounits", f"--loop-ms={args.interval_ms}"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    samples = []
    first_sample = threading.Event()

    def collect():
        for line in sampler.stdout:
            fields = [field.strip() for field in line.split(",")]
            if len(fields) != 4:
                continue
            try:
                samples.append({
                    "gpu": fields[0],
                    "timestamp": fields[1],
                    "usedMiB": int(fields[2]),
                    "totalMiB": int(fields[3]),
                })
                first_sample.set()
            except ValueError:
                continue

    reader = threading.Thread(target=collect, daemon=True)
    reader.start()
    if not first_sample.wait(timeout=15):
        sampler.terminate()
        reader.join(timeout=2)
        raise RuntimeError("nvidia-smi did not provide GPU memory samples")
    while len(samples) < 3 and sampler.poll() is None:
        time.sleep(0.05)
    if sampler.poll() is not None:
        raise RuntimeError("nvidia-smi sampler exited before the benchmark started")

    baseline = statistics.median(sample["usedMiB"] for sample in samples[:5])
    run_start = len(samples)
    started = datetime.now().astimezone().isoformat()
    try:
        child = subprocess.Popen(command)
        exit_code = child.wait()
    finally:
        sampler.terminate()
        sampler.wait(timeout=5)
        reader.join(timeout=5)

    run_samples = samples[run_start:]
    if not run_samples:
        raise RuntimeError("No GPU memory samples were collected during the benchmark")
    peak = max(sample["usedMiB"] for sample in run_samples)
    result = {
        "command": command,
        "startedAt": started,
        "finishedAt": datetime.now().astimezone().isoformat(),
        "exitCode": exit_code,
        "samplingIntervalMs": args.interval_ms,
        "measurement": "whole-GPU memory used; includes Windows and other GPU clients",
        "gpu": run_samples[0]["gpu"],
        "totalMiB": run_samples[0]["totalMiB"],
        "idleBaselineMedianMiB": baseline,
        "peakUsedMiB": peak,
        "peakDeltaOverIdleMiB": peak - baseline,
        "sampleCount": len(run_samples),
        "samples": run_samples,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "samples"}, sort_keys=True))
    if exit_code:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
