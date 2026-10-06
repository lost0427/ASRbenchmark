"""Use the local Motrix CLI to download files before model preparation."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


class MotrixDownloader:
    def __init__(self, executable="motrix", connections=8, proxy=None, timeout=0):
        cli = shutil.which(executable)
        if not cli:
            raise RuntimeError("Motrix CLI not found. Install it, or use --downloader requests.")
        cli = Path(cli).resolve()
        self.command = [str(cli)]
        if cli.suffix.lower() in (".cmd", ".bat", ".ps1", ".js"):
            # Invoke npm's JS entry directly; avoid passing URLs through cmd.exe.
            entry = cli if cli.suffix.lower() == ".js" else cli.parent / "node_modules/@motrix/cli/dist/bin/motrix.js"
            node = cli.parent / "node.exe"
            node = str(node) if node.is_file() else shutil.which("node")
            if not entry.is_file() or not node:
                raise RuntimeError("Cannot find Motrix's Node entry. Pass --motrix-cli with the path to motrix.js.")
            self.command = [node, str(entry)]
        self.connections = connections
        self.proxy = proxy
        self.timeout = timeout
        self.call("open", "--timeout", "30000")

    def call(self, *args):
        result = subprocess.run(self.command + ["--json", *args], capture_output=True,
                                text=True, encoding="utf-8", timeout=45)
        if result.returncode:
            raise RuntimeError(f"Motrix {args[0]} failed: {result.stderr.strip() or result.stdout.strip()}")
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"Motrix {args[0]} did not return JSON") from error

    def tasks(self):
        offset = 0
        while True:
            page = self.call("list", "--limit", "500", "--offset", str(offset))
            tasks = page["tasks"]
            yield from tasks
            offset += len(tasks)
            if not tasks or offset >= page["total"]:
                break

    def fetch(self, url, output):
        output = output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        staged = output.with_name(output.name + ".motrix-download")
        state = output.with_name(output.name + ".motrix-task.json")
        if output.is_file() and not state.exists():
            return
        if state.exists():
            saved = json.loads(state.read_text(encoding="utf-8"))
            if saved["url"] != url:
                raise RuntimeError(f"Download URL changed. Resolve the old Motrix task recorded in {state} first.")
            task_id = saved["taskId"]
        else:
            # Recover a task added just before an interrupted state-file write.
            matches = [task for task in self.tasks() if task["name"] == staged.name
                       and os.path.normcase(os.path.abspath(task["saveDir"])) == os.path.normcase(str(output.parent))]
            if len(matches) > 1:
                raise RuntimeError(f"Multiple Motrix tasks for {staged}. Keep one task and retry.")
            if matches:
                task_id = matches[0]["id"]
            else:
                args = ["add", url, "--save-dir", str(output.parent), "--filename", staged.name,
                        "--connections", str(self.connections)]
                if self.proxy:
                    args += ["--proxy", self.proxy]
                task = self.call(*args)
                task_id = task["id"]
            temporary = state.with_suffix(".tmp")
            temporary.write_text(json.dumps({"url": url, "taskId": task_id}) + "\n", encoding="utf-8")
            temporary.replace(state)
        print(f"Motrix task {task_id}: {output.name}", flush=True)
        start = time.monotonic()
        while True:
            task = next((task for task in self.tasks() if task["id"] == task_id), None)
            if task is None:
                raise RuntimeError(f"Motrix task {task_id} disappeared. Remove {state} to queue it again.")
            status = task["status"]
            if status == "completed":
                final = task.get("finalPath")
                if not final or Path(final).resolve() != staged or not staged.is_file():
                    raise RuntimeError(f"Motrix saved an unexpected path: {final}; expected {staged}")
                staged.replace(output)
                state.unlink()
                return
            if status in ("error", "paused"):
                raise RuntimeError(f"Motrix task {task_id}: {status}: {task.get('error') or ''}. "
                                   f"Resolve it in Motrix, resume it, then rerun this script.")
            if self.timeout and time.monotonic() - start >= self.timeout:
                raise TimeoutError(f"Waiting for {output.name} timed out. Motrix keeps downloading; rerun to continue.")
            total = task.get("bytesTotal")
            progress = f"{task['bytesDone'] / total:.1%}" if total else f"{task['bytesDone']} bytes"
            print(f"  {status}: {progress}, {task['speedBps'] / 1024 / 1024:.2f} MiB/s", flush=True)
            time.sleep(5)
