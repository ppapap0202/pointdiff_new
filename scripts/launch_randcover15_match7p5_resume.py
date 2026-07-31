from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(r"C:\pycharm\pointdiff_new")
CONFIG = ROOT / "config" / "train_randcover15_match7p5_from_0073.yaml"
LOG_DIR = ROOT / "logs"
STATUS_PATH = LOG_DIR / "randcover15_match7p5_launcher_status.txt"


def main() -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stdout_path = LOG_DIR / f"randcover15_match7p5_detached_{stamp}.stdout.log"
    stderr_path = LOG_DIR / f"randcover15_match7p5_detached_{stamp}.stderr.log"

    detached_flags = 0
    for name in ("DETACHED_PROCESS", "CREATE_NEW_PROCESS_GROUP"):
        detached_flags |= int(getattr(subprocess, name, 0))
    breakaway_flags = detached_flags | int(getattr(subprocess, "CREATE_BREAKAWAY_FROM_JOB", 0))

    stdout_file = stdout_path.open("wb")
    stderr_file = stderr_path.open("wb")
    try:
        try:
            proc = subprocess.Popen(
                [sys.executable, "main.py", "--config", str(CONFIG)],
                cwd=str(ROOT),
                stdin=subprocess.DEVNULL,
                stdout=stdout_file,
                stderr=stderr_file,
                close_fds=True,
                creationflags=breakaway_flags,
            )
            launch_mode = "breakaway"
        except PermissionError:
            proc = subprocess.Popen(
                [sys.executable, "main.py", "--config", str(CONFIG)],
                cwd=str(ROOT),
                stdin=subprocess.DEVNULL,
                stdout=stdout_file,
                stderr=stderr_file,
                close_fds=True,
                creationflags=detached_flags,
            )
            launch_mode = "detached"
    finally:
        stdout_file.close()
        stderr_file.close()

    STATUS_PATH.write_text(
        "\n".join(
            [
                "state=launched",
                f"started_at={datetime.now().isoformat()}",
                f"python_pid={proc.pid}",
                f"launch_mode={launch_mode}",
                f"stdout={stdout_path}",
                f"stderr={stderr_path}",
                f"config={CONFIG}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(proc.pid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
