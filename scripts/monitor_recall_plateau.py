import argparse
import ctypes
import os
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path


EPOCH_RE = re.compile(
    r"\[Epoch\s+(\d+)\].*?val_conf_no_nms_recall@6=([0-9.]+)"
)


def append_status(path: Path, message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"{timestamp} {message}"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    try:
        print(line, flush=True)
    except OSError:
        pass


def process_exists(pid: int) -> bool:
    if os.name != "nt":
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    STILL_ACTIVE = 259

    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
    if not handle:
        err = ctypes.get_last_error()
        return err == 5  # Access denied still means the PID probably exists.
    try:
        exit_code = ctypes.c_ulong()
        ok = kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
        return bool(ok) and int(exit_code.value) == STILL_ACTIVE
    finally:
        kernel32.CloseHandle(handle)


def stop_process_tree(pid: int) -> None:
    subprocess.run(
        ["taskkill", "/PID", str(pid), "/T", "/F"],
        capture_output=True,
        text=True,
        check=False,
    )


def wait_for_stable_checkpoint(
    out_dir: Path,
    epoch: int,
    status_path: Path,
    timeout_s: int,
    stable_s: int,
) -> bool:
    ckpt = out_dir / f"last_epoch{epoch:04d}.pth"
    deadline = time.time() + timeout_s
    last_size = -1
    stable_since = None
    append_status(status_path, f"waiting for checkpoint {ckpt}")
    while time.time() < deadline:
        if ckpt.exists():
            size = ckpt.stat().st_size
            if size > 0 and size == last_size:
                if stable_since is None:
                    stable_since = time.time()
                elif time.time() - stable_since >= stable_s:
                    append_status(status_path, f"checkpoint stable: {ckpt} size={size}")
                    return True
            else:
                last_size = size
                stable_since = None
        time.sleep(2)
    append_status(status_path, f"checkpoint wait timed out: {ckpt}")
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--status", required=True)
    parser.add_argument("--baseline-recall", type=float, required=True)
    parser.add_argument("--baseline-epoch", type=int, required=True)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--min-delta", type=float, default=1e-6)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--checkpoint-timeout-seconds", type=int, default=3600)
    parser.add_argument("--checkpoint-stable-seconds", type=int, default=8)
    args = parser.parse_args()

    log_path = Path(args.log)
    out_dir = Path(args.out_dir)
    status_path = Path(args.status)

    best_recall = float(args.baseline_recall)
    best_epoch = int(args.baseline_epoch)
    bad_count = 0
    seen_epochs = set()

    append_status(
        status_path,
        (
            f"monitor started pid={args.pid} baseline="
            f"{best_recall:.4f}@{best_epoch} patience={args.patience}"
        ),
    )

    while True:
        if not process_exists(args.pid):
            append_status(status_path, f"training process exited pid={args.pid}")
            return 0

        if log_path.exists():
            text = log_path.read_text(encoding="utf-8", errors="ignore")
            for match in EPOCH_RE.finditer(text):
                epoch = int(match.group(1))
                recall = float(match.group(2))
                if epoch in seen_epochs:
                    continue
                seen_epochs.add(epoch)

                if recall > best_recall + args.min_delta:
                    best_recall = recall
                    best_epoch = epoch
                    bad_count = 0
                    append_status(
                        status_path,
                        f"epoch={epoch:04d} recall={recall:.4f} improved best",
                    )
                else:
                    bad_count += 1
                    append_status(
                        status_path,
                        (
                            f"epoch={epoch:04d} recall={recall:.4f} "
                            f"no improvement ({bad_count}/{args.patience}); "
                            f"best={best_recall:.4f}@{best_epoch:04d}"
                        ),
                    )

                if bad_count >= args.patience:
                    wait_for_stable_checkpoint(
                        out_dir,
                        epoch,
                        status_path,
                        args.checkpoint_timeout_seconds,
                        args.checkpoint_stable_seconds,
                    )
                    append_status(
                        status_path,
                        (
                            f"stopping pid={args.pid}; recall plateau after "
                            f"{bad_count} validation checks"
                        ),
                    )
                    stop_process_tree(args.pid)
                    return 0

        time.sleep(args.poll_seconds)


def status_arg_from_argv(argv) -> Path | None:
    for idx, arg in enumerate(argv):
        if arg == "--status" and idx + 1 < len(argv):
            return Path(argv[idx + 1])
    return None


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        status_path = status_arg_from_argv(sys.argv)
        detail = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        if status_path is not None:
            append_status(status_path, f"fatal watcher error: {detail}")
            with status_path.open("a", encoding="utf-8") as f:
                f.write(traceback.format_exc() + "\n")
        else:
            raise
        raise
