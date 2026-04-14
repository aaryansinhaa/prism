import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys

STATE_DIR = Path(".prism")
SERVER_PID_FILE = STATE_DIR / "server.pid"
SERVER_LOG_FILE = STATE_DIR / "server.log"


def _run_command(command: list[str]) -> int:
    process = subprocess.run(command, check=False)
    return process.returncode


def _handle_format(target: str) -> int:
    return _run_command(["black", target])


def _handle_lint(target: str) -> int:
    return _run_command(["ruff", "check", target])


def _is_process_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _read_pid_file() -> int | None:
    if not SERVER_PID_FILE.exists():
        return None
    try:
        return int(SERVER_PID_FILE.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def _handle_run(port: int, host: str, reload_enabled: bool) -> int:
    existing_pid = _read_pid_file()
    if existing_pid and _is_process_running(existing_pid):
        print(
            "PRISM server is already running "
            f"(pid={existing_pid}). Stop it first with `prism stop`."
        )
        return 1

    STATE_DIR.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        "-m",
        "uvicorn",
        "app.main:app",
        "--host",
        host,
        "--port",
        str(port),
    ]
    if reload_enabled:
        command.append("--reload")

    with SERVER_LOG_FILE.open("a", encoding="utf-8") as log_file:
        process = subprocess.Popen(  # noqa: S603
            command,
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=log_file,
            start_new_session=True,
            close_fds=True,
        )

    SERVER_PID_FILE.write_text(str(process.pid), encoding="utf-8")
    print(f"PRISM server started in background (pid={process.pid}) on {host}:{port}.")
    print(f"Logs: {SERVER_LOG_FILE}")
    return 0


def _handle_stop() -> int:
    pid = _read_pid_file()
    if pid is None:
        print("No PRISM server PID file found. Is the server running?")
        return 1

    if not _is_process_running(pid):
        print(f"Stale PID file found for pid={pid}; removing it.")
        SERVER_PID_FILE.unlink(missing_ok=True)
        return 0

    os.kill(pid, signal.SIGTERM)
    SERVER_PID_FILE.unlink(missing_ok=True)
    print(f"Stopped PRISM server (pid={pid}).")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="prism", description="PRISM developer CLI")
    subparsers = parser.add_subparsers(dest="command")

    format_parser = subparsers.add_parser("format", help="Format code with Black")
    format_parser.add_argument(
        "target",
        nargs="?",
        default=".",
        help="File or directory to format (default: current directory)",
    )

    lint_parser = subparsers.add_parser("lint", help="Run Ruff lint checks")
    lint_parser.add_argument(
        "target",
        nargs="?",
        default=".",
        help="File or directory to lint (default: current directory)",
    )

    run_parser = subparsers.add_parser("run", help="Run PRISM server in detached mode")
    run_parser.add_argument(
        "port",
        nargs="?",
        type=int,
        default=8000,
        help="Port to run the server on (default: 8000)",
    )
    run_parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host interface to bind (default: 127.0.0.1)",
    )
    run_parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )

    subparsers.add_parser("stop", help="Stop detached PRISM server")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "format":
        return _handle_format(args.target)
    if args.command == "lint":
        return _handle_lint(args.target)
    if args.command == "run":
        return _handle_run(args.port, args.host, args.reload)
    if args.command == "stop":
        return _handle_stop()

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
