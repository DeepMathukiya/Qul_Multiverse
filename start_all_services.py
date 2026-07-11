"""Start and coordinate the whole QA system with one command.

    python start_all_services.py

Boot order (each step waits for the previous one to be healthy):
1. datascience service  -> http://127.0.0.1:8100  (all CV/OCR/ML processing)
2. backend service      -> http://0.0.0.0:5000    (phone uploads + orchestration)
3. frontend dashboard   -> http://127.0.0.1:8501  (Streamlit UI)

Press Ctrl+C to stop everything.
"""

from __future__ import annotations

import socket
import subprocess
import sys
import threading
import time
import urllib.request

SERVICES = [
    {
        "name": "datascience",
        "port": 8100,
        "command": [
            sys.executable, "-m", "uvicorn", "datascience.main:app",
            "--host", "127.0.0.1", "--port", "8100",
        ],
        "health_url": "http://127.0.0.1:8100/health",
    },
    {
        "name": "backend",
        "port": 5000,
        "command": [
            sys.executable, "-m", "uvicorn", "backend.main:app",
            "--host", "0.0.0.0", "--port", "5000",
        ],
        "health_url": "http://127.0.0.1:5000/health",
    },
    {
        "name": "frontend",
        "port": 8501,
        "command": [
            sys.executable, "-m", "streamlit", "run",
            "frontend/streamlit_dashboard.py",
            "--server.port", "8501", "--server.headless", "true",
        ],
        "health_url": "http://127.0.0.1:8501/healthz",
    },
]

HEALTH_TIMEOUT_SEC = 60


def _pump_output(name: str, process: subprocess.Popen) -> None:
    """Prefix every service log line with its name."""
    for line in iter(process.stdout.readline, ""):
        print(f"[{name}] {line.rstrip()}")


def _wait_healthy(name: str, url: str, timeout_sec: float) -> bool:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def _port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        return sock.connect_ex(("127.0.0.1", port)) == 0


def main() -> int:
    # Stream logs immediately even when stdout is piped (not a terminal).
    sys.stdout.reconfigure(line_buffering=True)

    busy = [s for s in SERVICES if _port_in_use(s["port"])]
    if busy:
        for service in busy:
            print(f"[coordinator] ERROR: port {service['port']} "
                  f"({service['name']}) is already in use.")
        print("[coordinator] Another instance of the system is probably "
              "running — stop it first (Ctrl+C in its window), then retry.")
        return 1

    processes: list[tuple[str, subprocess.Popen]] = []

    def stop_all() -> None:
        for name, process in reversed(processes):
            if process.poll() is None:
                print(f"[coordinator] stopping {name} ...")
                process.terminate()
        for _, process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()

    try:
        for service in SERVICES:
            name = service["name"]
            print(f"[coordinator] starting {name}: {' '.join(service['command'])}")

            process = subprocess.Popen(
                service["command"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            processes.append((name, process))

            threading.Thread(
                target=_pump_output, args=(name, process), daemon=True
            ).start()

            if not _wait_healthy(name, service["health_url"], HEALTH_TIMEOUT_SEC):
                print(f"[coordinator] ERROR: {name} did not become healthy "
                      f"within {HEALTH_TIMEOUT_SEC}s — aborting")
                stop_all()
                return 1

            print(f"[coordinator] {name} is healthy at {service['health_url']}")

        print()
        print("=" * 62)
        print("  All services running:")
        print("    phone upload : http://<this-pc-ip>:5000/upload")
        print("    backend API  : http://127.0.0.1:5000/docs")
        print("    datascience  : http://127.0.0.1:8100/docs")
        print("    dashboard    : http://127.0.0.1:8501")
        print("  Press Ctrl+C to stop everything.")
        print("=" * 62)
        print()

        # Keep running until Ctrl+C or a service dies.
        while True:
            time.sleep(1)
            for name, process in processes:
                if process.poll() is not None:
                    print(f"[coordinator] {name} exited with code "
                          f"{process.returncode} — shutting down the rest")
                    stop_all()
                    return process.returncode or 1

    except KeyboardInterrupt:
        print("\n[coordinator] Ctrl+C — shutting down")
        stop_all()
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
