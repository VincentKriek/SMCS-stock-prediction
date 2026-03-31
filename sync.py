import subprocess
import shutil
import os
from pathlib import Path

LOCKFILES = {
    "cuda": "uv-cuda.lock",
    "rocm": "uv-rocm.lock",
    "cpu": "uv-cpu.lock",
}


def run(cmd: list[str], env=None):
    print(f"👉 {' '.join(cmd)}")
    subprocess.run(cmd, check=True, env=env)


def detect_backend() -> str:
    if shutil.which("nvidia-smi"):
        return "cuda"
    if shutil.which("rocm-smi") or shutil.which("rocminfo"):
        return "rocm"
    return "cpu"


def ensure_lockfile(backend: str) -> Path:
    lockfile = Path(LOCKFILES[backend])
    if lockfile.exists():
        return lockfile

    print(f"🔒 Creating lockfile: {lockfile}")

    env = os.environ.copy()
    if backend != "cpu":
        env["UV_EXTRAS"] = backend
    else:
        env["UV_EXTRAS"] = ""

    # Run lock (default uv.lock)
    run(["uv", "lock"], env=env)

    # Rename uv.lock to backend-specific lockfile
    default_lock = Path("uv.lock")
    if default_lock.exists():
        default_lock.rename(lockfile)

    return lockfile


def main():
    backend = detect_backend()
    print(f"🔍 Detected backend: {backend}")

    ensure_lockfile(backend)

    if backend == "cpu":
        run(["uv", "sync"])
    else:
        run(["uv", "sync", "--extra", backend])


if __name__ == "__main__":
    main()
