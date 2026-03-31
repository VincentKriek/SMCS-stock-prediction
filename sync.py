import shutil
import subprocess
import sys


def detect_backend() -> str:
    if shutil.which("nvidia-smi"):
        return "cu121"
    # Check for ROCm (AMD)
    if shutil.which("rocm-smi") or shutil.which("rocminfo"):
        return "rocm60"
    return "cpu"


def main():
    backend = detect_backend()
    print(f"🔍 Hardware detected: {backend.upper()}")

    # We run uv sync with the specific extra
    cmd = ["uv", "sync", "--extra", backend]

    try:
        print(f"🚀 Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        print("✅ Environment synchronized!")

        # Verification
        # NOTE: We MUST use --extra here too, or uv run will hide torch
        print("\n🧪 Verifying...")
        subprocess.run(
            [
                "uv",
                "run",
                "python",
                "-c",
                "import torch; print(f'✅ Found Torch {torch.__version__} | CUDA/ROCm Available: {torch.cuda.is_available()}')",
            ],
            check=True,
        )

    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
