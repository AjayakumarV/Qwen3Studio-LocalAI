"""
One-Command Automated Installer for Qwen3Studio (LocalAI-Desktop).
Automatically sets up:
  1. ~/LocalAI directory structure (or $LOCALAI_HOME)
  2. llama.cpp Windows x64 CUDA 12.4 binary release (b12121+) + CUDA runtime DLLs
  3. Unsloth Dynamic Quantized Qwen3-30B-A3B-UD-IQ3_XXS.gguf (12.0 GB)
  4. Unsloth Dynamic Quantized Qwen3-0.6B-UD-Q4_K_XL.gguf (386.6 MB speculative draft model)
  5. Offline Faster-Whisper base.en INT8 Speech-to-Text model (141 MB)

Usage:
  pip install -r requirements.txt
  python setup_localai.py
"""

import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


LOCALAI_HOME = Path(os.environ.get("LOCALAI_HOME", str(Path.home() / "LocalAI")))
BIN_DIR = LOCALAI_HOME / "bin"
MODELS_DIR = LOCALAI_HOME / "models"
WHISPER_DIR = MODELS_DIR / "whisper-base.en"

LLAMA_TAG = "b12121"
LLAMA_CUDA_URL = f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_TAG}/llama-{LLAMA_TAG}-bin-win-cuda-12.4-x64.zip"
CUDART_URL = f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_TAG}/cudart-llama-bin-win-cuda-12.4-x64.zip"

MODEL_30B_URL = "https://huggingface.co/unsloth/Qwen3-30B-A3B-GGUF/resolve/main/Qwen3-30B-A3B-UD-IQ3_XXS.gguf"
MODEL_06B_URL = "https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-UD-Q4_K_XL.gguf"
WHISPER_FILES = ["config.json", "model.bin", "tokenizer.json", "vocabulary.txt"]


def download_with_curl(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "curl.exe" if os.name == "nt" else "curl",
        "-L",
        "--ssl-no-revoke" if os.name == "nt" else "-L",
        "-C", "-",
        "--fail",
        "--retry", "5",
        "-o", str(dest),
        url,
    ]
    print(f"  -> Downloading {dest.name}...")
    subprocess.run(cmd, check=True)


def install_llama_cpp():
    server_exe = BIN_DIR / ("llama-server.exe" if os.name == "nt" else "llama-server")
    if server_exe.exists():
        print(f"[OK] llama-server already installed at {server_exe}")
        return

    if os.name != "nt":
        print("[INFO] Non-Windows OS detected. Please install llama.cpp via 'brew install llama.cpp' or build from source.")
        return

    tmp_dir = LOCALAI_HOME / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    llama_zip = tmp_dir / "llama.zip"
    cudart_zip = tmp_dir / "cudart.zip"

    print(f"[1/4] Downloading llama.cpp {LLAMA_TAG} CUDA 12.4 binaries...")
    download_with_curl(LLAMA_CUDA_URL, llama_zip)
    download_with_curl(CUDART_URL, cudart_zip)

    print("  -> Extracting archives...")
    BIN_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(llama_zip, "r") as zf:
        zf.extractall(BIN_DIR)
    with zipfile.ZipFile(cudart_zip, "r") as zf:
        zf.extractall(BIN_DIR)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"[OK] Extracted llama-server to {BIN_DIR}")


def install_models(skip_30b: bool = False):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Whisper Base.en Offline STT
    print("[2/4] Checking Offline Faster-Whisper STT model...")
    WHISPER_DIR.mkdir(parents=True, exist_ok=True)
    for fname in WHISPER_FILES:
        fpath = WHISPER_DIR / fname
        if not fpath.exists():
            url = f"https://huggingface.co/Systran/faster-whisper-base.en/resolve/main/{fname}"
            download_with_curl(url, fpath)
    print(f"[OK] Offline Whisper STT ready at {WHISPER_DIR}")

    # 2. Qwen3-0.6B Speculative Draft Model (386 MB)
    draft_path = MODELS_DIR / "Qwen3-0.6B-UD-Q4_K_XL.gguf"
    print("[3/4] Checking Qwen3-0.6B Speculative Draft model (386.6 MB)...")
    if not draft_path.exists():
        download_with_curl(MODEL_06B_URL, draft_path)
    print(f"[OK] Draft model ready at {draft_path}")

    # 3. Qwen3-30B-A3B Target MoE Model (12.0 GB)
    target_path = MODELS_DIR / "Qwen3-30B-A3B-UD-IQ3_XXS.gguf"
    if skip_30b:
        print("[SKIP] Skipping 12.0 GB Qwen3-30B-A3B download (--skip-30b passed).")
        return

    print("[4/4] Checking Qwen3-30B-A3B-UD-IQ3_XXS.gguf (12.0 GB)...")
    if not target_path.exists() or target_path.stat().st_size < 11_000_000_000:
        download_with_curl(MODEL_30B_URL, target_path)
    print(f"[OK] Qwen3-30B-A3B ready at {target_path}")


def main():
    parser = argparse.ArgumentParser(description="Setup Qwen3Studio LocalAI Engine & Models")
    parser.add_argument("--skip-30b", action="store_true", help="Skip downloading the 12 GB 30B model")
    args = parser.parse_args()

    print("==================================================================")
    print(" Qwen3Studio — Automated Local AI Workstation Installer")
    print(f" Target Directory: {LOCALAI_HOME}")
    print("==================================================================")
    install_llama_cpp()
    install_models(skip_30b=args.skip_30b)
    print("\n[SUCCESS] All components installed! Launch the desktop studio with:")
    print("  python app.py")


if __name__ == "__main__":
    main()
