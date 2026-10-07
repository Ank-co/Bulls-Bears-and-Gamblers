#!/usr/bin/env bash
# Environment for bulls-bears-and-gamblers, inside WSL2 Ubuntu.
# Run from the repository root:   bash setup/setup_wsl.sh
#
# GPU in WSL2 only needs the NVIDIA driver installed on Windows.
# Do NOT install a CUDA toolkit or a Linux NVIDIA driver inside WSL.
set -euo pipefail

echo "== GPU visible from WSL?"
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found. Update the NVIDIA driver on Windows, then run 'wsl --update'."
  exit 1
fi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

echo "== System packages"
sudo apt-get update -qq
sudo apt-get install -y -qq python3-venv python3-pip git

# The venv lives on the Linux filesystem: much faster than a venv under /mnt/c.
VENV="${VENV:-$HOME/.venvs/bbg}"
echo "== Virtual environment ($VENV)"
python3 -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --upgrade pip -q

echo "== PyTorch built for CUDA 12.8 (required by RTX 50-series / sm_120)"
pip install -q torch --index-url https://download.pytorch.org/whl/cu128

echo "== Project dependencies"
pip install -q -r requirements.txt
pip install -q -e .
pip freeze > requirements.lock.txt

echo "== Unit tests"
python -m pytest -q tests

echo "== GPU check"
python scripts/check_gpu.py
