#!/usr/bin/env bash
# Create the `teleop` conda environment on Ubuntu 24.04.
#
# Linux counterpart to setup_env.sh. Two things differ, and both matter:
#
# 1. torch comes from the CUDA 12.8 index. The RTX 5080 is Blackwell (sm_120),
#    and a default `pip install torch` can give you a build without sm_120
#    kernels. That build imports fine, reports cuda.is_available() == True, and
#    then fails at the first real kernel launch -- a confusing way to find out.
#    check_linux_stack.py verifies with an actual matmul rather than trusting
#    is_available().
#
# 2. wujihandpy installs here. It has manylinux wheels and is the entire reason
#    Linux wheels only.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

ENV_NAME="${ENV_NAME:-teleop}"
CUDA_INDEX="${CUDA_INDEX:-https://download.pytorch.org/whl/cu128}"

if ! command -v conda >/dev/null 2>&1; then
  echo "error: conda not found on PATH." >&2
  exit 1
fi

# ---------------------------------------------------------------- system deps
echo "==> System packages (adb for the Quest, libusb for the hand)"
if command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update -qq
  sudo apt-get install -y android-sdk-platform-tools libusb-1.0-0 build-essential
else
  echo "    apt-get not found; install adb and libusb-1.0 manually." >&2
fi

# ---------------------------------------------------------------- conda env
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "==> Environment '$ENV_NAME' exists; updating it."
  conda env update -n "$ENV_NAME" -f environment.yml --prune
else
  echo "==> Creating environment '$ENV_NAME'."
  conda env create -f environment.yml
fi

# ---------------------------------------------------------------- torch (CUDA)
# environment.yml installs a default torch; replace it with the CUDA build.
echo "==> Installing torch from $CUDA_INDEX (Blackwell / sm_120)"
conda run -n "$ENV_NAME" pip uninstall -y torch >/dev/null 2>&1 || true
conda run -n "$ENV_NAME" pip install --index-url "$CUDA_INDEX" torch

# ---------------------------------------------------------------- the hand
echo "==> Installing wujihandpy"
conda run -n "$ENV_NAME" pip install wujihandpy

echo "==> Installing quest_teleop2 (editable)"
conda run -n "$ENV_NAME" pip install -e . --no-deps -q

# ---------------------------------------------------------------- verify
echo "==> Verifying"
conda run -n "$ENV_NAME" python - <<'PY'
import platform, sys
print(f"  python   {sys.version.split()[0]} ({platform.machine()})")

import torch
print(f"  torch    {torch.__version__}  cuda build: {torch.version.cuda}")
if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    print(f"  gpu      {name}  compute capability sm_{cap[0]}{cap[1]}")
    arches = torch.cuda.get_arch_list()
    print(f"  arches   {arches}")
    if f"sm_{cap[0]}{cap[1]}" not in arches:
        print("  \033[31mWARNING\033[0m this torch build has no kernels for your GPU.")
        print("          Reinstall with a newer CUDA index; see CUDA_INDEX in this script.")
else:
    print("  \033[33mgpu      CUDA not available -- training will fall back to CPU\033[0m")

try:
    import wujihandpy
    print(f"  wujihandpy {getattr(wujihandpy, '__version__', 'installed')}")
except Exception as exc:
    print(f"  \033[31mwujihandpy import failed: {exc}\033[0m")

import numpy, scipy, h5py, yaml, aiohttp, rtde  # noqa: F401
import quest_teleop2  # noqa: F401
print("  quest_teleop2 and all dependencies import OK")
PY

if command -v adb >/dev/null 2>&1; then
  echo "==> adb: $(adb version | head -1)"
else
  echo "==> \033[33madb not on PATH\033[0m -- needed to reach the Quest over USB."
fi

cat <<EOF

Done. Next:
    sudo setup/install_udev.sh     # USB access for the Quest and the hand
    # log out and back in for the plugdev group to take effect
    conda activate $ENV_NAME
    python scripts/troubleshooting/check_linux_stack.py
EOF
