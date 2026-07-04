#!/usr/bin/env bash
# Activate the Linux-filesystem WSL TensorFlow GPU environment for this project.
# Usage from project root:
#   source scripts/activate_wsl_gpu.sh

VENV_PATH="${CAR_CNN_WSL_VENV:-$HOME/.venvs/car-cnn-wsl}"
if [[ ! -f "$VENV_PATH/bin/activate" ]]; then
  echo "WSL GPU venv not found at: $VENV_PATH" >&2
  echo "Create it with: python3 -m venv $VENV_PATH" >&2
  return 1 2>/dev/null || exit 1
fi

# shellcheck disable=SC1090
source "$VENV_PATH/bin/activate"

_NVIDIA_LIB_PATHS="$($VENV_PATH/bin/python - <<'PY'
import glob
import os
import site
paths=[]
for sp in site.getsitepackages():
    paths.extend(sorted(glob.glob(os.path.join(sp, 'nvidia', '*', 'lib'))))
print(':'.join(paths))
PY
)"

if [[ -n "$_NVIDIA_LIB_PATHS" ]]; then
  export LD_LIBRARY_PATH="$_NVIDIA_LIB_PATHS:${LD_LIBRARY_PATH:-}"
fi

export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-0}"
echo "Activated WSL GPU venv: $VENV_PATH"
echo "NVIDIA pip library paths added to LD_LIBRARY_PATH."
