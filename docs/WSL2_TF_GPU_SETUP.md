WSL2 TensorFlow GPU setup (verified project path)

This project was verified in WSL2 on Ubuntu 24.04 with an NVIDIA GeForce GTX 1650. The important fix was to keep the Python virtual environment on the Linux filesystem and add TensorFlow's pip-installed NVIDIA CUDA libraries to `LD_LIBRARY_PATH` before importing TensorFlow.

Verified on 2026-07-04:

- `nvidia-smi` sees `NVIDIA GeForce GTX 1650`, driver `610.62`, `4096 MiB`
- TensorFlow `2.21.0` imports in WSL Python `3.12.3`
- `tensorflow[and-cuda]==2.21.0` installs CUDA 12 NVIDIA wheels, including cuDNN `9.24.0.43`
- TensorFlow lists `/physical_device:GPU:0`
- cuDNN loads during CNN training
- EfficientNetB0 and ResNet50 both completed tiny WSL GPU calibration training runs

1) Windows host driver
- Install NVIDIA driver for Windows that supports WSL (Studio or Game Ready driver that explicitly mentions WSL). Check NVIDIA docs: https://developer.nvidia.com/cuda/wsl
- Reboot Windows after driver install.

2) WSL2 distro preparation (assumes Ubuntu 22.04+)
- Update packages:

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential curl ca-certificates gnupg lsb-release
```

3) Python environment on Linux filesystem

Do not use a venv stored under `/mnt/c/...` for the WSL GPU path. Create it under `~`:

```bash
PROJECT_ROOT="/mnt/c/Antony/Christ uni/3rd year/SIC/car-recognition-maintenance-advisor"
python3 -m venv "$HOME/.venvs/car-cnn-wsl"
source "$HOME/.venvs/car-cnn-wsl/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r "$PROJECT_ROOT/requirements.txt"
python -m pip install --upgrade "tensorflow[and-cuda]==2.21.0"
```

4) Activate with CUDA library paths

Always activate the WSL GPU venv through the helper script before TensorFlow commands:

```bash
cd "$PROJECT_ROOT"
source scripts/activate_wsl_gpu.sh
```

The helper activates `$HOME/.venvs/car-cnn-wsl` and prepends the NVIDIA pip wheel library folders to `LD_LIBRARY_PATH`.

5) Verify GPU access

```bash
cd "$PROJECT_ROOT"
source scripts/activate_wsl_gpu.sh
python scripts/check_tf_gpu.py
```

Expected key lines:

```text
physical GPUs: ['/physical_device:GPU:0']
device_name: NVIDIA GeForce GTX 1650
```

6) Run short GPU calibration training

EfficientNetB0:

```bash
cd "$PROJECT_ROOT"
source scripts/activate_wsl_gpu.sh
python -m src.model.train_cnn --architecture efficientnetb0 \
	--data-dir data/car_data/car_data \
	--image-size 128 --batch-size 4 --epochs 1 \
	--smoke-batches 2 --weights none \
	--output-dir "$HOME/artifacts/wsl_gpu_calibration"
```

ResNet50:

```bash
cd "$PROJECT_ROOT"
source scripts/activate_wsl_gpu.sh
python -m src.model.train_cnn --architecture resnet50 \
	--data-dir data/car_data/car_data \
	--image-size 128 --batch-size 2 --epochs 1 \
	--smoke-batches 1 --weights none \
	--output-dir "$HOME/artifacts/wsl_gpu_calibration"
```

7) If TensorFlow installs but shows no GPUs
- Ensure Windows NVIDIA driver supports WSL and that `nvidia-smi` runs inside WSL and shows GPUs
- Ensure the command was run after `source scripts/activate_wsl_gpu.sh`
- Confirm `LD_LIBRARY_PATH` includes paths like `.../site-packages/nvidia/cudnn/lib`
- Confirm `tensorflow[and-cuda]==2.21.0` installed the `nvidia-*-cu12` packages
- The `/usr/local/cuda/version.txt` file may be absent when using TensorFlow's pip CUDA wheels; this is okay if TensorFlow lists the GPU.

8) Next Stage 2 steps
- Keep using `source scripts/activate_wsl_gpu.sh` for WSL GPU training
- Run longer calibration to choose safe batch sizes for the GTX 1650's 4 GB VRAM
- Proceed to full EfficientNetB0 and ResNet50 training only after short calibration is stable
