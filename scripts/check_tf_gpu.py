"""Lightweight TensorFlow GPU diagnostic for WSL2.

Run inside your project WSL2 Python environment. Prints system and GPU/TensorFlow details
that help determine missing pieces for GPU support.
"""
import platform
import subprocess
import sys
import json
from importlib import metadata

print("== System ==")
print("Platform:", platform.platform())
print("Python:", sys.version.replace('\n', ' '))

print('\n== uname -a ==')
try:
    print(subprocess.check_output(['uname', '-a'], universal_newlines=True))
except Exception as e:
    print('uname failed:', e)

print('\n== lsb_release -a ==')
try:
    print(subprocess.check_output(['lsb_release', '-a'], universal_newlines=True))
except Exception:
    pass

print('\n== nvidia-smi ==')
try:
    print(subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv'], universal_newlines=True))
except Exception as e:
    print('nvidia-smi not available or failed:', e)

print('\n== /usr/local/cuda/version.txt ==')
try:
    with open('/usr/local/cuda/version.txt','r') as f:
        print(f.read().strip())
except Exception:
    print('CUDA version file not found at /usr/local/cuda/version.txt')

print('\n== pip packages (tensorflow and NVIDIA CUDA wheels) ==')
try:
    package_names = sorted(dist.metadata['Name'] for dist in metadata.distributions())
    for key in package_names:
        normalized = key.lower().replace('_', '-')
        if normalized == 'tensorflow' or normalized.startswith('nvidia-'):
            print(f"{key}: {metadata.version(key)}")
except Exception as e:
    print('pip package scan failed:', e)

print('\n== LD_LIBRARY_PATH ==')
print(__import__('os').environ.get('LD_LIBRARY_PATH', '<unset>'))

print('\n== TensorFlow probe ==')
try:
    import tensorflow as tf
    print('tensorflow version:', tf.__version__)
    try:
        gpus = tf.config.list_physical_devices('GPU')
        print('physical GPUs:', [d.name for d in gpus])
        if gpus:
            for gpu in gpus:
                try:
                    details = tf.config.experimental.get_device_details(gpu)
                    print(' -', gpu, details)
                except Exception:
                    pass
    except Exception as e:
        print('tf.config.list_physical_devices failed:', e)

    print('\nRun a tiny GPU op to confirm execution device:')
    try:
        with tf.device('/GPU:0'):
            a = tf.constant([[1.0, 2.0],[3.0,4.0]])
            b = tf.matmul(a, a)
            print('matmul result:', b.numpy())
    except Exception as e:
        print('GPU matmul failed or GPU not found:', e)
except Exception as e:
    print('Importing TensorFlow failed:', e)

print('\n== end ==')
