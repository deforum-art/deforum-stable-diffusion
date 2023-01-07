#!/usr/bin/env python

import platform
import subprocess

# List of packages to install

packages = [
    "clean-fid",
    "colab-convert",
    "einops",
    "ftfy",
    "ipython",
    "ipywidgets",
    "jsonmerge",
    "jupyterlab",
    "jupyter_http_over_ws",
    "kornia",
    "matplotlib",
    "notebook",
    "numexpr",
    "omegaconf",
    "opencv-python",
    "pandas",
    "pytorch_lightning==1.7.7",
    "resize-right",
    "scikit-image",
    "scikit-learn",
    "timm",
    "torchdiffeq",
    "transformers==4.19.2",
    "safetensors",
    "albumentations",
    "more_itertools",
    "devtools",
    "validators",
    "numpngw",
    "open-clip-torch",
    "torchsde",
    "ninja",
]

linux_packages = [
    "triton",
    "git+https://github.com/facebookresearch/xformers.git@main#egg=xformers",
]

windows_packages = [
    #"https://github.com/C43H66N12O12S2/stable-diffusion-webui/releases/download/f/xformers-0.0.14.dev0-cp310-cp310-win_amd64.whl",
]

# Install each package
for package in packages:
    try:
        print(f"..installing {package}")
        running = subprocess.call(["pip", "install", "-q", package],shell=False)
    except Exception as e:
        print(f"failed to install {package}: {e}")

os_system = platform.system()
print(f"system detected: {os_system}")
packages = windows_packages if os_system == 'Windows' else linux_packages

for package in packages:
    try:
        print(f"..installing {package}")
        running = subprocess.call(["pip", "install", "-q", package],shell=False)
    except Exception as e:
        print(f"failed to install {package}: {e}")
