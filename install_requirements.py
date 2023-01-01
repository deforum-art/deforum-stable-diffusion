#!/usr/bin/env python

import subprocess

# List of packages to install
packages = [
    "clean-fid",
    "colab-convert",
    "einops",
    "ffmpeg",
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
    "triton",
    "git+https://github.com/facebookresearch/xformers.git@main#egg=xformers",
]

# Install each package
for package in packages:
    try:
        subprocess.call(["pip", "install", package])
    except Exception as e:
        print(f"Failed to install {package}: {e}")
