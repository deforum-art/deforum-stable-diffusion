import platform
import subprocess


def pip_install_packages(packages, extra_index_url=None):
    for package in packages:
        try:
            print(f"..installing {package}")
            if extra_index_url is not None:
                running = subprocess.call(
                    [
                        "pip",
                        "install",
                        "-q",
                        package,
                        "--extra-index-url",
                        extra_index_url,
                    ],
                    shell=False,
                )
            else:
                running = subprocess.call(
                    ["pip", "install", "-q", package], shell=False
                )
        except Exception as e:
            print(f"failed to install {package}: {e}")
    return


def install_requirements():
    # Detect System
    os_system = platform.system()
    print(f"system detected: {os_system}")

    # Install pytorch
    torch = ["torch", "torchvision", "torchaudio"]

    extra_index_url = (
        "https://download.pytorch.org/whl/cu117" if os_system == "Windows" else None
    )
    pip_install_packages(torch, extra_index_url=extra_index_url)

    # List of common packages to install
    common = [
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
        "opencv-contrib-python",
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

    pip_install_packages(common)

    # Xformers install
    linux_xformers = [
        "triton==2.0.0.dev20221202",
        "xformers==0.0.16",
    ]

    windows_xformers = [
        "https://huggingface.co/deforum/xformers/resolve/main/windows/xformers-0.0.15.dev0fd21b40.d20230107-cp310-cp310-win_amd64.whl",
    ]

    xformers = windows_xformers if os_system == "Windows" else linux_xformers
    pip_install_packages(xformers)


if __name__ == "__main__":
    install_requirements()
