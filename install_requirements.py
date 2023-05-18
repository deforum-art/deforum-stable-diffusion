import argparse
import platform
import subprocess


def pip_install_packages(packages, extra_index_url=None, verbose=False):
    for package in packages:
        try:
            print(f"..installing {package}")
            
            # base command
            cmd = ["pip", "install"]

            # add '-q' if not verbose
            if not verbose:
                cmd.append("-q")

            # add package name
            cmd.append(package)

            # add extra_index_url if it exists
            if extra_index_url:
                cmd.extend(["--extra-index-url", extra_index_url])

            # run the command and capture output
            result = subprocess.run(cmd, capture_output=not verbose, text=True)
            
            if verbose:
                # print stdout and stderr if verbose
                print(result.stdout)
                print(result.stderr)

        except Exception as e:
            print(f"failed to install {package}: {e}")
    return


def install_requirements(verbose=False):

    # Detect System
    os_system = platform.system()
    print(f"system detected: {os_system}")

    # Install pytorch
    torch = [
        "torch",
        "torchvision",
        "torchaudio"
    ]
    extra_index_url = "https://download.pytorch.org/whl/cu117" if os_system == 'Windows' else None
    pip_install_packages(torch, extra_index_url=extra_index_url, verbose=verbose)

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
        "opencv-python",
        "pandas",
        "pytorch_lightning==1.7.7",
        "resize-right",
        "scikit-image==0.19.3",
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
        "triton",
        "xformers",
    ]
    windows_xformers = [
        "xformers",
    ]
    xformers = windows_xformers if os_system == 'Windows' else linux_xformers
    pip_install_packages(xformers)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', action='store_true', help='print pip install stuff')
    args = parser.parse_args()
    install_requirements(verbose=args.verbose)