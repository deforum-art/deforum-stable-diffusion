import platform
import subprocess

os_system = platform.system()
print(f"system detected: {os_system}")

print(f"..installing packages in requirements.txt")
pip_requirements_txt_cmd = ["pip", "install", "-r requirements.txt"]

# For some reason torch needs a specific index url to install https://pytorch.org/get-started/locally/
torch_extras_if_windows = ["--extra-index-url","https://download.pytorch.org/whl/cu117"] if os_system == 'Windows' else []

subprocess.call(pip_requirements_txt_cmd + torch_extras_if_windows, shell=False)
