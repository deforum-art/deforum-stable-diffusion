import platform
import sys
import subprocess

os_system = platform.system()
print(f"system detected: {os_system}")

print(f"..installing packages in requirements.txt")
pip_requirements_txt_cmd = [sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt']

# For some reason torch needs a specific index url to install https://pytorch.org/get-started/locally/
torch_extras_if_windows = ['--extra-index-url','https://download.pytorch.org/whl/cu117'] if os_system == 'Windows' else []
command = pip_requirements_txt_cmd + torch_extras_if_windows
print(f"command: {command}")
subprocess.call(command, shell=False)
