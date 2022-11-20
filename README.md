
# Deforum Stable Diffusion

<p align="left">
    <a href="https://github.com/deforum-art/deforum-stable-diffusion/commits"><img alt="Last Commit" src="https://img.shields.io/github/last-commit/deforum-art/deforum-stable-diffusion"></a>
    <a href="https://github.com/deforum-art/deforum-stable-diffusion/issues"><img alt="GitHub issues" src="https://img.shields.io/github/issues/deforum-art/deforum-stable-diffusion"></a>
    <a href="https://github.com/deforum-art/deforum-stable-diffusion/stargazers"><img alt="GitHub stars" src="https://img.shields.io/github/stars/deforum-art/deforum-stable-diffusion"></a>
    <a href="https://github.com/deforum-art/deforum-stable-diffusion/network"><img alt="GitHub forks" src="https://img.shields.io/github/forks/deforum-art/deforum-stable-diffusion"></a>
    <a href="https://colab.research.google.com/github/deforum-art/deforum-stable-diffusion/blob/main/Deforum_Stable_Diffusion.ipynb"><img alt="Colab" src="https://colab.research.google.com/assets/colab-badge.svg"></a>  
</p>

## Before You Start
- make sure you have the latest nvidia drivers https://developer.nvidia.com/cuda-downloads
- install anaconda for managing python environments and packages https://www.anaconda.com/
- create a huggingface token which you will need for auto model download: https://huggingface.co/settings/tokens
- install ffmpeg https://ffmpeg.org/download.html
- install git for your system. you can install git with anaconda:
```
conda install -c anaconda git -y

```

## Getting Started
1. open anaconda powershell (on Windows) or terminal (Linux) and navigate to install location
2. clone the github repository:
```
git clone https://github.com/deforum-art/deforum-stable-diffusion.git
cd deforum-stable-diffusion

```
3. create anaconda environment:
```
conda create -n dsd python=3.10 -y
conda activate dsd
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia -y

```
4. install required packages:
```
python -m pip install -r requirements.txt

```
5. check your installation by running the .py
```
python Deforum_Stable_Diffusion.py

```
you have successfully installed deforum stable diffusion if the python file runs without any errors.


## Installing Xformers
xformers can be installed from source with the following commands
```
git clone https://github.com/facebookresearch/xformers.git
cd xformers
conda activate dsd
git submodule update --init --recursive
pip install -r requirements.txt
pip install -e .

```
xformers can be enabled by switching the attention.py
```
cp src/ldm/modules/attention_xformers.py src/ldm/modules/attention.py

```
to turn off xformers run the following:
```
cp src/ldm/modules/attention_backup.py src/ldm/modules/attention.py

```

## Starting Over
the stable-diffusion folder can be deleted and the dsd conda environment can be removed with the following set of commands:
```
conda deactivate
conda env remove -n dsd

```
with the dsd environment removed you can start over.


## Running Deforum Stable Diffusion
there are four ways to run deforum stable diffusion: locally with the .py file, locally with jupyter, locally through colab, and on colab severs

### Running Locally
make sure the dsd conda environment is active:
```
conda activate dsd

```
navigate to the stable-diffusion folder and run either the Deforum_Stable_Diffusion.py or the Deforum_Stable_Diffusion.ipynb. running the .py is the quickest and easiest way to check that your installation is working, however, it is not the best environment for tinkering with prompts and settings
```
python Deforum_Stable_Diffusion.py

```
if you prefer a more colab-like experience you can run the .ipynb in jupyter-lab or jupyter-notebook. activate jupyter-lab or jupyter-notebook from within the stable-diffusion folder with either of the following commands:
```
jupyter-lab

```
```
jupyter notebook

```


### Colab Local Runtime
make sure the dsd conda environment is active:
```
conda activate dsd

```
open google colab. file > upload notebook > select .ipynb file in the stable-diffusion folder. enable jupyter extension. note: you only need to run this cell one time
```
jupyter serverextension enable --py jupyter_http_over_ws

```
start server
```
jupyter notebook --NotebookApp.allow_origin='https://colab.research.google.com' --port=8888 --NotebookApp.port_retries=0
  
```
copy paste url token


### Colab Hosted Runtime
Deforum_Stable_Diffusion.ipynb can be uploaded to colab and run normally in a hosted session
