
# Deforum Stable Diffusion

<p align="left">
    <a href="https://github.com/deforum-art/deforum-stable-diffusion/commits"><img alt="Last Commit" src="https://img.shields.io/github/last-commit/deforum/stable-diffusion"></a>
    <a href="https://github.com/deforum-art/deforum-stable-diffusion/issues"><img alt="GitHub issues" src="https://img.shields.io/github/issues/deforum/stable-diffusion"></a>
    <a href="https://github.com/deforum-art/deforum-stable-diffusion/stargazers"><img alt="GitHub stars" src="https://img.shields.io/github/stars/deforum/stable-diffusion"></a>
    <a href="https://github.com/deforum-art/deforum-stable-diffusion/network"><img alt="GitHub forks" src="https://img.shields.io/github/forks/deforum-art/stable-diffusion"></a>
    <a href="https://colab.research.google.com/github/deforum/stable-diffusion/blob/main/Deforum_Stable_Diffusion.ipynb"><img alt="Colab" src="https://colab.research.google.com/assets/colab-badge.svg"></a>  
    <a href="https://replicate.com/deforum-art/deforum_stable_diffusion"><img alt="Replicate" src="https://replicate.com/deforum/deforum_stable_diffusion/badge"></a>
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
cd stable-diffusion

```
3. create anaconda environment:
```
conda create -n dsd python=3.10 -y
conda activate dsd
conda install pytorch cudatoolkit=11.6 torchvision torchaudio -c pytorch -c conda-forge -y

```
4. install required packages:
```
python -m pip install -r requirements.txt

```
5. check your installation by running the .py
```
python Deforum_Stable_Diffusion.py

```
you have successfully installed deforum stable diffusion if the python file runs without any errors. if you get "out of memory" errors you can try installing xformers for your system (see either Windows Users or Linux Users below)


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
Deforum_Stable_Diffusion.ipynb can be uploaded to colab and run normally in a hosted session.


## Windows Users
the midas and adabins model downloads are broken for windows at the moment. windows users will need to manually download model weights and place in the models folders. note: if you do not specify an existing models folder, the folder will be created automatically when you run either the .py or .ipynb for the first time.

manual download links:

https://github.com/intel-isl/DPT/releases/download/1_0/dpt_large-midas-2f21e586.pt

https://cloudflare-ipfs.com/ipfs/Qmd2mMnDLWePKmgfS8m6ntAg4nhV5VkUyAydYBp8cWWeB7/AdaBins_nyu.pt

xformers can be installed with the following commands:
```
wget https://github.com/neonsecret/xformers/releases/download/v0.14/xformers-0.0.14.dev0-cp39-cp39-win_amd64.whl
pip install xformers-0.0.14.dev0-cp39-cp39-win_amd64.whl

```
xformers can be enabled by switching the attention.py
```
mv src/ldm/modules/attention.py src/ldm/modules/attention_backup.py
mv src/ldm/modules/attention_xformers.py src/ldm/modules/attention.py

```
to turn off xformers run the following:
```
mv src/ldm/modules/attention.py src/ldm/modules/attention_xformers.py 
mv src/ldm/modules/attention_backup.py src/ldm/modules/attention.py

```

## Linux Users
xformers can be installed with the following commands:
```
conda install xformers -c xformers/label/dev -y

```
xformers can be enabled by switching the attention.py
```
mv src/ldm/modules/attention.py src/ldm/modules/attention_backup.py
mv src/ldm/modules/attention_xformers.py src/ldm/modules/attention.py

```
to turn off xformers run the following:
```
mv src/ldm/modules/attention.py src/ldm/modules/attention_xformers.py 
mv src/ldm/modules/attention_backup.py src/ldm/modules/attention.py

```


## Starting Over
the stable-diffusion folder can be deleted and the dsd conda environment can be removed with the following set of commands:
```
conda deactivate
conda env remove -n dsd

```
with the dsd environment removed you can start over.
