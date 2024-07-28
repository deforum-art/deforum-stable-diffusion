# Deforum Stable Diffusion

⚠️ **NOTICE: This project is no longer maintained.** ⚠️

This repository is no longer actively maintained or updated. Users are advised to find alternative solutions or fork the project if they wish to continue development.

<p align="left">
    <a href="https://github.com/deforum-art/deforum-stable-diffusion/commits"><img alt="Last Commit" src="https://img.shields.io/github/last-commit/deforum-art/deforum-stable-diffusion"></a>
    <a href="https://github.com/deforum-art/deforum-stable-diffusion/issues"><img alt="GitHub issues" src="https://img.shields.io/github/issues/deforum-art/deforum-stable-diffusion"></a>
    <a href="https://github.com/deforum-art/deforum-stable-diffusion/stargazers"><img alt="GitHub stars" src="https://img.shields.io/github/stars/deforum-art/deforum-stable-diffusion"></a>
    <a href="https://github.com/deforum-art/deforum-stable-diffusion/network"><img alt="GitHub forks" src="https://img.shields.io/github/forks/deforum-art/deforum-stable-diffusion"></a>
    <a href="https://colab.research.google.com/github/deforum-art/deforum-stable-diffusion/blob/main/Deforum_Stable_Diffusion.ipynb"><img alt="Colab" src="https://colab.research.google.com/assets/colab-badge.svg"></a>  
    <a href="https://replicate.com/deforum-art/deforum-stable-diffusion"><img alt="Replicate" src="https://replicate.com/deforum-art/deforum-stable-diffusion/badge"></a>  
</p>

Welcome to Deforum Stable Diffusion!

## Table of Contents

- [Introduction](#introduction)
- [Key Features](#key-features)
- [Before You Start](#before-you-start)
- [Getting Started](#getting-started)
- [Running Deforum](#running-deforum)
- [Starting Over](#starting-over)
- [Customization](#customization)
- [Contributing](#contributing)
- [Support Us](#support-us)

## Introduction

We are a community of programmers and artists who are passionate about making stable diffusion machine learning image synthesis accessible to everyone. Our open source project is designed to be free to use and easy to modify for custom needs and pipelines. We believe in the power of collaboration and are constantly working together to improve and evolve our implementation of stable diffusion. Whether you are an experienced developer or just getting started, we invite you to join us and be a part of this exciting project.

You can now also run Deforum Stable Diffusion easily on Replicate, check out the web demo and the API here [![Replicate](https://replicate.com/deforum-art/deforum-stable-diffusion/badge)](https://replicate.com/deforum-art/deforum-stable-diffusion) 

## Key Features

- Our implementation is written in an IPython notebook and was designed for use with Google Colab.
- In response to the implementation of the credit system, we have added support for local run times and will be adding a web user interfaces.
- The notebook includes a variety of features for generating interpolation, 2D and 3D animations, and RANSAC animations.
- We also offer CLIP, aesthetic, and color pallet conditioning.
- Our goal is to provide users with a range of tools and options for creating stable diffusion images.

## Before You Start

Before you start installing and using Deforum Stable Diffusion, there are a few things you need to do:

1. Install [ffmpeg](https://ffmpeg.org/download.html). FFmpeg is a free software project that produces libraries and programs for handling multimedia data. You will need it to process audio and video files. Follow the instructions on the website to download and install FFmpeg on your system (https://ffmpeg.org/ffmpeg.html). Once it is installed, make sure it is in your PATH by running `ffmpeg -h` in your terminal. If you don't get an error message, you're good to go. A guide for windows (https://phoenixnap.com/kb/ffmpeg-windows).
2. Install the latest NVIDIA drivers for CUDA 11.7 (may not be necessary for Windows users). NVIDIA CUDA is a parallel computing platform and programming model that enables developers to use the power of NVIDIA graphics processing units (GPUs) to speed up compute-intensive tasks. You will need to install the latest NVIDIA drivers to use Deforum Stable Diffusion. You can find the drivers [here](https://developer.nvidia.com/cuda-toolkit-archive). Follow the instructions on the website to download and install the drivers.
3. Create a [huggingface token](https://huggingface.co/settings/tokens). Hugging Face is a natural language processing platform that provides access to state-of-the-art models and tools. You will need to create a token in order to use some of the automatic model download features in Deforum Stable Diffusion. Follow the instructions on the Hugging Face website to create a token.
4. Install [Anaconda](https://www.anaconda.com/). Anaconda is a free and open-source distribution of Python and R. It includes a package manager called conda that makes it easy to install and manage Python environments and packages. Follow the instructions on the Anaconda website to download and install Anaconda on your system.
5. Install Git for your system. Git is a version control system that helps you track changes to your code and collaborate with other developers. You can install Git with Anaconda by running `conda install -c anaconda git -y` in your terminal. If you have trouble installing Git via Anaconda, you can use the following links instead:
   - [Git for Windows](https://git-scm.com/download/win)
   - [Git for Linux](https://git-scm.com/download/linux)

Once you have completed these steps, you will be ready to install Deforum Stable Diffusion.

## Getting Started

To install Deforum Stable Diffusion, follow these steps:

1. Create a suitable anaconda environment for Deforum and activate it:

```
conda create -n dsd python=3.10 -y
conda activate dsd
```

2. Clone this github repository and navigate to it:
```
git clone https://github.com/deforum-art/deforum-stable-diffusion.git
cd deforum-stable-diffusion
```

3. Install required packages with the install script:

```
python install_requirements.py
```

4. Check your installation by running the Python script:

```
python Deforum_Stable_Diffusion.py
```

## Running Deforum

There are four ways to run Deforum Stable Diffusion: locally with the `.py` file, locally with Jupyter, locally through Colab, and on Colab servers.

### Running Locally

To run Deforum Stable Diffusion locally, make sure the `dsd` conda environment is active:

```
conda activate dsd
```

Then navigate to the `stable-diffusion` folder and run either the `Deforum_Stable_Diffusion.py` or the `Deforum_Stable_Diffusion.ipynb` file. Running the `.py` file is the quickest and easiest way to check that your installation is working, however, it is not the best environment for tinkering with prompts and settings.

```
python Deforum_Stable_Diffusion.py
```

If you prefer a more Colab-like experience, you can run the `.ipynb` file in Jupyter Lab or Jupyter Notebook. To activate Jupyter Lab or Jupyter Notebook from within the `stable-diffusion` folder, use either of the following commands:

```
jupyter-lab
```

or

```
jupyter notebook
```

### Colab Local Runtime

To run Deforum Stable Diffusion using Colab Local Runtime, make sure the `dsd` conda environment is active:

```
conda activate dsd
```

Then, open Google Colab, select `File > Upload notebook`, and choose the `.ipynb` file in the `stable-diffusion` folder. Enable the Jupyter extension by running the following command:

```
jupyter serverextension enable --py jupyter_http_over_ws
```
Start the server by running the following command:
```
jupyter notebook --NotebookApp.allow_origin='https://colab.research.google.com' --port=8888 --NotebookApp.port_retries=0
```
Copy and paste the URL and token provided into the browser to access the Jupyter notebook.


## Starting Over

If you need to start over from scratch, you can delete the `stable-diffusion` folder and remove the `dsd` conda environment with the following set of commands:

```
conda deactivate
conda env remove -n dsd
```

With the `dsd` environment removed, you can start over with a fresh installation.

## Customization

Deforum Stable Diffusion provides a wide range of customization and configuration options that allow you to easily tailor the output to your specific needs and preferences. With over 100 different settings available in the main inference notebook, the possibilities are endless.

For more detailed information on how to customize and configure Deforum Stable Diffusion, check out the [guide](https://docs.google.com/document/d/1RrQv7FntzOuLg4ohjRZPVL7iptIyBhwwbcEYEW2OfcI/edit?usp=sharing) and stay tuned for the upcoming wiki. If you run into any issues while using Deforum Stable Diffusion, here are a few things you can try:

- Make sure you have installed all required dependencies and followed the installation instructions correctly.
- Check the [examples folder](examples/) for guidance.
- Check the most recent [user guide](https://docs.google.com/document/d/1RrQv7FntzOuLg4ohjRZPVL7iptIyBhwwbcEYEW2OfcI/edit?usp=sharing) for troubleshooting tips and solutions.
- If you still can't find a solution, feel free to reach out to the helpful and highly knowledgeable [Deforum Discord](https://discord.gg/deforum)

## Contributing

We welcome contributions to Deforum Stable Diffusion from anyone, regardless of experience level. If you are interested in contributing, please reach out to the developers of the [Deforum Discord](https://discord.gg/deforum) for more information.

All contributions are managed through GitHub, and we maintain branches for work in progress features. We follow the [GitFlow](https://nvie.com/posts/a-successful-git-branching-model/) branching model, with the `dev` branch serving as the main integration branch.

Before submitting a pull request, please make sure to:

- Write clear, concise, and well-documented code.
- Follow the coding style and conventions used in the project.
- Test your changes thoroughly to ensure they work as expected.

## Support Us

Deforum Stable Diffusion is a community-driven, open source project that is free to use and modify. We rely on the support of our users to keep the project going and help us improve it. If you would like to support us, you can make a donation on our [Patreon page](https://patreon.com/deforum). Any amount, big or small, is greatly appreciated!

Your support helps us cover the costs of hosting, development, and maintenance, and allows us to allocate more time and resources to improving Deforum Stable Diffusion. Thank you for your support!

`this readme was written in collaboration with chat-gpt`
