FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

SHELL ["/bin/bash", "-o", "pipefail", "-c"]`

ENV DEBIAN_FRONTEND=noninteractive
ENV SHELL=/bin/bash


RUN apt-get update --yes && \
    apt-get upgrade --yes && \
    apt install --yes --no-install-recommends git wget curl bash libgl1 software-properties-common openssh-server nginx && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt install python3.10-dev python3.10-venv -y --no-install-recommends && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    echo "en_US.UTF-8 UTF-8" > /etc/locale.gen

RUN ln -s /usr/bin/python3.10 /usr/bin/python && \
    rm /usr/bin/python3 && \
    ln -s /usr/bin/python3.10 /usr/bin/python3 && \
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python get-pip.py

RUN pip install --upgrade --no-cache-dir pip
RUN pip install --upgrade --no-cache-dir torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
RUN pip install --upgrade --no-cache-dir jupyterlab ipywidgets jupyter-archive jupyter_contrib_nbextensions

# Set up Jupyter Notebook
# RUN pip install notebook==6.5.5
# RUN jupyter contrib nbextension install --user && \
#     jupyter nbextension enable --py widgetsnbextension

WORKDIR /

COPY . .

RUN python -u /deforum/install_requirements.py

RUN python -m pip install torchmetrics==0.11.4

# NGINX Proxy
# COPY --from=proxy nginx.conf /etc/nginx/nginx.conf
# COPY --from=proxy readme.html /usr/share/nginx/html/readme.html

COPY /proxy/nginx.conf /etc/nginx/nginx.conf
COPY /proxy/readme.html /usr/share/nginx/html/readme.html

RUN chmod +x /start.sh

CMD ["/start.sh"]



# FROM ALEX WARPFUSION
# FROM nvidia/cuda:11.8.0-base-ubuntu22.04

# ENV DEBIAN_FRONTEND=noninteractive

# RUN apt-get update && apt-get install -y software-properties-common \
#     && add-apt-repository ppa:deadsnakes/ppa \
#     && apt-get update \
#     && apt-get install -y \ 
#     git python3.10 python3-pip python3-opencv imagemagick ffmpeg

# RUN ln -s /usr/bin/python3.10 /usr/bin/python \
#     && rm /usr/bin/pip \ 
#     && ln -s /usr/bin/pip3 /usr/bin/pip

# RUN python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
# RUN python -m pip uninstall torchtext -y 
# RUN python -m pip install xformers
# RUN python -m pip install requests mediapipe piexif safetensors lark Pillow==9.0.0 wget webdataset open_clip_torch opencv-python==4.5.5.64 pandas matplotlib fvcore ipywidgets==7.7.1 transformers==4.19.2 omegaconf einops "pytorch_lightning>1.4.1,<=1.7.7" scikit-image opencv-python ai-tools cognitive-face zprint kornia==0.5.0 lpips keras datetime timm==0.6.7 prettytable basicsr fairscale realesrgan torchmetrics==0.11.4
# WORKDIR /content
# RUN git clone https://github.com/Sxela/sxela-stablediffusion /content/stablediffusion && python -m pip install -e /content/stablediffusion
# RUN git clone https://github.com/Sxela/ControlNet-v1-1-nightly /content/ControlNet
# RUN python -m pip install -e git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers -e git+https://github.com/openai/CLIP.git@main#egg=clip
# RUN git clone https://github.com/crowsonkb/guided-diffusion /content/guided-diffusion && pip install -e /content/guided-diffusion
# RUN git clone https://github.com/Sxela/k-diffusion /content/k-diffusion && pip install -e /content/k-diffusion
# RUN git clone https://github.com/assafshocher/ResizeRight.git /content/ResizeRight
# RUN git clone https://github.com/salesforce/BLIP /content/BLIP
# RUN git clone https://github.com/pengbo-learn/python-color-transfer /content/python-color-transfer
# RUN python -m pip install entrypoints==0.4 ipython==8.10.0 jupyter_client==7.4.9 jupyter_core==5.2.0 packaging==22.0 tzdata==2022.7 ipykernel --force-reinstall
# RUN python -m ipykernel install --user && python -m pip install --upgrade jupyter_http_over_ws>=0.0.7

# RUN ln -s /usr/local/lib/python3.10/dist-packages/torch/lib/libnvrtc-672ee683.so.11.2 /usr/local/lib/python3.10/dist-packages/torch/lib/libnvrtc.so 

# ENTRYPOINT ["ls -la && pwd && entrypoint.sh"]