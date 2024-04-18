FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

ARG USERNAME
ARG USER_UID
ARG USER_GID

ENV HOME /home/$USERNAME
ENV SHELL /bin/bash

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt update \
    && apt upgrade --no-install-recommends -y \
    && apt install --no-install-recommends -y sudo curl wget git tmux \
    && apt install --no-install-recommends -y libglib2.0-0 libsm6 libxrender1 libxext6 libgl1-mesa-dev \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    && apt autoremove -y \
    && apt clean -y \
    && rm -rf /var/lib/apt/lists/*

USER $USERNAME
WORKDIR $HOME

RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    && pip install --upgrade pip \
    && pip install --upgrade ipykernel ipywidgets jupyter \
    && pip install transformers torch datasets numpy albumentations cmake lit pandas pycocotools\
    && pip install accelerate -U 


ENV PATH=/home/$USERNAME/.local/bin:$PATH 
