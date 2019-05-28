
FROM nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04
MAINTAINER Gautham Nandakumar<gautham@bostonmeditech.com>

ARG username=user
ARG password=1234


RUN apt-get update && \
    apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    unzip \
    yasm \
    pkg-config \
    curl 

RUN apt-get install -y \
    libswscale-dev libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev \
    libjasper-dev libavformat-dev libpq-dev libxine2-dev libglew-dev \
    libtiff5-dev zlib1g-dev libjpeg-dev libpng12-dev libjasper-dev \
    libavcodec-dev libavformat-dev libavutil-dev libpostproc-dev \
	libswscale-dev libeigen3-dev libtbb-dev libgtk2.0-dev 
    # libcudnn7=7.1.4.18-1+cuda9.0 

RUN apt-get install -y \
    python3-dev \
    python3-numpy \
    python3-pip 

RUN pip3 install --upgrade pip

## Cleanup
RUN rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip3 install -r requirements.txt

# Set the library path to use cuda and cupti
ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/lib64:$LD_LIBRARY_PATH

WORKDIR /home

RUN git clone --single-branch --branch master https://${username}:${password}@github.com/BMG-Jane/Gmodel.git /home
