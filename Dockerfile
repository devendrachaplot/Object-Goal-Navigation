# Base image
FROM nvidia/cudagl:10.1-devel-ubuntu16.04

# Setup basic packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    vim \
    ca-certificates \
    libjpeg-dev \
    libpng-dev \
    libglfw3-dev \
    libglm-dev \
    libx11-dev \
    libomp-dev \
    libegl1-mesa-dev \
    pkg-config \
    wget \
    zip \
    htop \
    tmux \
    unzip &&\
    rm -rf /var/lib/apt/lists/*

# Install conda
RUN wget -O $HOME/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  &&\
    chmod +x ~/miniconda.sh &&\
    ~/miniconda.sh -b -p /custom/conda &&\
    rm ~/miniconda.sh &&\
    /custom/conda/bin/conda install numpy pyyaml scipy ipython mkl mkl-include &&\
    /custom/conda/bin/conda clean -ya
ENV PATH /custom/conda/bin:$PATH

# Install cmake
RUN wget https://github.com/Kitware/CMake/releases/download/v3.14.0/cmake-3.14.0-Linux-x86_64.sh
RUN mkdir /opt/cmake
RUN sh /cmake-3.14.0-Linux-x86_64.sh --prefix=/opt/cmake --skip-license
RUN ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake
RUN cmake --version

# Setup habitat-sim
RUN git clone https://github.com/facebookresearch/habitat-sim.git
RUN /bin/bash -c "cd habitat-sim; git checkout tags/v0.1.5; pip install -r requirements.txt; python setup.py install --headless --with-cuda"

# Install challenge specific habitat-api
RUN git clone https://github.com/facebookresearch/habitat-api.git
RUN /bin/bash -c "cd habitat-api; git checkout tags/v0.1.5; pip install -e ."
RUN /bin/bash -c "cd habitat-api; wget http://dl.fbaipublicfiles.com/habitat/habitat-test-scenes.zip; unzip habitat-test-scenes.zip"

# Silence habitat-sim logs
ENV GLOG_minloglevel=2
ENV MAGNUM_LOG="quiet"

# Install project specific packages
RUN /bin/bash -c "apt-get update; apt-get install -y libsm6 libxext6 libxrender-dev; pip install opencv-python"
RUN /bin/bash -c "pip install --upgrade cython numpy"
RUN /bin/bash -c "pip install matplotlib seaborn==0.9.0 scikit-fmm==2019.1.30 scikit-image==0.15.0 imageio==2.6.0 scikit-learn==0.22.2.post1 ifcfg"

# Install pytorch and torch_scatter
RUN conda install pytorch=1.6.0 torchvision=0.7.0 cudatoolkit=10.2 -c pytorch
RUN /bin/bash -c "pip install torch_scatter"

# Install detectron2
RUN /bin/bash -c "python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.6/index.html"
