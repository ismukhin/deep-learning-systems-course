FROM nvidia/cuda:13.2.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    pkg-config \
    wget \
    unzip \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libopenexr-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libgtk-3-dev \
    libatlas-base-dev \
    gfortran \
    python3-dev \
    python3-numpy \
    python3-pip \
    libcanberra-gtk3-module \
    && rm -rf /var/lib/apt/lists/*

ARG OPENCV_VERSION=4.10.0
RUN git clone --branch ${OPENCV_VERSION} --depth 1 https://github.com/opencv/opencv.git /opencv && \
    git clone --branch ${OPENCV_VERSION} --depth 1 https://github.com/opencv/opencv_contrib.git /opencv_contrib

WORKDIR /opencv/build
RUN cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D BUILD_TESTS=OFF \
    -D BUILD_PERF_TESTS=OFF \
    -D WITH_CUDA=ON \
    -D WITH_CUDNN=ON \
    -D OPENCV_DNN_CUDA=OFF \
    -D ENABLE_FAST_MATH=1 \
    -D CUDA_FAST_MATH=1 \
    -D WITH_CUBLAS=1 \
    -D OPENCV_EXTRA_MODULES_PATH=/opencv_contrib/modules \
    -D BUILD_EXAMPLES=OFF \
    -D BUILD_opencv_python3=ON \
    -D PYTHON3_EXECUTABLE=$(which python3) \
    -D PYTHON3_INCLUDE_DIR=$(python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
    -D PYTHON3_PACKAGES_PATH=$(python3 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") \
    .. && \
    make -j$(nproc) && \
    make install && \
    ldconfig


ARG ORT_URL="https://github.com/microsoft/onnxruntime/releases/download/v1.24.3/onnxruntime-linux-x64-gpu_cuda13-1.24.3.tgz"
RUN wget --quiet ${ORT_URL} -O /tmp/ort.tgz && \
    mkdir -p /opt/onnxruntime && \
    tar -xzf /tmp/ort.tgz -C /opt/onnxruntime --strip-components=1 && \
    rm /tmp/ort.tgz

ENV LD_LIBRARY_PATH=/opt/onnxruntime/lib:${LD_LIBRARY_PATH}

RUN rm -rf /opencv /opencv_contrib && \
    apt-get autoremove -y && \
    apt-get clean

ENV PATH=/usr/local/cuda/bin:$PATH
CMD ["/bin/bash"]