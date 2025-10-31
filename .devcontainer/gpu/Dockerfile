# 基础镜像：ROS 2 Humble（官方桌面版，基于Ubuntu 22.04）
FROM elainasuki/ros:ros2-humble-full-0614

# 修复未定义变量
ARG USERNAME=Elaina
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# 设置非交互模式+pip缓存目录（避免占用根目录空间）
ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_CACHE_DIR=/tmp/pip-cache

# 1. 精简基础依赖（仅保留必需项）+ 清理缓存
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    ca-certificates \
    python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# 2. 仅安装CUDA运行时（极致精简）
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i cuda-keyring_1.1-1_all.deb \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    cuda-runtime-12-1 \
    libcudnn8=8.9.2.26-1+cuda12.1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* \
    && rm -f cuda-keyring_1.1-1_all.deb

# 3. 配置环境变量（修复未定义问题）
ENV PATH=/usr/local/cuda-12.1/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:${LD_LIBRARY_PATH:-/usr/local/lib}

# 4. 优化PyTorch安装（节省空间）
RUN pip3 install --no-cache-dir --upgrade pip \
    && pip3 install --no-cache-dir --ignore-installed sympy \
    && pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 \
    && rm -rf $PIP_CACHE_DIR /tmp/* /var/tmp/*

# 5. 配置ROS 2环境
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc