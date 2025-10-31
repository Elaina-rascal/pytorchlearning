# 基础镜像：ROS 2 Humble（官方桌面版，基于Ubuntu 22.04）
FROM elainasuki/ros:ros2-humble-full-0614

# 修复未定义变量
ARG USERNAME=Elaina
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# 设置非交互模式
ENV DEBIAN_FRONTEND=noninteractive

# 1. 安装基础依赖（补充apt-utils）
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    curl \
    ca-certificates \
    python3-pip \
    apt-utils \
    && rm -rf /var/lib/apt/lists/*

# 2. 仅安装CUDA运行时（精简版本）
# 注意：运行时依赖nvidia驱动，需确保宿主环境已安装匹配版本
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i cuda-keyring_1.1-1_all.deb \
    && apt-get update \
    && apt-get install -y --no-install-recommends cuda-runtime-12-1 libcudnn8=8.9.2.26-1+cuda12.1 \
    && rm -rf /var/lib/apt/lists/* \
    && rm -f cuda-keyring_1.1-1_all.deb

# 3. 配置运行时环境变量
ENV PATH=/usr/local/cuda-12.1/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:${LD_LIBRARY_PATH:-}

# 4. 安装PyTorch GPU运行时（适配CUDA 12.1）
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 5. 配置ROS 2环境
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc