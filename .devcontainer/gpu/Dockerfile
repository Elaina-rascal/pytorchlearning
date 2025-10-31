# 基础镜像：ROS 2 Humble（官方桌面版，基于Ubuntu 22.04）
FROM elainasuki/ros:ros2-humble-full-0614

ARG USERNAME=Elaina

ARG USER_GID=$USER_UID
# 设置非交互模式，避免apt弹窗
ENV DEBIAN_FRONTEND=noninteractive

# 1. 安装基础依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    ca-certificates \
    libjpeg-dev \
    libpng-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# 2. 安装CUDA 12.1（适配Ubuntu 22.04）
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin \
    && mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600 \
    && wget https://developer.download.nvidia.com/compute/cuda/12.1.1/local_installers/cuda-repo-ubuntu2204-12-1-local_12.1.1-530.30.02-1_amd64.deb \
    && dpkg -i cuda-repo-ubuntu2204-12-1-local_12.1.1-530.30.02-1_amd64.deb \
    && cp /var/cuda-repo-ubuntu2204-12-1-local/cuda-*-keyring.gpg /usr/share/keyrings/ \
    && apt-get update \
    && apt-get -y install cuda-12-1 \
    && rm -rf /var/lib/apt/lists/*

# 3. 配置CUDA环境变量
ENV PATH=/usr/local/cuda-12.1/bin${PATH:+:${PATH}}
ENV LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

# 4. 安装cuDNN 8.9（匹配CUDA 12.1）
RUN apt-get update && apt-get install -y --no-install-recommends \
    libcudnn8=8.9.2.26-1+cuda12.1 \
    libcudnn8-dev=8.9.2.26-1+cuda12.1 \
    && rm -rf /var/lib/apt/lists/*

# 5. 安装PyTorch GPU版（适配CUDA 12.1）
# 参考PyTorch官网：https://pytorch.org/get-started/locally/
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 6. 验证PyTorch GPU可用性（可选）
# RUN python3 -c "import torch; print('PyTorch版本:', torch.__version__); print('CUDA是否可用:', torch.cuda.is_available())"
# 7. 配置ROS 2环境（自动加载）
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc