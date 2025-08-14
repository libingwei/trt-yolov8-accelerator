# Minimal dev image for TRT YOLOv8 accelerator with SSH support
# Note: Uses official TensorRT image with CUDA 12.5 (compatible with host CUDA 12.7)
FROM nvcr.io/nvidia/tensorrt:24.07-py3

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# System deps + SSH server + CUDA development tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl vim build-essential cmake libopencv-dev python3-opencv \
    openssh-server sudo && \
    rm -rf /var/lib/apt/lists/*

# Note: TensorRT container already includes CUDA runtime and development tools
# Additional CUDA tools installation if needed can be done here

# 配置SSH服务
RUN mkdir -p /var/run/sshd && \
    sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    sed -i 's/#PasswordAuthentication yes/PasswordAuthentication yes/' /etc/ssh/sshd_config && \
    sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd && \
    ssh-keygen -A

# 设置CUDA环境变量
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 创建工作目录
WORKDIR /workspace

# 添加启动脚本
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# 暴露SSH端口
EXPOSE 22

# 使用自定义入口点启动服务 (密码通过环境变量在运行时设置)
ENTRYPOINT ["/entrypoint.sh"]