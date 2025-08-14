#!/bin/bash

# 设置root密码（如果环境变量存在）
if [ -n "$ROOT_PASSWORD" ]; then
    echo "Setting root password from environment variable"
    echo "root:$ROOT_PASSWORD" | chpasswd
else
    echo "WARNING: ROOT_PASSWORD environment variable not set!"
    echo "Using default password - please set ROOT_PASSWORD for security"
    echo "root:bingwell" | chpasswd
fi

# 生成SSH主机密钥（如果不存在）
if [ ! -f /etc/ssh/ssh_host_rsa_key ]; then
    echo "Generating SSH host keys..."
    ssh-keygen -A
fi

# 验证CUDA环境
echo "Verifying CUDA environment..."
if command -v nvcc &> /dev/null; then
    echo "NVCC version:"
    nvcc --version
else
    echo "WARNING: nvcc not found in PATH"
fi

# 启动SSH服务
echo "Starting SSH server..."
/usr/sbin/sshd -D &

# 显示连接信息
echo "=================================="
echo "Docker development environment ready!"
echo "SSH server is running on port 22"
echo "Default user: root"
echo "Default password: $ROOT_PASSWORD"
echo "Connect with: ssh root@<container_ip>"
echo "Or with port mapping: ssh root@localhost -p <mapped_port>"
echo "=================================="

# 保持容器运行（可根据需要添加其他服务启动命令）
echo "Container is running... (Ctrl+C to stop)"
tail -f /dev/null
