#!/bin/bash

# Docker 构建和运行脚本 - TensorRT YOLOv8 开发环境

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_NAME="trt-yolov8-dev"
CONTAINER_NAME="trt-yolov8-container"

# 函数：显示帮助信息
show_help() {
    cat << EOF
TensorRT YOLOv8 Docker 开发环境管理脚本

用法: $0 [命令] [选项]

命令:
  build         构建Docker镜像
  run           运行新容器
  start         启动已存在的容器
  stop          停止容器
  shell         进入容器shell
  logs          查看容器日志
  clean         清理容器和镜像
  ssh-info      显示SSH连接信息
  verify        验证TensorRT版本兼容性

选项:
  -p, --password PASSWORD   设置root密码 (默认: dockerdev)
  -h, --help               显示此帮助信息

示例:
  $0 verify                # 验证TensorRT版本
  $0 build                 # 构建镜像
  $0 run -p mypassword     # 运行容器并设置密码
  $0 shell                 # 进入运行中的容器
  $0 ssh-info              # 获取SSH连接信息

EOF
}

# 函数：构建镜像
build_image() {
    echo "🔨 构建Docker镜像..."
    cd "${PROJECT_ROOT}/projects/trt-yolov8-accelerator/docker"
    docker build -t "${IMAGE_NAME}:latest" .
    echo "✅ 镜像构建完成: ${IMAGE_NAME}:latest"
}

# 函数：运行容器
run_container() {
    local password="${1:-dockerdev}"
    
    # 检查容器是否已存在
    if docker ps -a --format 'table {{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo "⚠️  容器 ${CONTAINER_NAME} 已存在"
        echo "使用 '$0 start' 启动现有容器，或 '$0 clean' 清理后重新运行"
        return 1
    fi
    
    echo "🚀 启动新容器..."
    docker run -d \
        --name "${CONTAINER_NAME}" \
        --gpus all \
        -p 2222:22 \
        -e ROOT_PASSWORD="${password}" \
        -v "${PROJECT_ROOT}:/workspace" \
        "${IMAGE_NAME}:latest"
    
    echo "✅ 容器启动成功: ${CONTAINER_NAME}"
    echo "📁 项目目录已挂载到: /workspace"
    sleep 2
    show_ssh_info
}

# 函数：启动容器
start_container() {
    echo "▶️  启动容器 ${CONTAINER_NAME}..."
    docker start "${CONTAINER_NAME}"
    echo "✅ 容器已启动"
    sleep 2
    show_ssh_info
}

# 函数：停止容器
stop_container() {
    echo "⏹️  停止容器 ${CONTAINER_NAME}..."
    docker stop "${CONTAINER_NAME}" || true
    echo "✅ 容器已停止"
}

# 函数：进入容器shell
enter_shell() {
    echo "🖥️  进入容器shell..."
    docker exec -it "${CONTAINER_NAME}" /bin/bash
}

# 函数：查看日志
show_logs() {
    echo "📋 容器日志:"
    docker logs "${CONTAINER_NAME}"
}

# 函数：清理
clean_up() {
    echo "🧹 清理Docker资源..."
    docker stop "${CONTAINER_NAME}" 2>/dev/null || true
    docker rm "${CONTAINER_NAME}" 2>/dev/null || true
    echo "✅ 容器已清理"
    
    read -p "是否同时删除镜像? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        docker rmi "${IMAGE_NAME}:latest" 2>/dev/null || true
        echo "✅ 镜像已删除"
    fi
}

# 函数：验证TensorRT版本
verify_versions() {
    echo "🔍 启动TensorRT版本验证..."
    local verify_script="${PROJECT_ROOT}/projects/trt-yolov8-accelerator/docker/verify-tensorrt-versions.sh"
    
    if [ ! -f "$verify_script" ]; then
        echo "❌ 验证脚本未找到: $verify_script"
        return 1
    fi
    
    chmod +x "$verify_script"
    "$verify_script" recommend
}
show_ssh_info() {
    if ! docker ps --format 'table {{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo "❌ 容器未运行"
        return 1
    fi
    
    local container_ip=$(docker inspect -f '{{range.NetworkSettings.Networks}}{{.IPAddress}}{{end}}' "${CONTAINER_NAME}")
    
    echo "🔗 SSH连接信息:"
    echo "   方式1 (端口映射): ssh root@localhost -p 2222"
    echo "   方式2 (容器IP):   ssh root@${container_ip}"
    echo "   默认密码: dockerdev (可通过ROOT_PASSWORD环境变量修改)"
    echo ""
    echo "💡 测试连接:"
    echo "   ssh root@localhost -p 2222 'nvcc --version'"
}

# 函数：检查Docker环境
check_docker() {
    if ! command -v docker &> /dev/null; then
        echo "❌ Docker未安装或不在PATH中"
        exit 1
    fi
    
    if ! docker ps &> /dev/null; then
        echo "❌ Docker服务未运行或权限不足"
        exit 1
    fi
    
    # 检查NVIDIA Docker支持
    if ! docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi &> /dev/null; then
        echo "⚠️  警告: NVIDIA Docker支持可能未正确配置"
        echo "   请确保安装了nvidia-container-toolkit"
    fi
}

# 主逻辑
main() {
    local command="${1:-help}"
    local password="dockerdev"
    
    # 解析参数
    shift || true
    while [[ $# -gt 0 ]]; do
        case $1 in
            -p|--password)
                password="$2"
                shift 2
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                echo "未知选项: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # 检查Docker环境
    check_docker
    
    # 执行命令
    case $command in
        build)
            build_image
            ;;
        run)
            run_container "$password"
            ;;
        start)
            start_container
            ;;
        stop)
            stop_container
            ;;
        shell)
            enter_shell
            ;;
        logs)
            show_logs
            ;;
        clean)
            clean_up
            ;;
        ssh-info)
            show_ssh_info
            ;;
        verify)
            verify_versions
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            echo "❌ 未知命令: $command"
            show_help
            exit 1
            ;;
    esac
}

# 运行主函数
main "$@"
