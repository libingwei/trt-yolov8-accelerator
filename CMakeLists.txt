cmake_minimum_required(VERSION 3.18)
project(trt_yolov8_accelerator CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

# 占位可执行，后续实现
add_executable(onnx_to_trt_yolo src/onnx_to_trt_yolo.cpp)
add_executable(yolo_trt_infer src/yolo_trt_infer.cpp)
