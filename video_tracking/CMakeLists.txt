cmake_minimum_required(VERSION 3.18)
project(video_tracking CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置输出目录 - 使用顶层build的bin目录
if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    # 独立构建：使用本地bin目录
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)
else()
    # 作为子项目构建：使用顶层bin目录
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
endif()

# 包含父项目的TensorRT设置
if(NOT TENSORRT_INCLUDE_DIRS OR NOT TENSORRT_LIBRARY_DIRS)
    message(FATAL_ERROR "Expected TENSORRT_INCLUDE_DIRS/TENSORRT_LIBRARY_DIRS from parent CMake.")
endif()
include_directories(${TENSORRT_INCLUDE_DIRS})

# 查找OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# 包含头文件目录
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# 创建库
add_library(video_tracking_lib
    src/object_tracker.cpp
    src/zone_counter.cpp
    src/video_processor.cpp
)

target_link_libraries(video_tracking_lib 
    PUBLIC 
    trt_utils 
    ${OpenCV_LIBS}
)

target_include_directories(video_tracking_lib 
    PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# 创建可执行文件
add_executable(video_tracker main.cpp)

target_link_libraries(video_tracker 
    PRIVATE 
    video_tracking_lib
)

target_compile_options(video_tracker PRIVATE -Wno-deprecated-declarations)
target_compile_options(video_tracking_lib PRIVATE -Wno-deprecated-declarations)
