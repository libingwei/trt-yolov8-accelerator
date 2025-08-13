cmake_minimum_required(VERSION 3.18)
project(trt_yolov8_accelerator CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Place executables under the build tree (./build/bin) to align with README usage
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)
# Optionally build experimental plugins (disabled by default)
option(YOLO_BUILD_PLUGINS "Build experimental YOLO TensorRT plugins" OFF)


# CUDA
find_package(CUDAToolkit QUIET)
if(NOT CUDAToolkit_FOUND)
	find_package(CUDA REQUIRED)
	if(CUDA_FOUND)
		add_library(CUDA::cudart UNKNOWN IMPORTED)
		set_property(TARGET CUDA::cudart PROPERTY IMPORTED_LOCATION "${CUDA_CUDART_LIBRARY}")
		set_property(TARGET CUDA::cudart PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${CUDA_INCLUDE_DIRS}")
	else()
		message(FATAL_ERROR "CUDA not found")
	endif()
endif()

# TensorRT from parent project
if(NOT TENSORRT_INCLUDE_DIRS OR NOT TENSORRT_LIBRARY_DIRS)
	message(FATAL_ERROR "Expected TENSORRT_INCLUDE_DIRS/TENSORRT_LIBRARY_DIRS from parent CMake. Please configure at root.")
endif()
include_directories(${TENSORRT_INCLUDE_DIRS})

# OpenCV for image IO/preprocess (INT8 calibrator & optional infer demo)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(onnx_to_trt_yolo
	src/onnx_to_trt_yolo.cpp
	src/yolo_int8_calibrator.cpp)
target_link_libraries(onnx_to_trt_yolo PRIVATE trt_utils ${OpenCV_LIBS})
target_compile_options(onnx_to_trt_yolo PRIVATE -Wno-deprecated-declarations)

add_executable(yolo_trt_infer src/yolo_trt_infer.cpp)
target_link_libraries(yolo_trt_infer PRIVATE trt_utils ${OpenCV_LIBS})
target_compile_options(yolo_trt_infer PRIVATE -Wno-deprecated-declarations)

if(YOLO_BUILD_PLUGINS)
	# Build DecodeYolo plugin
	add_library(decode_yolo_plugin STATIC
		plugins/decode_yolo/decode_yolo_plugin.h
		plugins/decode_yolo/decode_yolo_plugin.cu)
	target_link_libraries(decode_yolo_plugin PRIVATE trt_utils CUDA::cudart)
	target_include_directories(decode_yolo_plugin PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/plugins)
	set_target_properties(decode_yolo_plugin PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
	target_compile_features(decode_yolo_plugin PUBLIC cxx_std_17)
	
	# Build EfficientNMS plugin
	add_library(efficient_nms_plugin STATIC
		plugins/efficient_nms/efficient_nms_plugin.h
		plugins/efficient_nms/efficient_nms_plugin.cu)
	target_link_libraries(efficient_nms_plugin PRIVATE trt_utils CUDA::cudart)
	target_include_directories(efficient_nms_plugin PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/plugins)
	set_target_properties(efficient_nms_plugin PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
	target_compile_features(efficient_nms_plugin PUBLIC cxx_std_17)
	
	# Link plugins into tools and define macro for conditional registration
	target_link_libraries(onnx_to_trt_yolo PRIVATE decode_yolo_plugin efficient_nms_plugin)
	target_link_libraries(yolo_trt_infer PRIVATE decode_yolo_plugin efficient_nms_plugin)
	target_compile_definitions(onnx_to_trt_yolo PRIVATE YOLO_BUILD_PLUGINS)
	target_compile_definitions(yolo_trt_infer PRIVATE YOLO_BUILD_PLUGINS)
endif()

