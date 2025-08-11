cmake_minimum_required(VERSION 3.18)
project(trt_yolov8_accelerator CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

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
# Prefer linking to a prebuilt shared utils library if available in the parent build
if(TARGET trt_utils)
	target_link_libraries(onnx_to_trt_yolo PRIVATE trt_utils ${OpenCV_LIBS})
else()
	# Resolve TRT libs from provided dir
	find_library(NVINFER_LIB nvinfer HINTS ${TENSORRT_LIBRARY_DIRS} REQUIRED)
	find_library(NVONNXPARSER_LIB nvonnxparser HINTS ${TENSORRT_LIBRARY_DIRS} REQUIRED)
	target_link_libraries(onnx_to_trt_yolo PRIVATE CUDA::cudart ${NVINFER_LIB} ${NVONNXPARSER_LIB} ${OpenCV_LIBS})
endif()

add_executable(yolo_trt_infer src/yolo_trt_infer.cpp)
if(TARGET trt_utils)
	target_link_libraries(yolo_trt_infer PRIVATE trt_utils ${OpenCV_LIBS})
else()
	find_library(NVINFER_LIB nvinfer HINTS ${TENSORRT_LIBRARY_DIRS} REQUIRED)
	target_link_libraries(yolo_trt_infer PRIVATE CUDA::cudart ${NVINFER_LIB} ${OpenCV_LIBS})
endif()
