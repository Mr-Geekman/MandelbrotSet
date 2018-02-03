cmake_minimum_required(VERSION 3.8)
project(Mandelbrot_set)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES main.cpp include/CUDA_wrappers.hpp src/CUDA.cu)

# LibGD Package
find_package(GD REQUIRED)
include_directories(${GD_INCLUDE_DIR})

# CUDA Package
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# OpenCV Package
find_package(OpenCV REQUIRED)
include_directories($(OpenCV_INCLUDE_DIRS))

# Compile CU files
list( APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_30,code=sm_30; -std=c++11")
cuda_add_library(CudaKernels src/CUDA.cu)

# Compile and link
cuda_add_executable(Mandelbrot_set ${SOURCE_FILES})
target_link_libraries(Mandelbrot_set CudaKernels ${GD_LIBRARIES} ${CUDA_LIBRARIES} ${OpenCV_LIBS})
