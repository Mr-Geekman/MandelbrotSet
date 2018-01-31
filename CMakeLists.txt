cmake_minimum_required(VERSION 3.8)
project(Mandelbrot_set)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES main.cpp include/CUDA_wrappers.hpp src/CUDA.cu)

# LIBGD Package
find_package(GD REQUIRED)
include_directories(${GD_INCLUDE_DIR})

# CUDA Package
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# Compile CU files
#file(GLOB CUDA_FILES "src/" *.cu)
list( APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_20,code=sm_20; -std=c++11")
cuda_compile(CudaKernel src/CUDA.cu)
cuda_add_library(CudaKernels src/CUDA.cu)

# COMPILE AND LINK
cuda_add_executable(Mandelbrot_set $(SOURCE_FILES) $(CudaKernels))
target_link_libraries(Mandelbrot_set ${GD_LIBRARIES} ${CUDA_LIBRARIES})
