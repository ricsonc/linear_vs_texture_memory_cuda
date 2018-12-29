cmake_minimum_required (VERSION 3.0)

project (memorytest LANGUAGES CUDA CXX)

set(CMAKE_CXX_COMPILER "/usr/bin/g++-5")
set(CMAKE_CXX_STANDARD 14)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode=arch=compute_50")

add_executable(memorytest main.cpp main.cu)
