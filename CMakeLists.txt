cmake_minimum_required (VERSION 3.0)

project (memorytest LANGUAGES CUDA CXX)

set(CMAKE_CXX_COMPILER "/usr/bin/g++-5")
set(CMAKE_CXX_STANDARD 14)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3")
set(CMAKE_CUDA_FLAGS "-arch=sm_61 -O3")

add_executable(memorytest main.cpp main.cu)
set_property(TARGET memorytest PROPERTY CUDA_STANDARD 14)

