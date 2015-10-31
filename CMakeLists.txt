cmake_minimum_required(VERSION 3.0)
project(PerlinNoiseDemo C CXX)

find_package(Threads)
find_package(CUDA REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -std=c++11")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -g -G -gencode arch=compute_35,code=sm_35 -gencode arch=compute_52,code=sm_52")
set(CUDA_HOST_COMPILER "/usr/bin/g++-4.9")

include_directories(${CUDA_INCLUDE_DIRS})

add_library(libppm ppm.cpp)

cuda_add_executable(demo-0 PerlinDemo_00.cpp) 
cuda_add_executable(demo-1 PerlinDemo_01.cpp)
cuda_add_executable(demo-2 PerlinDemo_02.cu)
cuda_add_executable(demo-3 PerlinDemo_03.cu)

target_link_libraries(demo-0 libppm)
target_link_libraries(demo-1 libppm)
target_link_libraries(demo-2 libppm)
target_link_libraries(demo-3 libppm)
