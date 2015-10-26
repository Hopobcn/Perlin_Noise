cmake_minimum_required(VERSION 2.8)
project(PerlinNoiseDemo CXX)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

add_library(libppm ppm.cpp)
add_library(libPerlinNoise PerlinNoise.cpp)

add_executable(demo-0 PerlinDemo_00.cpp) 
add_executable(demo-1 PerlinDemo_01.cpp)

target_link_libraries(demo-0 libppm libPerlinNoise)
target_link_libraries(demo-1 libppm libPerlinNoise)
