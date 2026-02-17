cmake_minimum_required(VERSION 3.22.1)

project(Assignment02_Task03)

if (NOT CMAKE_CUDA_COMPILER)
    set(CMAKE_CUDA_COMPILER "nvcc")
endif()

project(${PROJECT_NAME} LANGUAGES CXX CUDA)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/build/bin)

add_subdirectory(${CMAKE_SOURCE_DIR}/../src ${CMAKE_SOURCE_DIR}/build/src)

add_executable(main main.cu compute.cu)

target_link_libraries(main PRIVATE library)