cmake_minimum_required(VERSION 3.28.3)

project(Assignment02_Task02)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/build/bin)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_subdirectory(${CMAKE_SOURCE_DIR}/../src ${CMAKE_SOURCE_DIR}/build/src)

add_executable(main main.cpp compute.cpp)

target_link_libraries(main PRIVATE library)