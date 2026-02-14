cmake_minimum_required(VERSION 3.28.3)

project(Assignment02_Task02)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/build/bin)

add_subdirectory(${CMAKE_SOURCE_DIR}/../src ${CMAKE_SOURCE_DIR}/build/src)

add_executable(main.o main.cpp compute.cpp)

target_link_libraries(main.o PRIVATE library)