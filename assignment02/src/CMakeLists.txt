set(SRCS
    file.cpp
    utils.cpp
    matrix.cpp
)

add_library(library ${SRCS})

target_include_directories(library PUBLIC ${CMAKE_SOURCE_DIR}/../include)