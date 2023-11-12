cmake_minimum_required(VERSION 2.8)
project(llama2)

set(LIBSOPHON /workspace/libsophon/install/libsophon-0.4.9)
include_directories(${LIBSOPHON}/include)
include_directories(${PROJECT_SOURCE_DIR}/include)
link_directories(${LIBSOPHON}/lib)
link_directories(${PROJECT_SOURCE_DIR}/lib)

add_definitions(-DDEBUG --std=c++17 -fPIC -Wall -Werror)
set(CMAKE_BUILD_TYPE "Debug")

add_executable(llama2 demo.cpp)
target_link_libraries(llama2 bmrt bmlib sentencepiece)
