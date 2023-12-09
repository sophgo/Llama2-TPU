cmake_minimum_required(VERSION 2.8)
project(llama2)

# 交叉编译使用
# set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
# set(CMAKE_ASM_COMPILER aarch64-linux-gnu-gcc)
# set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)
# set(LIBSOPHON /workspace/soc0701)

include_directories(${LIBSOPHON}/include)
include_directories(${PROJECT_SOURCE_DIR}/include)
link_directories(${LIBSOPHON}/lib)
link_directories(${PROJECT_SOURCE_DIR}/lib)

add_definitions(-DDEBUG --std=c++17 -fPIC -Wall -Werror)
set(CMAKE_BUILD_TYPE "Debug")

add_executable(llama2 demo.cpp)
target_link_libraries(llama2 bmrt bmlib sentencepiece)
