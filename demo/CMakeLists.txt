cmake_minimum_required(VERSION 2.8)
project(llama2)


set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE INTERNAL "") 

if (NOT DEFINED TARGET_ARCH)
    set(TARGET_ARCH pcie)
endif()

include_directories(${PROJECT_SOURCE_DIR}/../src/include)

if (${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL "aarch64")
	add_definitions(-DSOC_TARGET)
	link_directories(${PROJECT_SOURCE_DIR}/../src/lib_soc)
	message("SoC mode, starting......")
elseif (${TARGET_ARCH} STREQUAL "pcie")
    add_definitions(-DPCIE_TARGET)
    link_directories(${PROJECT_SOURCE_DIR}/../src/lib_pcie)
	message("PCIE mode, starting......")
elseif (${TARGET_ARCH} STREQUAL "soc")
    add_definitions(-DSOC_TARGET)
    set(CMAKE_C_COMPILER /opt/aarch64-linux-gnu-7.5.0/bin/aarch64-linux-gnu-gcc)
    set(CMAKE_ASM_COMPILER /opt/aarch64-linux-gnu-7.5.0/bin/aarch64-linux-gnu-gcc)
    set(CMAKE_CXX_COMPILER /opt/aarch64-linux-gnu-7.5.0/bin/aarch64-linux-gnu-g++)
    link_directories(${PROJECT_SOURCE_DIR}/lib_soc)
	message("SoC mode, starting......")
endif()

set(SOURCE_FILE demo.cpp)
set(EXECUTABLE_NAME llama2)

add_definitions(-DNDEBUG --std=c++17 -fPIC -Wall)
set(CMAKE_BUILD_TYPE "Release")

add_executable(${EXECUTABLE_NAME} ${SOURCE_FILE})
target_link_libraries(${EXECUTABLE_NAME} bmlib bmrt sentencepiece)