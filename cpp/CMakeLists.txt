cmake_minimum_required(VERSION 3.10)

project(smoke_app LANGUAGES C CXX)

# Option to enable Address Sanitizer (ASAN)
option(ENABLE_ASAN "Enable Address Sanitizer" OFF)

if (ENABLE_ASAN)
    message(STATUS "Building with Address Sanitizer enabled")
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
    set(CMAKE_EXE_LINKER_FLAGS_DEBUG "${CMAKE_EXE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
endif()

# Add RV1106-specific definitions
if (TARGET_SOC STREQUAL "rv1106")
    add_definitions(-DRV1106_1103)
endif()

# Sysroot and library linker flags
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -L/{PATH}/rknn-toolkit2/rknpu2/examples/rv1006/arm-rockchip830-linux-uclibcgnueabihf/arm-rockchip830-linux-uclibcgnueabihf/sysroot/lib -L/{PATH}/rknn_model_zoo/3rdparty/rknpu2/Linux/armhf-uclibc -Wl,-rpath-link,/{PATH}rknn-toolkit2/rknpu2/examples/rv1006/arm-rockchip830-linux-uclibcgnueabihf/arm-rockchip830-linux-uclibcgnueabihf/sysroot/lib")

# Include directories
include_directories(
    /home/luckfox/rknn_model_zoo/3rdparty/rknpu2/include
    /home/luckfox/rknn_toolkit2/rknpu2/examples/3rdparty/rga/include
    /home/luckfox/rknn_toolkit2/rknpu2/examples/3rdparty/stb
    /home/luckfox/rknn_model_zoo/3rdparty/jpeg_turbo/include
    /home/rknn_toolkit2/rknpu2/runtime/Linux/librknn_api/include/
    /home/luckfox/rknn_model_zoo/utils
    /home/luckfox/Downloads/luckfox_pico_rknn_example/utils
)

# Link directories
link_directories(
    /home/luckfox/rknn_model_zoo/3rdparty/rknpu2/Linux/armhf-uclibc
    /home/luckfox/rknn_model_zoo/3rdparty/librga/Linux/armhf_uclibc
    /home/luckfox/rknn_model_zoo/3rdparty/jpeg_turbo/Linux/armhf_uclibc
)

# Source files
set(SOURCES
    main.cc
    smoke_model.cc
    postprocess.cc
    /home/luckfox/Downloads/luckfox_pico_rknn_example/utils/image_utils.c
    /home/luckfox/Downloads/luckfox_pico_rknn_example/utils/file_utils.c
    /home/luckfox/Downloads/luckfox_pico_rknn_example/utils/image_drawing.c
)

# Add executable
add_executable(${PROJECT_NAME} ${SOURCES})

# Link libraries
target_link_libraries(${PROJECT_NAME}
    /home/luckfox/rknn_model_zoo/3rdparty/rknpu2/Linux/armhf-uclibc/librknnrt.so
    /home/luckfox/rknn_model_zoo/3rdparty/librga/Linux/armhf_uclibc/librga.a
    /home/luckfox/rknn_model_zoo/3rdparty/jpeg_turbo/Linux/armhf_uclibc/libturbojpeg.a
    pthread
    dl
    m
)

# Installation
install(TARGETS ${PROJECT_NAME} DESTINATION .)
