# Toolchain file for Rockchip RV1106 (arm-linux-uclibcgnueabihf)

SET(CMAKE_SYSTEM_NAME Linux)
SET(CMAKE_SYSTEM_PROCESSOR arm)

# Path to the toolchain binaries
SET(TOOLCHAIN_PREFIX arm-rockchip830-linux-uclibcgnueabihf)
SET(TOOLCHAIN_PATH /home/luckfox/rknn_model_zoo/examples/yolox/python/rknn-toolkit2/rknpu2/examples/rv1006/arm-rockchip830-linux-uclibcgnueabihf/bin)

# Compilers
SET(CMAKE_C_COMPILER ${TOOLCHAIN_PATH}/${TOOLCHAIN_PREFIX}-gcc)
SET(CMAKE_CXX_COMPILER ${TOOLCHAIN_PATH}/${TOOLCHAIN_PREFIX}-g++)

# Optional: strip, ar, etc.
SET(CMAKE_STRIP ${TOOLCHAIN_PATH}/${TOOLCHAIN_PREFIX}-strip)
SET(CMAKE_AR ${TOOLCHAIN_PATH}/${TOOLCHAIN_PREFIX}-ar)
SET(CMAKE_LINKER ${TOOLCHAIN_PATH}/${TOOLCHAIN_PREFIX}-ld)

# Don’t use host paths
SET(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
SET(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
SET(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
