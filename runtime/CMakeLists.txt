cmake_minimum_required(VERSION 3.16)
project(rk_mrt2 VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ═══════════════════════════════════════════════════════
# 依赖
# ═══════════════════════════════════════════════════════

# RKNPU2 Runtime (从 Rockchip SDK)
# 路径需根据实际 SDK 位置调整
option(RKNN_SDK_PATH "Path to Rockchip rknpu2 SDK" "")
if(RKNN_SDK_PATH)
    set(RKNN_INCLUDE_DIR "${RKNN_SDK_PATH}/include")
    set(RKNN_LIB_DIR "${RKNN_SDK_PATH}/lib")
    find_library(RKNN_RT_LIB rknnrt PATHS ${RKNN_LIB_DIR} REQUIRED)
else()
    message(STATUS "RKNN_SDK_PATH not set — building in stub mode")
    set(RKNN_RT_LIB "")
endif()

# Signalsmith DSP (iSTFT)
# 从 GitHub 下载或使用 FetchContent
# include(FetchContent)
# FetchContent_Declare(signalsmith-dsp
#     GIT_REPOSITORY https://github.com/signalsmith-audio/dsp.git
#     GIT_TAG main
# )
# FetchContent_MakeAvailable(signalsmith-dsp)

# ALSA (Linux audio output)
find_package(ALSA QUIET)

# PortAudio (跨平台 audio output, fallback)
find_package(PortAudio QUIET)

# ═══════════════════════════════════════════════════════
# 库: rkmrt2_core
# ═══════════════════════════════════════════════════════

add_library(rkmrt2_core INTERFACE)
target_include_directories(rkmrt2_core INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${RKNN_INCLUDE_DIR}
)
target_compile_features(rkmrt2_core INTERFACE cxx_std_17)
target_compile_options(rkmrt2_core INTERFACE
    -Wall -Wextra -O3 -march=armv8.2-a+dotprod
)

# ═══════════════════════════════════════════════════════
# 示例: rk_mrt2_demo
# ═══════════════════════════════════════════════════════

add_executable(rk_mrt2_demo
    demo/main.cpp
)
target_link_libraries(rk_mrt2_demo PRIVATE
    rkmrt2_core
    ${RKNN_RT_LIB}
)
if(ALSA_FOUND)
    target_link_libraries(rk_mrt2_demo PRIVATE ${ALSA_LIBRARIES})
    target_compile_definitions(rk_mrt2_demo PRIVATE HAS_ALSA)
endif()
if(PortAudio_FOUND)
    target_link_libraries(rk_mrt2_demo PRIVATE ${PortAudio_LIBRARIES})
    target_compile_definitions(rk_mrt2_demo PRIVATE HAS_PORTAUDIO)
endif()

# ═══════════════════════════════════════════════════════
# 测试
# ═══════════════════════════════════════════════════════

option(BUILD_TESTS "Build tests" ON)
if(BUILD_TESTS)
    enable_testing()
    add_executable(test_kv_cache
        tests/test_kv_cache.cpp
    )
    target_link_libraries(test_kv_cache PRIVATE rkmrt2_core)
    add_test(NAME kv_cache COMMAND test_kv_cache)
endif()

# ═══════════════════════════════════════════════════════
# 安装
# ═══════════════════════════════════════════════════════

install(TARGETS rk_mrt2_demo
    RUNTIME DESTINATION bin
)
install(DIRECTORY ./
    DESTINATION include/rkmrt2
    FILES_MATCHING PATTERN "*.hpp"
)
