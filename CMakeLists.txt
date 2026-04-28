# Zephyr build for nRF52840 + LSM6DSOX KWS firmware.
#
# Build:
#   west build -b nrf52840dk_nrf52840 deploy/firmware
#   west flash
cmake_minimum_required(VERSION 3.20.0)
find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(kws_firmware C)

target_sources(app PRIVATE
    src/main.c
    src/ble_kws.c
    src/imu_lsm6dsox.c
    src/kws_pipeline.c
    src/dsp.c
    src/kws_int8.c
    model/dscnn_m_w8a8.c
)

target_include_directories(app PRIVATE
    src
    model
)

# CMSIS-DSP + CMSIS-NN are linked automatically via CONFIG_CMSIS_DSP/NN in prj.conf.
target_compile_definitions(app PRIVATE
    USE_CMSIS_NN=1
    ARM_MATH_CM4=1
    __FPU_PRESENT=1
)

# Push the hottest compute kernels to -O3 + loop unrolling. -O2 (from
# CONFIG_SPEED_OPTIMIZATIONS) is the global default; bumping these specific
# files gets us better auto-vectorization in the int8 conv inner loops.
set_source_files_properties(
    src/kws_int8.c
    src/dsp.c
    PROPERTIES COMPILE_OPTIONS "-O3;-funroll-loops"
)
