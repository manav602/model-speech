// LSM6DSOX driver: configure 3.33 kHz accel, drain hardware FIFO into a
// ring buffer on every watermark interrupt.
//
// Public API (start in main, then forget — everything else is interrupt-driven).
#pragma once
#include <stdbool.h>
#include "ring_buffer.h"

// Which accel axis to push into the model. Must match the axis used during
// training (set via training pipeline metadata; see deploy/README.md).
typedef enum {
    IMU_AXIS_X = 0,
    IMU_AXIS_Y = 1,
    IMU_AXIS_Z = 2,
    IMU_AXIS_MAGNITUDE = 3,  // sqrt(x^2+y^2+z^2) - matches training if used
} imu_axis_t;

// Initialize the SPI bus, configure the LSM6DSOX, install the FIFO-watermark
// IRQ, and start streaming into `ring`. Returns 0 on success, negative errno.
int imu_start(struct sample_ring* ring, imu_axis_t axis);

// Stop streaming and put the IMU in power-down. Safe to call from main thread.
void imu_stop(void);

// Diagnostic counters (atomic, may be read at any time).
struct imu_stats {
    uint32_t fifo_overruns;   // sensor reported FIFO_OVR
    uint32_t spi_errors;
    uint32_t samples_pushed;
    uint32_t isr_count;
};
void imu_get_stats(struct imu_stats* out);
