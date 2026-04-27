// LSM6DSOX driver — accel-only, 3.33 kHz ODR, hardware FIFO watermark IRQ.
// I2C transport (not SPI).
//
// Design notes:
//   * The model was trained on 3333 Hz int16 Z-axis accel samples.
//   * FIFO watermark = 64 samples (~19 ms). At 400 kHz I2C, draining
//     64 frames × 7 B takes ~11 ms — fits within the next watermark.
//   * INT1 is active-low (H_LACTIVE=1 in CTRL3_C) to match the overlay flag.
//   * FIFO drain runs in a dedicated thread — I2C TWIM uses DMA+semaphores
//     and cannot be called from a GPIO ISR.
//
// Wiring (boards/nrf52840dk_nrf52840.overlay):
//   I2C0 → LSM6DSOX  (INT1 = imu-int1 alias, active-low)
#include "imu_lsm6dsox.h"

#include <zephyr/kernel.h>
#include <zephyr/drivers/i2c.h>
#include <zephyr/drivers/gpio.h>
#include <zephyr/sys/atomic.h>
#include <zephyr/logging/log.h>
#include <errno.h>
#include <math.h>
#include <string.h>

LOG_MODULE_REGISTER(imu, LOG_LEVEL_INF);

// LSM6DSOX register map (subset)
#define REG_WHO_AM_I          0x0F
#define REG_FIFO_CTRL1        0x07
#define REG_FIFO_CTRL2        0x08
#define REG_FIFO_CTRL3        0x09
#define REG_FIFO_CTRL4        0x0A
#define REG_INT1_CTRL         0x0D
#define REG_CTRL1_XL          0x10
#define REG_CTRL3_C           0x12
#define REG_CTRL6_C           0x15
#define REG_FIFO_STATUS1      0x3A
#define REG_FIFO_STATUS2      0x3B
#define REG_FIFO_DATA_OUT_TAG 0x78

#define WHO_AM_I_VALUE    0x6C
#define ODR_3K33_FS_4G    0x98  // ODR=3333 Hz, FS=±4 g
#define FIFO_WTM_LEVEL    64
#define FIFO_BDR_XL_3K33  0x0A  // 1010 = 3333.3 Hz (Table 54)
#define FIFO_MODE_BYPASS  0x00
#define FIFO_MODE_CONT    0x06

#define I2C_NODE  DT_ALIAS(imu_i2c)
#define INT1_NODE DT_ALIAS(imu_int1)

static const struct i2c_dt_spec imu_i2c  = I2C_DT_SPEC_GET(I2C_NODE);
static const struct gpio_dt_spec int1_pin = GPIO_DT_SPEC_GET(INT1_NODE, gpios);
static struct gpio_callback int1_cb;

static struct sample_ring *g_ring;
static imu_axis_t          g_axis;
static struct imu_stats    g_stats;
static atomic_t            g_running;

// Semaphore: ISR gives, drain thread takes.
static K_SEM_DEFINE(g_fifo_sem, 0, 1);

// ── low-level I2C (thread context only) ──
static int reg_write(uint8_t reg, uint8_t val) {
    return i2c_reg_write_byte_dt(&imu_i2c, reg, val);
}

static int reg_read(uint8_t reg, uint8_t *dst, size_t n) {
    return i2c_burst_read_dt(&imu_i2c, reg, dst, n);
}

// ── ISR — must not call I2C ──
static void on_int1(const struct device *dev, struct gpio_callback *cb,
                    uint32_t pins) {
    ARG_UNUSED(dev); ARG_UNUSED(cb); ARG_UNUSED(pins);
    g_stats.isr_count++;
    k_sem_give(&g_fifo_sem);
    /* Log first hit and every 500th to confirm IRQ is firing without flooding RTT. */
    if (g_stats.isr_count == 1 || (g_stats.isr_count % 500) == 0) {
        LOG_INF("INT1 fired #%u", g_stats.isr_count);
    }
}

// ── FIFO drain (runs in imu_drain thread) ──
static void drain_fifo(void) {
    uint8_t st[2];
    if (reg_read(REG_FIFO_STATUS1, st, 2) < 0) {
        g_stats.spi_errors++;
        return;
    }
    uint16_t fifo_level = ((uint16_t)(st[1] & 0x03) << 8) | st[0];
    bool overrun = (st[1] & 0x40) != 0;
    if (overrun) g_stats.fifo_overruns++;
    if (fifo_level == 0) return;
    if (fifo_level > FIFO_WTM_LEVEL * 2) fifo_level = FIFO_WTM_LEVEL * 2;

    static uint8_t frames[FIFO_WTM_LEVEL * 2 * 7];
    if (reg_read(REG_FIFO_DATA_OUT_TAG, frames, fifo_level * 7) < 0) {
        g_stats.spi_errors++;
        return;
    }

    for (uint16_t i = 0; i < fifo_level; ++i) {
        const uint8_t *f = &frames[i * 7];
        if (((f[0] >> 3) & 0x1F) != 0x02) continue;  // accel tag only
        int16_t x = (int16_t)((f[2] << 8) | f[1]);
        int16_t y = (int16_t)((f[4] << 8) | f[3]);
        int16_t z = (int16_t)((f[6] << 8) | f[5]);

        int16_t s;
        switch (g_axis) {
        case IMU_AXIS_X: s = x; break;
        case IMU_AXIS_Y: s = y; break;
        case IMU_AXIS_Z: s = z; break;
        case IMU_AXIS_MAGNITUDE:
        default: {
            int32_t sx = x, sy = y, sz = z;
            uint32_t m = (uint32_t)sqrtf((float)(sx*sx + sy*sy + sz*sz));
            s = (int16_t)(m > 32767 ? 32767 : m);
            break;
        }
        }
        ring_push(g_ring, s);
    }
    g_stats.samples_pushed += fifo_level;
}

// ── drain thread ──
#define IMU_DRAIN_STACK 1024
K_THREAD_STACK_DEFINE(g_drain_stack, IMU_DRAIN_STACK);
static struct k_thread g_drain_thread;

static void drain_thread_fn(void *a, void *b, void *c) {
    ARG_UNUSED(a); ARG_UNUSED(b); ARG_UNUSED(c);
    while (atomic_get(&g_running)) {
        k_sem_take(&g_fifo_sem, K_FOREVER);
        if (atomic_get(&g_running)) drain_fifo();
    }
}

// ── public API ──
int imu_start(struct sample_ring *ring, imu_axis_t axis) {
    if (!device_is_ready(imu_i2c.bus)) {
        LOG_ERR("I2C bus %s not ready", imu_i2c.bus->name);
        return -ENODEV;
    }
    if (!gpio_is_ready_dt(&int1_pin)) {
        LOG_ERR("INT1 GPIO port not ready");
        return -ENODEV;
    }
    LOG_INF("I2C bus '%s' ready, target addr=0x%02x", imu_i2c.bus->name, imu_i2c.addr);

    g_ring = ring;
    g_axis = axis;
    memset(&g_stats, 0, sizeof(g_stats));

    // Try I2C bus recovery in case lines are stuck (e.g., a previous run left
    // the chip mid-transaction). Harmless if not supported.
    int rc = i2c_recover_bus(imu_i2c.bus);
    if (rc != 0 && rc != -ENOSYS) {
        LOG_WRN("i2c_recover_bus returned %d", rc);
    }

    // The LSM6DSOX boot time after power-on is up to 10 ms. Give it a margin
    // in case the MCU reset before the chip finished its own boot.
    k_msleep(20);

    // Probe with WHO_AM_I, retrying briefly to ride out any startup glitches.
    uint8_t whoami = 0;
    int last_err = 0;
    for (int attempt = 0; attempt < 5; ++attempt) {
        last_err = reg_read(REG_WHO_AM_I, &whoami, 1);
        if (last_err == 0) break;
        LOG_WRN("WHO_AM_I read attempt %d failed: %d", attempt + 1, last_err);
        k_msleep(20);
    }
    if (last_err != 0) {
        LOG_ERR("WHO_AM_I read failed after retries: %d (check SCL=P0.17, "
                "SDA=P0.20, 3V3, GND, SA0)", last_err);
        return -EIO;
    }
    if (whoami != WHO_AM_I_VALUE) {
        LOG_ERR("WHO_AM_I=0x%02x expected 0x6C (I2C addr / wiring?)", whoami);
        return -ENODEV;
    }
    LOG_INF("WHO_AM_I=0x%02x OK", whoami);

    // Soft reset, then configure CTRL3_C:
    //   bit 6 BDU       = 1 (block data update)
    //   bit 5 H_LACTIVE = 1 (INT pin active-low, matches GPIO_ACTIVE_LOW in DT)
    //   bit 4 PP_OD     = 0 (push-pull — no pull-up needed)
    //   bit 2 IF_INC    = 1 (auto-increment register address on burst reads)
    //   → 0x40 | 0x20 | 0x04 = 0x64
    // Previous 0x54 set PP_OD instead of H_LACTIVE, so INT1 was active-HIGH and
    // Zephyr's EDGE_TO_ACTIVE (falling edge) never saw a transition.
    reg_write(REG_CTRL3_C, 0x01);
    k_msleep(15);
    reg_write(REG_CTRL3_C, 0x64);

    uint8_t ctrl3 = 0;
    if (reg_read(REG_CTRL3_C, &ctrl3, 1) == 0 && ctrl3 != 0x64) {
        LOG_ERR("CTRL3_C readback=0x%02x expected 0x64", ctrl3);
    }

    reg_write(REG_CTRL6_C, 0x00);  // high-perf accel mode

    // FIFO: WTM=64, BDR_XL=3333 Hz, continuous mode
    reg_write(REG_FIFO_CTRL1, FIFO_WTM_LEVEL & 0xFF);
    reg_write(REG_FIFO_CTRL2, 0x00);
    reg_write(REG_FIFO_CTRL3, FIFO_BDR_XL_3K33);
    reg_write(REG_FIFO_CTRL4, FIFO_MODE_CONT);

    // Route FIFO threshold interrupt to INT1
    reg_write(REG_INT1_CTRL, 0x08);  // INT1_FIFO_TH

    // Reset FIFO to bypass so INT1 de-asserts before we arm the edge interrupt.
    // Without this, INT1 may already be active when we enable edge detection,
    // and the first falling edge is never seen.
    reg_write(REG_FIFO_CTRL4, FIFO_MODE_BYPASS);
    k_msleep(2);
    reg_write(REG_FIFO_CTRL4, FIFO_MODE_CONT);

    // Arm GPIO interrupt (falling edge = INT1 going active-low).
    // Order: configure pin → register callback → THEN enable edge detection,
    // so a callback is in place before any edge can fire.
    gpio_pin_configure_dt(&int1_pin, GPIO_INPUT);
    int idle_lvl = gpio_pin_get_dt(&int1_pin);
    LOG_INF("INT1 idle logical=%d (expect 0 = inactive; FIFO empty after bypass)",
            idle_lvl);

    gpio_init_callback(&int1_cb, on_int1, BIT(int1_pin.pin));
    gpio_add_callback(int1_pin.port, &int1_cb);
    gpio_pin_interrupt_configure_dt(&int1_pin, GPIO_INT_EDGE_TO_ACTIVE);

    // Start drain thread before enabling the accel so no IRQ is missed
    atomic_set(&g_running, 1);
    k_thread_create(&g_drain_thread, g_drain_stack, IMU_DRAIN_STACK,
                    drain_thread_fn, NULL, NULL, NULL,
                    K_PRIO_PREEMPT(5), 0, K_NO_WAIT);
    k_thread_name_set(&g_drain_thread, "imu_drain");

    // Start accelerometer
    reg_write(REG_CTRL1_XL, ODR_3K33_FS_4G);

    // Sanity: at 3.33 kHz with WTM=64, FIFO reaches watermark in ~19 ms.
    // If the chip is OK, after 50 ms the FIFO level should be ≥ 64 and INT1
    // should be asserted (logical=1 with active-low). If FIFO fills but INT1
    // never asserts, INT routing is wrong. If neither, accel/FIFO is broken.
    k_msleep(50);
    uint8_t st[2] = {0};
    (void)reg_read(REG_FIFO_STATUS1, st, 2);
    uint16_t lvl = ((uint16_t)(st[1] & 0x03) << 8) | st[0];
    int active_lvl = gpio_pin_get_dt(&int1_pin);
    LOG_INF("After accel start: fifo_level=%u INT1 logical=%d "
            "(expect lvl>=64, INT1=1)", lvl, active_lvl);

    LOG_INF("LSM6DSOX (I2C@0x%02x) streaming @ 3.33 kHz, axis=%d",
            imu_i2c.addr, (int)axis);
    return 0;
}

void imu_stop(void) {
    reg_write(REG_CTRL1_XL, 0x00);        // power-down accel
    gpio_remove_callback(int1_pin.port, &int1_cb);
    atomic_set(&g_running, 0);
    k_sem_give(&g_fifo_sem);              // unblock drain thread so it can exit
    k_thread_join(&g_drain_thread, K_FOREVER);
}

void imu_get_stats(struct imu_stats *out) { *out = g_stats; }
