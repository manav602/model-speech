// Production firmware entry point for nRF52840 + LSM6DSOX.
//
// Boot sequence:
//   1. Enable DWT cycle counter (for inference benchmarking).
//   2. Init the lock-free sample ring.
//   3. Start the IMU streaming Z-axis @ 3.33 kHz into the ring.
//   4. Start the KWS inference thread (sliding-window every 250 ms).
//   5. Idle the main thread (deepsleep between IRQs via WFI).
//
// On detection: toggle LED + log + (optionally) send BLE notification.
#include <zephyr/kernel.h>
#include <zephyr/drivers/gpio.h>
#include <zephyr/logging/log.h>

#include "ring_buffer.h"
#include "imu_lsm6dsox.h"
#include "kws_pipeline.h"

LOG_MODULE_REGISTER(main, LOG_LEVEL_INF);

#define LED0_NODE DT_ALIAS(led0)
static const struct gpio_dt_spec led0 = GPIO_DT_SPEC_GET(LED0_NODE, gpios);

static struct sample_ring g_ring;

static const char* KW_NAMES[] = {
    "begin_activity", "stop_activity", "wake_up", "unknown",
};

static void on_keyword(keyword_t kw, float conf) {
    LOG_INF(">>> %s (conf=%.2f)", KW_NAMES[kw], (double)conf);
    gpio_pin_toggle_dt(&led0);
    // TODO(integration): publish to BLE NUS / MQTT / your event bus here.
}

static void enable_dwt(void) {
    *(volatile uint32_t*)0xE000EDFC |= (1u << 24);  // DEMCR.TRCENA
    *(volatile uint32_t*)0xE0001004  = 0;            // DWT_CYCCNT = 0
    *(volatile uint32_t*)0xE0001000 |= 1u;           // DWT_CTRL.CYCCNTENA
}

int main(void) {
    LOG_INF("DS-CNN-M KWS firmware starting (build " __DATE__ " " __TIME__ ")");

    enable_dwt();
    gpio_pin_configure_dt(&led0, GPIO_OUTPUT_INACTIVE);
    ring_init(&g_ring);

    int rc = imu_start(&g_ring, IMU_AXIS_Z);   // training axis
    if (rc < 0) {
        LOG_ERR("imu_start failed: %d", rc);
        return rc;
    }

    struct kws_config cfg = {
        .ring                 = &g_ring,
        .on_keyword           = on_keyword,
        .confidence_threshold = 0.70f,
        .debounce_ms          = 1500,
        .slide_ms             = 250,
    };
    rc = kws_pipeline_start(&cfg);
    if (rc < 0) {
        LOG_ERR("kws_pipeline_start failed: %d", rc);
        return rc;
    }

    // Periodic health log (every 30 s).
    while (1) {
        k_sleep(K_SECONDS(30));
        struct imu_stats is; struct kws_pipeline_stats ks;
        imu_get_stats(&is);
        kws_pipeline_get_stats(&ks);
        LOG_INF("health: imu_irq=%u samples=%u ovr=%u spi_err=%u | "
                "inf=%u det=%u quiet=%u peak=%.4f rms=%.4f avg_cyc=%u "
                "(dsp=%u inf=%u)",
                is.isr_count, is.samples_pushed, is.fifo_overruns,
                is.spi_errors, ks.inferences, ks.detections,
                ks.skipped_quiet, (double)ks.last_peak, (double)ks.last_rms,
                ks.avg_cycles, ks.avg_dsp_cycles, ks.avg_inf_cycles);
    }
    return 0;
}
