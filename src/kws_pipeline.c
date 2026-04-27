// Inference thread.
//
//   while (running):
//      sleep(slide_ms)
//      snapshot last DSCNN_WINDOW_SAMPLES from ring   (~13 KB memcpy)
//      dsp_logmel(...)                                (~30 ms with CMSIS RFFT)
//      kws_forward_int8(...)                          (~10–20 ms with CMSIS-NN)
//      softmax → argmax → threshold → debounce → callback
//
// Designed for a single Zephyr thread at MAIN_PRIORITY-1 so the IMU SPI ISR
// always preempts inference — losing a sample is much worse than missing a
// 250 ms slide.
#include "kws_pipeline.h"
#include "dsp.h"
#include "kws_int8.h"

#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>
#include <math.h>
#include <string.h>

LOG_MODULE_REGISTER(kws, LOG_LEVEL_INF);

// SRAM-resident buffers (matches main.c QEMU build, sized for nRF52840 256 KB).
static int8_t  s_arena[KWS_ARENA_BYTES] __aligned(8);
static int16_t s_window[DSCNN_WINDOW_SAMPLES];                       // 13.3 KB
static float   s_logmel[DSCNN_N_MELS * KWS_FEAT_T];
static float   s_logits[DSCNN_N_CLASSES];
static float   s_dsp_scratch[WINDOW_SAMPLES + WINDOW_SAMPLES
                             + DSCNN_N_FFT + (DSCNN_N_FFT/2 + 1)];

#define KWS_THREAD_STACK 4096
K_THREAD_STACK_DEFINE(kws_stack, KWS_THREAD_STACK);
static struct k_thread kws_thread;
static k_tid_t         kws_tid;

static struct kws_config s_cfg;
static volatile bool     s_running;
static struct kws_pipeline_stats s_stats;
static int64_t           s_last_detect_ms;

static int argmax_softmax(const float* x, int n, float* out_p) {
    int best = 0;
    float m = x[0];
    for (int i = 1; i < n; ++i) if (x[i] > m) { m = x[i]; best = i; }
    float sum = 0.f;
    for (int i = 0; i < n; ++i) sum += expf(x[i] - m);
    *out_p = 1.0f / sum;  // = exp(x[best]-m) / sum, and exp(0)=1
    return best;
}

static inline uint32_t cyc_now(void) {
    // DWT cycle counter (enabled in main).
    return *(volatile uint32_t*)0xE0001004;
}

static void kws_loop(void* a, void* b, void* c) {
    ARG_UNUSED(a); ARG_UNUSED(b); ARG_UNUSED(c);
    LOG_INF("KWS thread up: window=%d samples, slide=%u ms, thr=%.2f",
            DSCNN_WINDOW_SAMPLES, s_cfg.slide_ms,
            (double)s_cfg.confidence_threshold);

    while (s_running) {
        k_msleep(s_cfg.slide_ms);

        if (!ring_snapshot_last(s_cfg.ring, s_window, DSCNN_WINDOW_SAMPLES)) {
            s_stats.skipped_cold_start++;
            continue;
        }

        uint32_t c0 = cyc_now();

        if (dsp_logmel(s_window, DSCNN_WINDOW_SAMPLES,
                       s_logmel, s_dsp_scratch,
                       sizeof(s_dsp_scratch)/sizeof(float)) != 0) {
            LOG_WRN("dsp_logmel failed");
            continue;
        }
        uint32_t c1 = cyc_now();
        int pred = kws_forward_int8(s_logmel, s_logits,
                                    s_arena, KWS_ARENA_BYTES);
        uint32_t c2 = cyc_now();

        uint32_t cyc      = c2 - c0;
        uint32_t cyc_dsp  = c1 - c0;
        uint32_t cyc_inf  = c2 - c1;
        s_stats.last_cycles = cyc;
        if (s_stats.inferences) {
            uint64_t n = s_stats.inferences;
            s_stats.avg_cycles     = (uint32_t)(((uint64_t)s_stats.avg_cycles    * n + cyc)     / (n + 1));
            s_stats.avg_dsp_cycles = (uint32_t)(((uint64_t)s_stats.avg_dsp_cycles* n + cyc_dsp) / (n + 1));
            s_stats.avg_inf_cycles = (uint32_t)(((uint64_t)s_stats.avg_inf_cycles* n + cyc_inf) / (n + 1));
        } else {
            s_stats.avg_cycles     = cyc;
            s_stats.avg_dsp_cycles = cyc_dsp;
            s_stats.avg_inf_cycles = cyc_inf;
        }
        s_stats.inferences++;

        float conf;
        int cls = argmax_softmax(s_logits, DSCNN_N_CLASSES, &conf);
        (void)pred;  // == cls

        // Reject low confidence and unknowns.
        if (cls == KW_UNKNOWN) continue;
        if (conf < s_cfg.confidence_threshold) continue;

        // Debounce: don't re-fire within debounce_ms.
        int64_t now = k_uptime_get();
        if (now - s_last_detect_ms < (int64_t)s_cfg.debounce_ms) continue;
        s_last_detect_ms = now;

        s_stats.detections++;
        if (s_cfg.on_keyword) s_cfg.on_keyword((keyword_t)cls, conf);
    }
}

int kws_pipeline_start(const struct kws_config* cfg) {
    if (!cfg || !cfg->ring) return -EINVAL;
    s_cfg = *cfg;
    if (s_cfg.slide_ms == 0) s_cfg.slide_ms = 250;
    if (s_cfg.debounce_ms == 0) s_cfg.debounce_ms = 1500;
    if (s_cfg.confidence_threshold <= 0.f) s_cfg.confidence_threshold = 0.70f;

    memset(&s_stats, 0, sizeof(s_stats));
    s_last_detect_ms = -((int64_t)s_cfg.debounce_ms);
    s_running = true;

    kws_tid = k_thread_create(&kws_thread, kws_stack, KWS_THREAD_STACK,
                              kws_loop, NULL, NULL, NULL,
                              K_PRIO_PREEMPT(7), 0, K_NO_WAIT);
    k_thread_name_set(kws_tid, "kws");
    return 0;
}

void kws_pipeline_stop(void) {
    s_running = false;
    if (kws_tid) k_thread_join(kws_tid, K_FOREVER);
    kws_tid = NULL;
}

void kws_pipeline_get_stats(struct kws_pipeline_stats* out) { *out = s_stats; }
