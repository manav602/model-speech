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
#include "ble_kws.h"
#include "dsp.h"
#include "kws_int8.h"

#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>
#include <zephyr/shell/shell.h>
#include <math.h>
#include <string.h>

LOG_MODULE_REGISTER(kws, LOG_LEVEL_INF);

// SRAM-resident buffers. Sizing assumes the steady-state path where
// dsp_logmel is fed exactly WINDOW_SAMPLES, so wav aliases hp_in inside DSP
// and the scratch only needs WINDOW + N_FFT + (N_FFT/2+1) floats (~27 KB
// vs the previous ~54 KB).
//
// Lifetime overlay: s_window is consumed by dsp_logmel up-front, and s_logits
// is only written after DSP finishes. Both can share the tail of s_dsp_scratch
// via a union — saves ~13 KB SRAM. The union is sized to the larger member
// (the scratch). Layout per pass:
//   t=0     : DSP reads s_window (int16)  -> writes hp_in/frame/power floats
//   t=DSP   : s_window is dead; scratch owns the buffer
//   t=infer : kws_forward_int8 writes s_logits at the end of the buffer.
static int8_t  s_arena[KWS_ARENA_BYTES] __aligned(8);
static float   s_logmel[DSCNN_N_MELS * KWS_FEAT_T];

#define KWS_DSP_SCRATCH_FLOATS \
    (WINDOW_SAMPLES + DSCNN_N_FFT + (DSCNN_N_FFT/2 + 1))

static union {
    int16_t window[DSCNN_WINDOW_SAMPLES];                            // 13.3 KB
    float   scratch[KWS_DSP_SCRATCH_FLOATS];                         // ~27 KB
} s_buf __aligned(8);

#define s_window      (s_buf.window)
#define s_dsp_scratch (s_buf.scratch)

static float   s_logits[DSCNN_N_CLASSES];

// Inference thread does no recursion and uses static buffers; 2 KB is plenty
// (was 4 KB). Logging frames + a few hundred bytes of locals is the worst case.
#define KWS_THREAD_STACK 2048
K_THREAD_STACK_DEFINE(kws_stack, KWS_THREAD_STACK);
static struct k_thread kws_thread;
static k_tid_t         kws_tid;

static struct kws_config s_cfg;
static volatile bool     s_running;
static struct kws_pipeline_stats s_stats;
static int64_t           s_last_detect_ms;

// N-of-N consecutive agreement state. A class only fires after it has been
// the argmax (above threshold) on KWS_AGREE_N consecutive 250 ms windows.
// At slide_ms=250 ms with KWS_AGREE_N=3 this means ~500 ms of sustained
// agreement (windows overlap), which kills almost all single-spike false
// positives without adding noticeable latency for real activity.
#ifndef KWS_AGREE_N
#define KWS_AGREE_N 3
#endif
static int s_run_cls;
static int s_run_len;
static float s_run_min_conf;

// ── Diagnostic windows ────────────────────────────────────────────────────
// These let an operator validate every threshold/assumption from the live log
// without recompiling. Counters are cleared after each rolling window dump.
//
//   peak_min/max/sum, rms_min/max/sum, n  → idle noise distribution
//   per_class[]                           → softmax-argmax frequency
//   conf_sum / conf_n                     → average confidence across all
//                                           non-quiet inferences
//   below_thr                             → windows where argmax was clear
//                                           but confidence < threshold
//   agree_break                           → number of run resets (low conf
//                                           or class flip) — high values
//                                           mean false-positive churn
//   logits_max_sum                        → mean of pre-softmax max logit
//                                           (helps diagnose model bias)
#define KWS_DIAG_PERIOD 30   // dump every N inferences
struct kws_diag {
    uint32_t n;
    float    peak_min, peak_max; double peak_sum;
    float    rms_min,  rms_max;  double rms_sum;
    uint32_t per_class[DSCNN_N_CLASSES];
    double   conf_sum;   uint32_t conf_n;
    double   logit_max_sum;
    uint32_t below_thr;
    uint32_t agree_break;
};
static struct kws_diag s_diag;

static void diag_reset(void) {
    memset(&s_diag, 0, sizeof(s_diag));
    s_diag.peak_min = 1e9f;  s_diag.rms_min = 1e9f;
}

static void diag_dump(void) {
    if (s_diag.n == 0) return;
    const float pavg = (float)(s_diag.peak_sum / s_diag.n);
    const float ravg = (float)(s_diag.rms_sum  / s_diag.n);
    const float cavg = s_diag.conf_n
        ? (float)(s_diag.conf_sum / s_diag.conf_n) : 0.f;
    const float lavg = (float)(s_diag.logit_max_sum / s_diag.n);
    LOG_INF("diag[n=%u]: peak min/avg/max=%.4f/%.4f/%.4f "
            "rms min/avg/max=%.4f/%.4f/%.4f",
            s_diag.n,
            (double)s_diag.peak_min, (double)pavg, (double)s_diag.peak_max,
            (double)s_diag.rms_min,  (double)ravg, (double)s_diag.rms_max);
    LOG_INF("diag[n=%u]: cls counts begin=%u stop=%u wake=%u unk=%u | "
            "conf_avg=%.2f below_thr=%u agree_break=%u logit_max_avg=%.2f",
            s_diag.n,
            s_diag.per_class[KW_BEGIN_ACTIVITY],
            s_diag.per_class[KW_STOP_ACTIVITY],
            s_diag.per_class[KW_WAKE_UP],
            s_diag.per_class[KW_UNKNOWN],
            (double)cavg, s_diag.below_thr, s_diag.agree_break,
            (double)lavg);
    diag_reset();
}

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

        int dsp_rc = dsp_logmel(s_window, DSCNN_WINDOW_SAMPLES,
                                s_logmel, s_dsp_scratch,
                                sizeof(s_dsp_scratch)/sizeof(float));
        s_stats.last_peak = dsp_last_peak();
        s_stats.last_rms  = dsp_last_rms();
        if (dsp_rc > 0) {
            // Window below quiet floor — no motion, skip inference entirely.
            s_stats.skipped_quiet++;
            ble_kws_notify_quiet();
            if ((s_stats.skipped_quiet & 0x1F) == 1) {
                LOG_INF("quiet gate: peak=%.5f skipped=%u",
                        (double)s_stats.last_peak, s_stats.skipped_quiet);
            }
            continue;
        }
        if (dsp_rc < 0) {
            LOG_WRN("dsp_logmel failed: rc=%d", dsp_rc);
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

        // ── Per-window diagnostics ───────────────────────────────────────
        // Every metric the threshold logic depends on is tracked so it can
        // be cross-checked against the fixed assumptions (DSP_QUIET_FLOOR,
        // confidence_threshold, KWS_AGREE_N) from the live log.
        s_diag.n++;
        if (s_stats.last_peak < s_diag.peak_min) s_diag.peak_min = s_stats.last_peak;
        if (s_stats.last_peak > s_diag.peak_max) s_diag.peak_max = s_stats.last_peak;
        s_diag.peak_sum += s_stats.last_peak;
        if (s_stats.last_rms  < s_diag.rms_min)  s_diag.rms_min  = s_stats.last_rms;
        if (s_stats.last_rms  > s_diag.rms_max)  s_diag.rms_max  = s_stats.last_rms;
        s_diag.rms_sum += s_stats.last_rms;
        if (cls >= 0 && cls < DSCNN_N_CLASSES) s_diag.per_class[cls]++;
        s_diag.conf_sum += conf;  s_diag.conf_n++;
        s_diag.logit_max_sum += s_logits[cls];

        // Per-window verbose log — flip the `kws` module to DEBUG via Kconfig
        // (or `log enable dbg kws` from the shell) to see every inference's
        // logits, peak, rms, class. No recompile required.
        LOG_DBG("inf: cls=%d conf=%.3f peak=%.4f rms=%.4f "
                "logits=[%.2f %.2f %.2f %.2f]",
                cls, (double)conf,
                (double)s_stats.last_peak, (double)s_stats.last_rms,
                (double)s_logits[0], (double)s_logits[1],
                (double)s_logits[2], (double)s_logits[3]);

        if (s_diag.n >= KWS_DIAG_PERIOD) diag_dump();

        // Reject low confidence and unknowns. These break the agreement run.
        if (cls == KW_UNKNOWN || conf < s_cfg.confidence_threshold) {
            if (cls != KW_UNKNOWN) s_diag.below_thr++;
            if (s_run_len > 0) s_diag.agree_break++;
            s_run_cls = KW_UNKNOWN;
            s_run_len = 0;
            continue;
        }

        // Track consecutive same-class windows. Only fire once we have N in
        // a row — single-window false positives can no longer trigger.
        if (cls == s_run_cls) {
            s_run_len++;
            if (conf < s_run_min_conf) s_run_min_conf = conf;
        } else {
            if (s_run_len > 0) s_diag.agree_break++;
            s_run_cls = cls;
            s_run_len = 1;
            s_run_min_conf = conf;
        }
        LOG_DBG("run: cls=%d len=%d/%d min_conf=%.2f",
                s_run_cls, s_run_len, KWS_AGREE_N, (double)s_run_min_conf);
        if (s_run_len < KWS_AGREE_N) continue;

        // Debounce: don't re-fire within debounce_ms.
        int64_t now = k_uptime_get();
        if (now - s_last_detect_ms < (int64_t)s_cfg.debounce_ms) continue;
        s_last_detect_ms = now;
        s_run_len = 0;  // require a fresh run before re-firing

        s_stats.detections++;
        LOG_INF("detect: cls=%d conf=%.2f (min=%.2f, %dx) peak=%.4f rms=%.4f "
                "cyc=%u (dsp=%u inf=%u)",
                cls, (double)conf, (double)s_run_min_conf, KWS_AGREE_N,
                (double)s_stats.last_peak, (double)s_stats.last_rms,
                cyc, cyc_dsp, cyc_inf);
        if (s_cfg.on_keyword) s_cfg.on_keyword((keyword_t)cls, s_run_min_conf);
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
    s_run_cls = KW_UNKNOWN;
    s_run_len = 0;
    s_run_min_conf = 0.f;
    diag_reset();
    LOG_INF("KWS thresholds: quiet_peak=%.4f quiet_rms=%.4f "
            "conf_thr=%.2f agree_n=%d debounce=%ums slide=%ums",
            (double)DSP_QUIET_FLOOR, (double)DSP_RMS_FLOOR,
            (double)s_cfg.confidence_threshold, KWS_AGREE_N,
            s_cfg.debounce_ms, s_cfg.slide_ms);
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

// ── Shell commands for live tuning / verification ───────────────────────────
// Build with CONFIG_SHELL=y. Then over UART/RTT:
//   kws diag           — dump rolling diagnostics immediately
//   kws thr            — print all active thresholds
#ifdef CONFIG_SHELL
static int cmd_kws_diag(const struct shell *sh, size_t argc, char **argv) {
    ARG_UNUSED(argc); ARG_UNUSED(argv);
    shell_print(sh, "inferences=%u detections=%u quiet=%u "
                    "last_peak=%.4f last_rms=%.4f",
                s_stats.inferences, s_stats.detections, s_stats.skipped_quiet,
                (double)s_stats.last_peak, (double)s_stats.last_rms);
    diag_dump();
    return 0;
}
static int cmd_kws_thr(const struct shell *sh, size_t argc, char **argv) {
    ARG_UNUSED(argc); ARG_UNUSED(argv);
    shell_print(sh, "quiet_peak=%.4f quiet_rms=%.4f conf_thr=%.2f "
                    "agree_n=%d debounce=%ums slide=%ums",
                (double)DSP_QUIET_FLOOR, (double)DSP_RMS_FLOOR,
                (double)s_cfg.confidence_threshold, KWS_AGREE_N,
                s_cfg.debounce_ms, s_cfg.slide_ms);
    return 0;
}
SHELL_STATIC_SUBCMD_SET_CREATE(sub_kws,
    SHELL_CMD(diag, NULL, "Dump KWS rolling diagnostics", cmd_kws_diag),
    SHELL_CMD(thr,  NULL, "Print KWS thresholds",         cmd_kws_thr),
    SHELL_SUBCMD_SET_END
);
SHELL_CMD_REGISTER(kws, &sub_kws, "Keyword spotting controls", NULL);
#endif
