// Inference pipeline: snapshots the most recent 2.0 s of IMU samples,
// runs DSP + W8A8 KWS, applies softmax + confidence + debounce, and reports
// detected keywords via callback.
#pragma once
#include <stdint.h>
#include "ring_buffer.h"

typedef enum {
    KW_BEGIN_ACTIVITY = 0,
    KW_STOP_ACTIVITY  = 1,
    KW_WAKE_UP        = 2,
    KW_UNKNOWN        = 3,
    KW_NONE           = -1,   // below confidence threshold
} keyword_t;

typedef void (*kws_event_cb_t)(keyword_t kw, float confidence);

struct kws_config {
    struct sample_ring* ring;
    kws_event_cb_t      on_keyword;
    float               confidence_threshold;  // e.g. 0.70
    uint32_t            debounce_ms;           // e.g. 1500
    uint32_t            slide_ms;              // e.g. 250
};

// Spawn the inference thread. Returns 0 on success.
int kws_pipeline_start(const struct kws_config* cfg);
void kws_pipeline_stop(void);

struct kws_pipeline_stats {
    uint32_t inferences;
    uint32_t detections;
    uint32_t skipped_cold_start;
    uint32_t skipped_quiet;       // gated out by DSP_QUIET_FLOOR
    uint32_t last_cycles;
    uint32_t avg_cycles;
    uint32_t avg_dsp_cycles;
    uint32_t avg_inf_cycles;
    float    last_peak;           // last observed post-HP peak
    float    last_rms;            // last observed post-HP RMS
};
void kws_pipeline_get_stats(struct kws_pipeline_stats* out);
