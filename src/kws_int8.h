// W8A8 DS-CNN-M forward pass (INT8 weights + INT8 activations)
//   input:  log-mel float [N_MELS x T] (output of dsp_logmel)
//   output: float logits [N_CLASSES]
#pragma once
#include <stdint.h>
#include "../model/dscnn_m_w8a8.h"

#define KWS_FEAT_T   ((DSCNN_WINDOW_SAMPLES - DSCNN_N_FFT) / DSCNN_HOP + 1)  // = 99

#define KWS_STEM_OUT_H  ((DSCNN_N_MELS + 2*DSCNN_STEM_PADDING_H - DSCNN_STEM_KERNEL_H) / DSCNN_STEM_STRIDE_H + 1)
#define KWS_STEM_OUT_W  ((KWS_FEAT_T   + 2*DSCNN_STEM_PADDING_W - DSCNN_STEM_KERNEL_W) / DSCNN_STEM_STRIDE_W + 1)

// Two ping-pong INT8 buffers, each = DSCNN_FILTERS × KWS_STEM_OUT_H × KWS_STEM_OUT_W bytes.
#define KWS_ACT_BYTES   (DSCNN_FILTERS * KWS_STEM_OUT_H * KWS_STEM_OUT_W)
#define KWS_ARENA_BYTES (2 * KWS_ACT_BYTES + DSCNN_N_MELS * KWS_FEAT_T)
//                       ^ ping-pong int8        ^ feat input int8

// Run W8A8 inference. Returns predicted class index.
int kws_forward_int8(const float* feat,        // [N_MELS, T] log-mel float input
                      float* logits_out,        // [N_CLASSES] float
                      int8_t* arena,            // workspace, size >= KWS_ARENA_BYTES
                      int arena_bytes);
