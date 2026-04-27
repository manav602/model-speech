// DS-CNN-M INT8-weight forward pass (float activations)
//   input:  log-mel features [N_MELS x T] (float, standardized)
//   output: class logits [N_CLASSES] (float)
#pragma once
#include <stdint.h>
#include "../model/dscnn_m_int8.h"

#define KWS_FEAT_T   100   // mel time frames

// Stem output spatial dims (depend on stem_stride from training)
#define KWS_STEM_OUT_H  ((DSCNN_N_MELS + 2*DSCNN_STEM_PADDING_H - DSCNN_STEM_KERNEL_H) / DSCNN_STEM_STRIDE_H + 1)
#define KWS_STEM_OUT_W  ((KWS_FEAT_T   + 2*DSCNN_STEM_PADDING_W - DSCNN_STEM_KERNEL_W) / DSCNN_STEM_STRIDE_W + 1)

// Two ping-pong buffers, each sized for the largest activation tensor.
// Largest tensor = DSCNN_FILTERS × KWS_STEM_OUT_H × KWS_STEM_OUT_W floats.
#define KWS_ACT_FLOATS  (DSCNN_FILTERS * KWS_STEM_OUT_H * KWS_STEM_OUT_W)
#define KWS_ARENA_BYTES (2 * KWS_ACT_FLOATS * (int)sizeof(float))

// Run inference. Returns predicted class index (0..N_CLASSES-1).
// `feat` is row-major [DSCNN_N_MELS, KWS_FEAT_T].
// `logits_out` receives [DSCNN_N_CLASSES] floats.
// `arena` is workspace of >= KWS_ARENA_BYTES; lifetime: caller-owned.
int kws_forward(const float* feat, float* logits_out, void* arena, int arena_bytes);
