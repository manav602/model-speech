// DS-CNN-M INT8-weight forward pass (float activations)
//
// Layer pipeline (matches the PyTorch DSCNN class exactly after BN folding):
//   1. stem       Conv2d(1->172, kernel=10x4, stride=2x2, pad=5x1) + ReLU
//   2-5. blocks   for i in 0..3:
//                   Conv2d depthwise 3x3 stride 1 + ReLU
//                   Conv2d pointwise 1x1          + ReLU
//   6. global avg pool over (H, W)
//   7. fc Linear (172 -> N_CLASSES)
//
// All weights are int8 with per-output-channel float scale.
// Compute is float32 (small SRAM footprint hit, but matches PTQ math bit-exact
// and avoids needing per-layer activation calibration).

#include "kws.h"
#include <math.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>

#define MAX(a,b) ((a)>(b)?(a):(b))
#define MIN(a,b) ((a)<(b)?(a):(b))
#define IDX3(c,h,w, H,W) (((c)*(H)+(h))*(W)+(w))
#define IDX4(o,i,kh,kw, IC,KH,KW) ((((o)*(IC)+(i))*(KH)+(kh))*(KW)+(kw))

static inline float relu(float x) { return x > 0.0f ? x : 0.0f; }

// Conv2d (single-input or multi-input, NOT depthwise). Float in/out, INT8 weights.
//   in    : [in_c, in_h, in_w]
//   W_q   : [out_c, in_c, kh, kw] (int8)
//   S     : [out_c] (float, per-out-channel scale)
//   B     : [out_c] (float, folded bias)
//   out   : [out_c, out_h, out_w]
static void conv2d_int8w(const float* in, int in_c, int in_h, int in_w,
                          const int8_t* W_q, const float* S, const float* B,
                          int out_c, int kh, int kw, int sh, int sw, int ph, int pw,
                          float* out, int relu_act) {
    const int out_h = (in_h + 2 * ph - kh) / sh + 1;
    const int out_w = (in_w + 2 * pw - kw) / sw + 1;
    for (int oc = 0; oc < out_c; ++oc) {
        const float scale = S[oc];
        const float bias  = B[oc];
        for (int oh = 0; oh < out_h; ++oh) {
            for (int ow = 0; ow < out_w; ++ow) {
                float acc = 0.0f;
                for (int ic = 0; ic < in_c; ++ic) {
                    for (int ki = 0; ki < kh; ++ki) {
                        const int ih = oh * sh + ki - ph;
                        if (ih < 0 || ih >= in_h) continue;
                        for (int kj = 0; kj < kw; ++kj) {
                            const int iw = ow * sw + kj - pw;
                            if (iw < 0 || iw >= in_w) continue;
                            const float v = in[IDX3(ic, ih, iw, in_h, in_w)];
                            const int8_t qw = W_q[IDX4(oc, ic, ki, kj, in_c, kh, kw)];
                            acc += v * (float)qw;
                        }
                    }
                }
                float y = acc * scale + bias;
                if (relu_act) y = relu(y);
                out[IDX3(oc, oh, ow, out_h, out_w)] = y;
            }
        }
    }
}

// Depthwise conv2d (groups = channels). Weight shape [c, 1, kh, kw].
static void dwconv2d_int8w(const float* in, int c, int in_h, int in_w,
                            const int8_t* W_q, const float* S, const float* B,
                            int kh, int kw, int sh, int sw, int ph, int pw,
                            float* out, int relu_act) {
    const int out_h = (in_h + 2 * ph - kh) / sh + 1;
    const int out_w = (in_w + 2 * pw - kw) / sw + 1;
    for (int ch = 0; ch < c; ++ch) {
        const float scale = S[ch];
        const float bias  = B[ch];
        for (int oh = 0; oh < out_h; ++oh) {
            for (int ow = 0; ow < out_w; ++ow) {
                float acc = 0.0f;
                for (int ki = 0; ki < kh; ++ki) {
                    const int ih = oh * sh + ki - ph;
                    if (ih < 0 || ih >= in_h) continue;
                    for (int kj = 0; kj < kw; ++kj) {
                        const int iw = ow * sw + kj - pw;
                        if (iw < 0 || iw >= in_w) continue;
                        const float v = in[IDX3(ch, ih, iw, in_h, in_w)];
                        // dw weight layout: [c, 1, kh, kw] → flat index [ch][0][ki][kj]
                        const int8_t qw = W_q[((ch * 1 + 0) * kh + ki) * kw + kj];
                        acc += v * (float)qw;
                    }
                }
                float y = acc * scale + bias;
                if (relu_act) y = relu(y);
                out[IDX3(ch, oh, ow, out_h, out_w)] = y;
            }
        }
    }
}

// Linear layer (INT8 weights). W_q: [out_f, in_f]
static void linear_int8w(const float* in, int in_f,
                          const int8_t* W_q, const float* S, const float* B,
                          int out_f, float* out) {
    for (int o = 0; o < out_f; ++o) {
        float acc = 0.0f;
        for (int i = 0; i < in_f; ++i) {
            acc += in[i] * (float)W_q[o * in_f + i];
        }
        out[o] = acc * S[o] + B[o];
    }
}

// ─── Forward pass ───
int kws_forward(const float* feat, float* logits_out, void* arena, int arena_bytes) {
    // Use the arena as two ping-pong activation buffers
    const int max_act_floats = arena_bytes / (2 * sizeof(float));
    float* buf0 = (float*)arena;
    float* buf1 = buf0 + max_act_floats;

    // Input feat: [N_MELS, T] = [30, 100], treat as [in_c=1, H=30, W=100]
    const int in_c = 1;
    int H = DSCNN_N_MELS;        // 30
    int W = KWS_FEAT_T;          // 100

    // 1. stem: Conv2d(1 -> 172, kernel from header, stride from header, pad from header)
    {
        const int kh=DSCNN_STEM_KERNEL_H, kw=DSCNN_STEM_KERNEL_W;
        const int sh=DSCNN_STEM_STRIDE_H, sw=DSCNN_STEM_STRIDE_W;
        const int ph=DSCNN_STEM_PADDING_H, pw=DSCNN_STEM_PADDING_W;
        const int oh = (H + 2*ph - kh) / sh + 1;
        const int ow = (W + 2*pw - kw) / sw + 1;
        if ((int)(DSCNN_FILTERS * oh * ow * sizeof(float)) > arena_bytes / 2) return -1;
        conv2d_int8w(feat, in_c, H, W,
                      stem_W, stem_S, stem_B,
                      DSCNN_FILTERS, kh, kw, sh, sw, ph, pw,
                      buf0, 1);
        H = oh; W = ow;
    }

    int cur = 0;  // 0 = buf0 has activations, 1 = buf1
    float* curp = buf0;
    float* nxt  = buf1;

    // 2-5. Blocks: depthwise(3x3 s1 p1) → relu → pointwise(1x1) → relu
    const int8_t* dw_Ws[4] = { block0_dw_W, block1_dw_W, block2_dw_W, block3_dw_W };
    const float*  dw_Ss[4] = { block0_dw_S, block1_dw_S, block2_dw_S, block3_dw_S };
    const float*  dw_Bs[4] = { block0_dw_B, block1_dw_B, block2_dw_B, block3_dw_B };
    const int8_t* pw_Ws[4] = { block0_pw_W, block1_pw_W, block2_pw_W, block3_pw_W };
    const float*  pw_Ss[4] = { block0_pw_S, block1_pw_S, block2_pw_S, block3_pw_S };
    const float*  pw_Bs[4] = { block0_pw_B, block1_pw_B, block2_pw_B, block3_pw_B };

    for (int blk = 0; blk < DSCNN_N_BLOCKS; ++blk) {
        // Depthwise — same H,W
        dwconv2d_int8w(curp, DSCNN_FILTERS, H, W,
                        dw_Ws[blk], dw_Ss[blk], dw_Bs[blk],
                        3, 3, 1, 1, 1, 1, nxt, 1);
        cur = 1 - cur; { float* t = curp; curp = nxt; nxt = t; }
        // Pointwise — same H,W, channels in = out = 172
        conv2d_int8w(curp, DSCNN_FILTERS, H, W,
                      pw_Ws[blk], pw_Ss[blk], pw_Bs[blk],
                      DSCNN_FILTERS, 1, 1, 1, 1, 0, 0, nxt, 1);
        cur = 1 - cur; { float* t = curp; curp = nxt; nxt = t; }
    }

    // 6. Global average pool over (H, W) → [DSCNN_FILTERS]
    static float pooled[DSCNN_FILTERS];
    const float inv_hw = 1.0f / (float)(H * W);
    for (int c = 0; c < DSCNN_FILTERS; ++c) {
        float s = 0.0f;
        const float* p = curp + c * H * W;
        for (int i = 0; i < H * W; ++i) s += p[i];
        pooled[c] = s * inv_hw;
    }

    // 7. FC: Linear(172 -> N_CLASSES)
    linear_int8w(pooled, DSCNN_FILTERS, fc_W, fc_S, fc_B, DSCNN_N_CLASSES, logits_out);

    // argmax
    int best = 0; float best_v = logits_out[0];
    for (int c = 1; c < DSCNN_N_CLASSES; ++c) {
        if (logits_out[c] > best_v) { best_v = logits_out[c]; best = c; }
    }
    return best;
}
