// W8A8 DS-CNN-M forward pass.
//
// Layer pipeline (matches the PyTorch DSCNN class after BN folding):
//   1. quant float feat → int8 (per-tensor scale = DSCNN_FEAT_IN_SCALE)
//   2. stem    Conv2d(1→172, 10x4 stride S_h x S_w pad 5x1) + ReLU  → int8
//   3-6. blocks (×4):
//        DW Conv 3x3 + ReLU                                          → int8
//        PW Conv 1x1 + ReLU                                          → int8
//   7. global avg pool over (H, W) → [172] int32 sum / area (kept int)
//   8. FC Linear → float logits (dequantized for argmax/softmax)
//
// All conv kernels: int8 in × int8 weight → int32 acc → +int32 bias →
// (acc * float M[oc]) → optional ReLU → clip to int8.
// One float multiply per output element, all multiply-accumulate is integer.

#include "kws_int8.h"
#include <stdint.h>
#include <math.h>
#include <string.h>

#define IDX3(c,h,w, H,W) (((c)*(H)+(h))*(W)+(w))
#define IDX4(o,i,kh,kw, IC,KH,KW) ((((o)*(IC)+(i))*(KH)+(kh))*(KW)+(kw))

static inline int8_t requant_relu(int32_t acc, float M, int relu_act) {
    float y = (float)acc * M;
    if (relu_act && y < 0.0f) y = 0.0f;
    int q = (int)lrintf(y);
    if (q >  127) q =  127;
    if (q < -128) q = -128;
    return (int8_t)q;
}

// ─── Specialized 1×1 conv (point-wise): no bounds checks, contiguous load.
// This is the hot path — 4 PW layers with 172×172 channels dominate inference.
static void conv1x1_w8a8(const int8_t* in, int in_c, int in_h, int in_w,
                          const int8_t* W, const int32_t* bias, const float* M,
                          int out_c, int8_t* out, int relu_act) {
    const int spatial = in_h * in_w;
    for (int oc = 0; oc < out_c; ++oc) {
        const int8_t* wrow = W + oc * in_c;
        const float    M_oc = M[oc];
        const int32_t  b_oc = bias[oc];
        int8_t* op = out + oc * spatial;
        for (int sp = 0; sp < spatial; ++sp) {
            int32_t acc = b_oc;
            const int8_t* ip = in + sp;
            // Inner sum over input channels. Contiguous along ic in W (stride 1)
            // but strided in `in` (stride = spatial). Compiler with -O2 can
            // schedule the dual loads efficiently on Cortex-M4F.
            for (int ic = 0; ic < in_c; ++ic) {
                acc += (int32_t)ip[ic * spatial] * (int32_t)wrow[ic];
            }
            op[sp] = requant_relu(acc, M_oc, relu_act);
        }
    }
}

// ─── Conv2d (multi-input, NOT depthwise) ───
static void conv2d_w8a8(const int8_t* in, int in_c, int in_h, int in_w,
                         const int8_t* W, const int32_t* bias, const float* M,
                         int out_c, int kh, int kw, int sh, int sw, int ph, int pw,
                         int8_t* out, int relu_act) {
    const int out_h = (in_h + 2 * ph - kh) / sh + 1;
    const int out_w = (in_w + 2 * pw - kw) / sw + 1;
    for (int oc = 0; oc < out_c; ++oc) {
        const float    M_oc = M[oc];
        const int32_t  b_oc = bias[oc];
        for (int oh = 0; oh < out_h; ++oh) {
            for (int ow = 0; ow < out_w; ++ow) {
                int32_t acc = 0;
                for (int ic = 0; ic < in_c; ++ic) {
                    for (int ki = 0; ki < kh; ++ki) {
                        int ih = oh * sh + ki - ph;
                        if (ih < 0 || ih >= in_h) continue;
                        for (int kj = 0; kj < kw; ++kj) {
                            int iw = ow * sw + kj - pw;
                            if (iw < 0 || iw >= in_w) continue;
                            int8_t a = in[IDX3(ic, ih, iw, in_h, in_w)];
                            int8_t w = W[IDX4(oc, ic, ki, kj, in_c, kh, kw)];
                            acc += (int32_t)a * (int32_t)w;
                        }
                    }
                }
                acc += b_oc;
                out[IDX3(oc, oh, ow, out_h, out_w)] = requant_relu(acc, M_oc, relu_act);
            }
        }
    }
}

// ─── Depthwise (groups = channels). Weight shape [c, 1, kh, kw]. ───
static void dwconv2d_w8a8(const int8_t* in, int c, int in_h, int in_w,
                           const int8_t* W, const int32_t* bias, const float* M,
                           int kh, int kw, int sh, int sw, int ph, int pw,
                           int8_t* out, int relu_act) {
    const int out_h = (in_h + 2 * ph - kh) / sh + 1;
    const int out_w = (in_w + 2 * pw - kw) / sw + 1;
    for (int ch = 0; ch < c; ++ch) {
        const float    M_oc = M[ch];
        const int32_t  b_oc = bias[ch];
        for (int oh = 0; oh < out_h; ++oh) {
            for (int ow = 0; ow < out_w; ++ow) {
                int32_t acc = 0;
                for (int ki = 0; ki < kh; ++ki) {
                    int ih = oh * sh + ki - ph;
                    if (ih < 0 || ih >= in_h) continue;
                    for (int kj = 0; kj < kw; ++kj) {
                        int iw = ow * sw + kj - pw;
                        if (iw < 0 || iw >= in_w) continue;
                        int8_t a = in[IDX3(ch, ih, iw, in_h, in_w)];
                        int8_t w = W[((ch * 1 + 0) * kh + ki) * kw + kj];
                        acc += (int32_t)a * (int32_t)w;
                    }
                }
                acc += b_oc;
                out[IDX3(ch, oh, ow, out_h, out_w)] = requant_relu(acc, M_oc, relu_act);
            }
        }
    }
}

// ─── Quantize float feat → int8 ───
static void quant_feat(const float* feat_fp, int n, int8_t* out, float scale) {
    const float inv = 1.0f / scale;
    for (int i = 0; i < n; ++i) {
        int q = (int)lrintf(feat_fp[i] * inv);
        if (q >  127) q =  127;
        if (q < -128) q = -128;
        out[i] = (int8_t)q;
    }
}

// ─── Forward pass ───
int kws_forward_int8(const float* feat, float* logits_out,
                      int8_t* arena, int arena_bytes) {
    if (arena_bytes < KWS_ARENA_BYTES) return -1;

    // Layout: [feat_int8 | buf0 | buf1]
    int8_t* feat_q = arena;
    int8_t* buf0   = feat_q + DSCNN_N_MELS * KWS_FEAT_T;
    int8_t* buf1   = buf0   + KWS_ACT_BYTES;

    // 0. Quantize float log-mel → int8
    quant_feat(feat, DSCNN_N_MELS * KWS_FEAT_T, feat_q, DSCNN_FEAT_IN_SCALE);

    int H = DSCNN_N_MELS, W = KWS_FEAT_T;

    // 1. Stem: Conv2d(1 → 172)
    {
        const int kh=DSCNN_STEM_KERNEL_H, kw=DSCNN_STEM_KERNEL_W;
        const int sh=DSCNN_STEM_STRIDE_H, sw=DSCNN_STEM_STRIDE_W;
        const int ph=DSCNN_STEM_PADDING_H, pw=DSCNN_STEM_PADDING_W;
        conv2d_w8a8(feat_q, 1, H, W,
                     stem_W, stem_B32, stem_M,
                     DSCNN_FILTERS, kh, kw, sh, sw, ph, pw,
                     buf0, /*relu*/1);
        H = (H + 2*ph - kh) / sh + 1;
        W = (W + 2*pw - kw) / sw + 1;
    }

    int8_t* curp = buf0;
    int8_t* nxt  = buf1;

    // 2-5. Blocks: DW(3x3 s1 p1) + ReLU → PW(1x1) + ReLU
    const int8_t*  dw_W [4] = { block0_dw_W, block1_dw_W, block2_dw_W, block3_dw_W };
    const int32_t* dw_B [4] = { block0_dw_B32, block1_dw_B32, block2_dw_B32, block3_dw_B32 };
    const float*   dw_M [4] = { block0_dw_M, block1_dw_M, block2_dw_M, block3_dw_M };
    const int8_t*  pw_W [4] = { block0_pw_W, block1_pw_W, block2_pw_W, block3_pw_W };
    const int32_t* pw_B [4] = { block0_pw_B32, block1_pw_B32, block2_pw_B32, block3_pw_B32 };
    const float*   pw_M [4] = { block0_pw_M, block1_pw_M, block2_pw_M, block3_pw_M };

    for (int blk = 0; blk < DSCNN_N_BLOCKS; ++blk) {
        dwconv2d_w8a8(curp, DSCNN_FILTERS, H, W,
                       dw_W[blk], dw_B[blk], dw_M[blk],
                       3, 3, 1, 1, 1, 1, nxt, 1);
        { int8_t* t = curp; curp = nxt; nxt = t; }
        conv1x1_w8a8(curp, DSCNN_FILTERS, H, W,
                      pw_W[blk], pw_B[blk], pw_M[blk],
                      DSCNN_FILTERS, nxt, 1);
        { int8_t* t = curp; curp = nxt; nxt = t; }
    }

    // 6. Global avg pool over (H, W) — accumulate int32, divide at end (float)
    static float pooled[DSCNN_FILTERS];
    const float scale = DSCNN_FC_IN_SCALE;             // dequant scale
    const float inv_hw = 1.0f / (float)(H * W);
    for (int c = 0; c < DSCNN_FILTERS; ++c) {
        int32_t s = 0;
        const int8_t* p = curp + c * H * W;
        for (int i = 0; i < H * W; ++i) s += (int32_t)p[i];
        pooled[c] = (float)s * scale * inv_hw;          // dequant + average
    }

    // 7. FC: float in × int8 weight → float out (cheap, only 172 × 4 multiplies)
    for (int oc = 0; oc < DSCNN_N_CLASSES; ++oc) {
        float acc = 0.0f;
        const int8_t* wrow = &fc_W[oc * DSCNN_FILTERS];
        for (int ic = 0; ic < DSCNN_FILTERS; ++ic) {
            acc += pooled[ic] * (float)wrow[ic];
        }
        logits_out[oc] = acc * fc_S[oc] + fc_B[oc];
    }

    int best = 0; float best_v = logits_out[0];
    for (int c = 1; c < DSCNN_N_CLASSES; ++c) {
        if (logits_out[c] > best_v) { best_v = logits_out[c]; best = c; }
    }
    return best;
}
