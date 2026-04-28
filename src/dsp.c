// DSP front-end implementation. Hand-rolled DFT + mel filterbank.
#include "dsp.h"
#include "../model/mel_filterbank.h"
#include <math.h>
#include <stdint.h>
#include <zephyr/logging/log.h>

LOG_MODULE_REGISTER(dsp, LOG_LEVEL_INF);

static float s_last_peak;          // most recent post-HP peak (for telemetry)
static float s_last_rms;           // most recent post-HP RMS  (for telemetry)
float dsp_last_peak(void) { return s_last_peak; }
float dsp_last_rms(void)  { return s_last_rms;  }

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

// CMSIS RFFT requires pow-of-2 N; DSCNN_N_FFT=133 is not, so we use a cached
// naive DFT path. The CMSIS branch was removed to save flash.

// 4th-order Butterworth HP @ 80 Hz, fs=3333 Hz, designed in scipy:
//   sosbutter(4, 80/(3333/2), btype='high') → 2 biquads, transposed direct II
//
// Pre-computed SOS coefficients (b0,b1,b2,a1,a2 per stage; a0=1)
static const float HP_SOS[2][5] = {
    // stage 1
    { 0.92250928f, -1.84501857f,  0.92250928f, -1.84017057f,  0.84986657f },
    // stage 2
    { 1.00000000f, -2.00000000f,  1.00000000f, -1.92129237f,  0.93098457f },
};

void dsp_hp_filter_inplace(float* x, int n) {
    // Cascade of 2 biquads, transposed direct form II
    for (int s = 0; s < 2; ++s) {
        const float b0 = HP_SOS[s][0], b1 = HP_SOS[s][1], b2 = HP_SOS[s][2];
        const float a1 = HP_SOS[s][3], a2 = HP_SOS[s][4];
        float z1 = 0.0f, z2 = 0.0f;
        for (int i = 0; i < n; ++i) {
            float in = x[i];
            float out = b0 * in + z1;
            z1 = b1 * in - a1 * out + z2;
            z2 = b2 * in - a2 * out;
            x[i] = out;
        }
    }
}

void dsp_peak_normalize(float* x, int n) {
    float peak = 0.0f;
    for (int i = 0; i < n; ++i) {
        float a = fabsf(x[i]);
        if (a > peak) peak = a;
    }
    if (peak > 0.0f) {
        const float inv = 1.0f / peak;
        for (int i = 0; i < n; ++i) x[i] *= inv;
    }
}

// Returns the start sample of the highest-energy `win`-sample window.
// If n <= win, copies x with zero-padding centered.
int dsp_max_energy_crop(const float* x, int n, int win, float* out) {
    if (n <= win) {
        int pad = win - n;
        int l = pad / 2;
        for (int i = 0; i < l; ++i) out[i] = 0.0f;
        for (int i = 0; i < n; ++i) out[l + i] = x[i];
        for (int i = l + n; i < win; ++i) out[i] = 0.0f;
        return 0;
    }
    // 25 ms smoothing window for short-time energy
    const int k = (int)(0.025f * DSCNN_SAMPLE_RATE);  // ~83
    // Compute squared signal and 1D box-smooth via running sum.
    float sum = 0.0f;
    // Cumulative energy  c[i] = sum_{j<i} x[j]^2
    // We avoid allocating: compute window energies in-place via sliding sum.
    // First pass: window energy at each start.
    float win_e_max = -1.0f; int best_start = 0;
    // Build initial sliding-energy of 'win' samples
    float win_sum = 0.0f;
    for (int i = 0; i < win; ++i) win_sum += x[i] * x[i];
    win_e_max = win_sum; best_start = 0;
    for (int i = win; i < n; ++i) {
        win_sum += x[i] * x[i] - x[i - win] * x[i - win];
        if (win_sum > win_e_max) { win_e_max = win_sum; best_start = i - win + 1; }
    }
    (void)k;  // smoothing isn't strictly needed for argmax of 2s window
    for (int i = 0; i < win; ++i) out[i] = x[best_start + i];
    return best_start;
}

void dsp_log_standardize(float* m, int n_mels, int t) {
    // (m - mean) / std, computed over the full (n_mels x t) tensor
    int n = n_mels * t;
    double sum = 0.0, sumsq = 0.0;
    for (int i = 0; i < n; ++i) {
        float v = logf(m[i] + 1e-6f);
        m[i] = v;
        sum += v; sumsq += (double)v * v;
    }
    double mean = sum / n;
    double var  = sumsq / n - mean * mean;
    if (var < 0) var = 0;
    float inv_std = 1.0f / (sqrtf((float)var) + 1e-5f);
    for (int i = 0; i < n; ++i) m[i] = (m[i] - (float)mean) * inv_std;
}

// ─── log-mel main entry ───
int dsp_logmel(const int16_t* pcm, int n_in,
                float* logmel_out, float* scratch, int n_scratch) {
    // Scratch layout (compact):
    //   When n_in <= WINDOW_SAMPLES (steady-state path): wav aliases hp_in,
    //   so we only need WINDOW + N_FFT + (N_FFT/2+1) floats. Saves ~26 KB.
    //   When n_in > WINDOW_SAMPLES we still allocate a separate wav for the
    //   max-energy crop output. Pipeline always feeds WINDOW_SAMPLES, so the
    //   small layout is the hot path.
    int n_proc = n_in;
    if (n_proc > 2 * WINDOW_SAMPLES) n_proc = 2 * WINDOW_SAMPLES;

    const int alias_wav = (n_proc <= WINDOW_SAMPLES);
    int need = (alias_wav ? WINDOW_SAMPLES
                          : n_proc + WINDOW_SAMPLES)
             + DSCNN_N_FFT + (DSCNN_N_FFT / 2 + 1);
    if (n_scratch < need) return -1;

    float* hp_in = scratch;
    float* wav   = alias_wav ? hp_in : (scratch + n_proc);
    float* frame = (alias_wav ? scratch + WINDOW_SAMPLES
                              : wav + WINDOW_SAMPLES);
    float* power = frame + DSCNN_N_FFT;

    // 1. int16 → float, walk backward so this is safe when pcm aliases hp_in
    // (callers may overlay the int16 PCM buffer onto the float scratch via a
    // union to save SRAM). Since sizeof(float)>sizeof(int16_t) and we walk i
    // descending, every read of pcm[i] precedes the write of hp_in[i] and the
    // hp_in[j>i] writes only ever touch addresses above pcm[i].
    for (int i = n_proc - 1; i >= 0; --i) {
        hp_in[i] = (float)pcm[i] * (1.0f / 32768.0f);
    }

    // 2. HP filter @ 80 Hz
    dsp_hp_filter_inplace(hp_in, n_proc);

    // 2b. Quiet-floor gate: peak-normalize destroys amplitude info, so without
    // this guard pure idle noise gets blown up to ±1.0 and the model fires on
    // every window. Reject windows whose post-HP peak is below the floor.
    {
        // Combined peak + RMS gate. Peak alone is fooled by a single spike of
        // electrical noise on the SPI line; RMS catches sustained low-level
        // hum that the peak gate also misses. Both must clear thresholds.
        float peak = 0.0f;
        double sumsq = 0.0;
        for (int i = 0; i < n_proc; ++i) {
            float v = hp_in[i];
            float a = fabsf(v);
            if (a > peak) peak = a;
            sumsq += (double)v * v;
        }
        float rms = sqrtf((float)(sumsq / (double)n_proc));
        s_last_peak = peak;
        s_last_rms  = rms;
        if (peak < DSP_QUIET_FLOOR || rms < DSP_RMS_FLOOR) {
            LOG_DBG("quiet: peak=%.5f rms=%.5f", (double)peak, (double)rms);
            return 1;  // positive = silent, not error
        }
        LOG_DBG("active: peak=%.5f rms=%.5f", (double)peak, (double)rms);
    }

    // 3. Peak normalize
    dsp_peak_normalize(hp_in, n_proc);

    // 4. Max-energy 2.0s crop. When n_proc == WINDOW the crop is the identity
    // and wav already aliases hp_in, so skip the redundant memcpy.
    if (!(alias_wav && n_proc == WINDOW_SAMPLES)) {
        dsp_max_energy_crop(hp_in, n_proc, WINDOW_SAMPLES, wav);
    }

    // 5. Naive RDFT with cached Hann + twiddles. N=133 is not pow-2 so we
    // cannot use arm_rfft_fast_f32 directly without rebuilding the filterbank.
    const int N    = DSCNN_N_FFT;
    const int HOP_ = DSCNN_HOP;
    const int T    = (WINDOW_SAMPLES - N) / HOP_ + 1;
    const int NB   = N / 2 + 1;

    static float  s_hann[DSCNN_N_FFT];
    static float  s_tw_c[DSCNN_N_FFT / 2 + 1];   // cos(2πk/N) per bin
    static float  s_tw_s[DSCNN_N_FFT / 2 + 1];   // -sin(2πk/N) per bin (FFT sign)
    static int    s_tables_init = 0;
    if (!s_tables_init) {
        for (int i = 0; i < N; ++i) {
            s_hann[i] = 0.5f - 0.5f * cosf(2.0f * (float)M_PI * i / (N - 1));
        }
        for (int k = 0; k < NB; ++k) {
            float a = 2.0f * (float)M_PI * (float)k / (float)N;
            s_tw_c[k] =  cosf(a);
            s_tw_s[k] = -sinf(a);  // FFT sign convention: e^(-j2πkn/N)
        }
        s_tables_init = 1;
    }

    for (int t = 0; t < T; ++t) {
        const int s0 = t * HOP_;
        for (int i = 0; i < N; ++i) frame[i] = wav[s0 + i] * s_hann[i];

        // |DFT|² via angle-recurrence twiddles. For bin k, twiddle steps by
        // (dc, ds) = (cos(2πk/N), -sin(2πk/N)) starting from (1, 0).
        for (int k = 0; k < NB; ++k) {
            const float dc = s_tw_c[k];
            const float ds = s_tw_s[k];
            float c = 1.0f, s = 0.0f;
            float re = 0.0f, im = 0.0f;
            for (int n = 0; n < N; ++n) {
                const float xn = frame[n];
                re += xn * c;
                im += xn * s;
                const float c_new = c * dc - s * ds;
                const float s_new = c * ds + s * dc;
                c = c_new; s = s_new;
            }
            power[k] = re * re + im * im;
        }

        // Apply mel filterbank: logmel_out[mel_idx, t] = sum_k FB[m,k] * power[k]
        for (int m = 0; m < DSCNN_N_MELS; ++m) {
            float sum_m = 0.0f;
            const float* fb = &MEL_FILTERBANK[m * MEL_N_BINS];
            for (int k = 0; k < NB; ++k) sum_m += fb[k] * power[k];
            logmel_out[m * T + t] = sum_m;
        }
    }

    // 6. log + per-clip standardize
    dsp_log_standardize(logmel_out, DSCNN_N_MELS, T);
    return 0;
}
