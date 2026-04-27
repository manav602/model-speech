// DSP front-end implementation. CMSIS-DSP for FFT, hand-rolled for the rest.
#include "dsp.h"
#include "../model/mel_filterbank.h"
#include <math.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

#ifdef USE_CMSIS_DSP
  #include "arm_math.h"
  static arm_rfft_fast_instance_f32 g_rfft;
  static int g_rfft_init = 0;
#else
  // Naive radix-2 / radix-3 RFFT — slow but works for benchmarking on host
  #include <complex.h>
#endif

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
    // Layout of scratch:
    //   [0 .. WINDOW_SAMPLES)              : preprocessed waveform
    //   [WINDOW_SAMPLES .. +DSCNN_N_FFT)   : windowed frame
    //   [+DSCNN_N_FFT .. +N_BINS)          : power spectrum
    // Layout: scratch = [hp_in (n_proc) | wav (WINDOW_SAMPLES) | frame (N_FFT) | power]
    // We cap n_in at 2 × WINDOW_SAMPLES for the in-place HP path (saves 100+ KB of static).
    int n_proc = n_in;
    if (n_proc > 2 * WINDOW_SAMPLES) n_proc = 2 * WINDOW_SAMPLES;
    int need = n_proc + WINDOW_SAMPLES + DSCNN_N_FFT + (DSCNN_N_FFT / 2 + 1);
    if (n_scratch < need) return -1;

    float* hp_in = scratch;
    float* wav   = scratch + n_proc;
    float* frame = wav + WINDOW_SAMPLES;
    float* power = frame + DSCNN_N_FFT;

    // 1. int16 → float, in-place into scratch
    for (int i = 0; i < n_proc; ++i) hp_in[i] = (float)pcm[i] / 32768.0f;

    // 2. HP filter @ 80 Hz
    dsp_hp_filter_inplace(hp_in, n_proc);

    // 3. Peak normalize
    dsp_peak_normalize(hp_in, n_proc);

    // 4. Max-energy 2.0s crop
    dsp_max_energy_crop(hp_in, n_proc, WINDOW_SAMPLES, wav);

    // 5. Compute mel power spectrogram, frame by frame
#ifdef USE_CMSIS_DSP
    if (!g_rfft_init) {
        // CMSIS RFFT requires power-of-two length. n_fft=133 is not pow2.
        // Pad to 256 for CMSIS path (overrides DSCNN_N_FFT for FFT only;
        // mel_filterbank.h must be regenerated to match).
        #error "CMSIS-DSP RFFT needs n_fft pow-of-2; rebuild mel filterbank with n_fft=256."
    }
#endif

    // Naive RDFT (works for any N, including non-pow-2). N=133 is not pow-2 so
    // we cannot use arm_rfft_fast_f32 directly without zero-padding + a new
    // mel filterbank. Instead we accelerate the naive path:
    //   * Pre-compute the Hann window once (eliminates ~13k cosf per inference).
    //   * For each frequency bin, generate twiddle factors via angle recurrence
    //     instead of calling cosf/sinf inside the n-loop. This drops trig calls
    //     from O(T·NB·N) ≈ 882k per inference to O(N) once at init.
    //   * Twiddle step per bin k is (cos, -sin)(2π k / N), pre-computed.
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
