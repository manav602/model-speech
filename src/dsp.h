// DSP front-end for DS-CNN-M KWS
//   raw int16 IMU (3333 Hz) → log-mel spectrogram (n_mels=30, T=101)
#pragma once
#include <stdint.h>
#include "../model/dscnn_m_w8a8.h"

// Sizes (from model header, repeated for clarity)
#define WINDOW_SAMPLES   DSCNN_WINDOW_SAMPLES   // 6666 = 2.0s @ 3333 Hz
#define MEL_T            ((WINDOW_SAMPLES - DSCNN_N_FFT) / DSCNN_HOP + 1)  // ≈ 100

// Post-HP peak/RMS thresholds (normalized [-1,1] units) below which the window
// is treated as silent and inference is skipped. The previous 0.005 peak-only
// floor was too permissive — sustained low-level IMU noise was clearing it.
// Defaults bumped 10× and an RMS gate added.  Tune empirically: log
// `health: ... peak= rms=` while idle and set thresholds 3-5× above observed.
#ifndef DSP_QUIET_FLOOR
#define DSP_QUIET_FLOOR  0.05f      // peak floor
#endif
#ifndef DSP_RMS_FLOOR
#define DSP_RMS_FLOOR    0.015f     // RMS floor (rejects steady-state hum)
#endif

// dsp_logmel return values:
//   0  — features written, run inference
//   1  — window is below quiet floor, treat as no-keyword (skip inference)
//   <0 — error

// Process raw int16 PCM samples -> normalized log-mel features.
// Inputs:
//   pcm:        raw int16 samples (any length n_in)
//   n_in:       number of input samples
//   logmel_out: output buffer, shape [DSCNN_N_MELS x MEL_T] row-major (n_mels rows)
//   scratch:    workspace, must hold at least:
//                 WINDOW_SAMPLES floats         (preprocessed waveform)
//               + DSCNN_N_FFT floats            (FFT input window)
//               + (DSCNN_N_FFT/2+1) floats      (power spectrum)
//   n_scratch:  scratch buffer size in floats
// Returns 0 on success, nonzero on error.
int dsp_logmel(const int16_t* pcm, int n_in,
                float* logmel_out, float* scratch, int n_scratch);

// Internal primitives (exposed for testing)
void dsp_hp_filter_inplace(float* x, int n);
void dsp_peak_normalize(float* x, int n);
int  dsp_max_energy_crop(const float* x, int n, int win, float* out);
void dsp_log_standardize(float* m, int n_mels, int t);

// Most recent post-HP peak / RMS (telemetry / threshold tuning).
float dsp_last_peak(void);
float dsp_last_rms(void);
