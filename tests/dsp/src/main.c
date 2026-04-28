#include <zephyr/ztest.h>
#include <math.h>
#include "dsp.h"

/* Tolerance for float comparisons */
#define FLOAT_TOL 1e-4f

static void assert_float_close(float actual, float expected, float tol,
			       const char *msg)
{
	zassert_true(fabsf(actual - expected) < tol,
		     "%s: got %f expected %f (tol %f)",
		     msg, (double)actual, (double)expected, (double)tol);
}

/* ------------------------------------------------------------------ */
/*  dsp_peak_normalize                                                 */
/* ------------------------------------------------------------------ */

ZTEST(dsp, test_peak_normalize_basic)
{
	float x[] = {1.0f, -2.0f, 0.5f, -0.5f};

	dsp_peak_normalize(x, 4);

	/* Peak was 2.0 so everything should be scaled by 0.5 */
	assert_float_close(x[0], 0.5f, FLOAT_TOL, "x[0]");
	assert_float_close(x[1], -1.0f, FLOAT_TOL, "x[1]");
	assert_float_close(x[2], 0.25f, FLOAT_TOL, "x[2]");
	assert_float_close(x[3], -0.25f, FLOAT_TOL, "x[3]");
}

ZTEST(dsp, test_peak_normalize_already_unit)
{
	float x[] = {0.0f, 1.0f, -1.0f, 0.0f};

	dsp_peak_normalize(x, 4);

	assert_float_close(x[1], 1.0f, FLOAT_TOL, "x[1]");
	assert_float_close(x[2], -1.0f, FLOAT_TOL, "x[2]");
}

ZTEST(dsp, test_peak_normalize_all_zero)
{
	float x[] = {0.0f, 0.0f, 0.0f};

	dsp_peak_normalize(x, 3);

	/* Should remain zero -- no division by zero */
	zassert_equal(x[0], 0.0f);
	zassert_equal(x[1], 0.0f);
	zassert_equal(x[2], 0.0f);
}

/* ------------------------------------------------------------------ */
/*  dsp_hp_filter_inplace                                              */
/* ------------------------------------------------------------------ */

ZTEST(dsp, test_hp_filter_removes_dc)
{
	/* DC signal (constant) should be driven towards zero by HP filter */
	float x[200];

	for (int i = 0; i < 200; i++) {
		x[i] = 1.0f;
	}

	dsp_hp_filter_inplace(x, 200);

	/* After settling, the last samples should be near zero */
	assert_float_close(x[199], 0.0f, 0.05f, "DC should be removed");
}

ZTEST(dsp, test_hp_filter_passes_high_freq)
{
	/*
	 * A high-frequency signal (well above 80 Hz cutoff) should pass
	 * with only minor attenuation after the filter settles.
	 */
	float x[500];
	float freq = 500.0f;
	float fs = 3333.0f;

	for (int i = 0; i < 500; i++) {
		x[i] = sinf(2.0f * (float)M_PI * freq * i / fs);
	}

	dsp_hp_filter_inplace(x, 500);

	/* Check amplitude of last few cycles (after settling) */
	float peak = 0.0f;

	for (int i = 400; i < 500; i++) {
		float a = fabsf(x[i]);

		if (a > peak) {
			peak = a;
		}
	}

	/* Should retain most energy (>0.8 of unit amplitude) */
	zassert_true(peak > 0.8f, "HP filter attenuated 500 Hz too much: peak=%f",
		     (double)peak);
}

/* ------------------------------------------------------------------ */
/*  dsp_max_energy_crop                                                */
/* ------------------------------------------------------------------ */

ZTEST(dsp, test_max_energy_crop_short_input)
{
	/* Input shorter than window -- should zero-pad */
	float x[] = {1.0f, 2.0f, 3.0f};
	float out[6] = {-1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f};

	int start = dsp_max_energy_crop(x, 3, 6, out);

	zassert_equal(start, 0);

	/* Padding on left: (6-3)/2 = 1 zero */
	assert_float_close(out[0], 0.0f, FLOAT_TOL, "left pad");
	assert_float_close(out[1], 1.0f, FLOAT_TOL, "data[0]");
	assert_float_close(out[2], 2.0f, FLOAT_TOL, "data[1]");
	assert_float_close(out[3], 3.0f, FLOAT_TOL, "data[2]");
	assert_float_close(out[4], 0.0f, FLOAT_TOL, "right pad 0");
	assert_float_close(out[5], 0.0f, FLOAT_TOL, "right pad 1");
}

ZTEST(dsp, test_max_energy_crop_finds_energy)
{
	/*
	 * Create a signal with a burst of energy in the middle.
	 * The crop should center on that burst.
	 */
	float x[100];

	for (int i = 0; i < 100; i++) {
		x[i] = 0.0f;
	}

	/* Put energy at indices 60..79 */
	for (int i = 60; i < 80; i++) {
		x[i] = 1.0f;
	}

	float out[30];
	int start = dsp_max_energy_crop(x, 100, 30, out);

	/*
	 * The 30-sample window with the most energy should overlap
	 * the 60..79 region significantly. Best start should be in
	 * the range [50, 70].
	 */
	zassert_true(start >= 50 && start <= 70,
		     "Expected start in [50,70], got %d", start);
}

/* ------------------------------------------------------------------ */
/*  dsp_log_standardize                                                */
/* ------------------------------------------------------------------ */

ZTEST(dsp, test_log_standardize_output_stats)
{
	/*
	 * After log + standardize the output should have approximately
	 * zero mean and unit variance.
	 */
	const int mels = 4;
	const int t = 8;
	float m[4 * 8];

	/* Fill with some positive values (log requires >0) */
	for (int i = 0; i < mels * t; i++) {
		m[i] = 0.1f + 0.01f * i;
	}

	dsp_log_standardize(m, mels, t);

	/* Check mean is near zero */
	double sum = 0.0;

	for (int i = 0; i < mels * t; i++) {
		sum += m[i];
	}

	double mean = sum / (mels * t);

	zassert_true(fabs(mean) < 0.01, "Mean should be ~0, got %f", mean);

	/* Check variance is near 1 */
	double sumsq = 0.0;

	for (int i = 0; i < mels * t; i++) {
		sumsq += (m[i] - mean) * (m[i] - mean);
	}

	double var = sumsq / (mels * t);

	zassert_true(fabs(var - 1.0) < 0.1,
		     "Variance should be ~1, got %f", var);
}

ZTEST_SUITE(dsp, NULL, NULL, NULL, NULL, NULL);
