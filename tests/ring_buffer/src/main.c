#include <zephyr/ztest.h>
#include "ring_buffer.h"

static struct sample_ring ring;

static void *ring_setup(void)
{
	return NULL;
}

static void ring_before(void *fixture)
{
	ARG_UNUSED(fixture);
	ring_init(&ring);
}

/* --- ring_init --- */

ZTEST(ring_buffer, test_init_head_zero)
{
	zassert_equal(ring_head(&ring), 0);
}

/* --- ring_push / ring_head --- */

ZTEST(ring_buffer, test_push_advances_head)
{
	ring_push(&ring, 42);
	zassert_equal(ring_head(&ring), 1);

	ring_push(&ring, 43);
	zassert_equal(ring_head(&ring), 2);
}

ZTEST(ring_buffer, test_push_stores_value)
{
	ring_push(&ring, 1234);
	zassert_equal(ring.buf[0], 1234);
}

/* --- ring_push_block --- */

ZTEST(ring_buffer, test_push_block)
{
	int16_t data[] = {10, 20, 30, 40, 50};

	ring_push_block(&ring, data, 5);
	zassert_equal(ring_head(&ring), 5);
	zassert_equal(ring.buf[0], 10);
	zassert_equal(ring.buf[4], 50);
}

/* --- ring_snapshot_last: cold-start --- */

ZTEST(ring_buffer, test_snapshot_cold_start)
{
	int16_t dst[10];

	/* Only 3 samples pushed, but requesting 10 -- should fail */
	ring_push(&ring, 1);
	ring_push(&ring, 2);
	ring_push(&ring, 3);
	zassert_false(ring_snapshot_last(&ring, dst, 10));
}

/* --- ring_snapshot_last: basic --- */

ZTEST(ring_buffer, test_snapshot_basic)
{
	/* Push 10 known values */
	for (int16_t i = 0; i < 10; i++) {
		ring_push(&ring, i * 100);
	}

	int16_t dst[5];
	bool ok = ring_snapshot_last(&ring, dst, 5);

	zassert_true(ok);
	/* Last 5 values should be 500..900 */
	zassert_equal(dst[0], 500);
	zassert_equal(dst[1], 600);
	zassert_equal(dst[2], 700);
	zassert_equal(dst[3], 800);
	zassert_equal(dst[4], 900);
}

/* --- ring_snapshot_last: wrap-around --- */

ZTEST(ring_buffer, test_snapshot_wraps)
{
	/* Fill past RING_CAPACITY to force wrap */
	for (uint32_t i = 0; i < RING_CAPACITY + 100; i++) {
		ring_push(&ring, (int16_t)(i & 0x7FFF));
	}

	int16_t dst[4];
	bool ok = ring_snapshot_last(&ring, dst, 4);

	zassert_true(ok);

	/* Should get the last 4 values pushed */
	uint32_t h = ring_head(&ring);

	for (int i = 0; i < 4; i++) {
		int16_t expected = (int16_t)((h - 4 + i) & 0x7FFF);

		zassert_equal(dst[i], expected,
			      "dst[%d]=%d expected=%d", i, dst[i], expected);
	}
}

/* --- ring_push_block + snapshot combo --- */

ZTEST(ring_buffer, test_block_then_snapshot)
{
	int16_t block[8] = {-1, -2, -3, -4, -5, -6, -7, -8};

	ring_push_block(&ring, block, 8);

	int16_t dst[3];
	bool ok = ring_snapshot_last(&ring, dst, 3);

	zassert_true(ok);
	zassert_equal(dst[0], -6);
	zassert_equal(dst[1], -7);
	zassert_equal(dst[2], -8);
}

ZTEST_SUITE(ring_buffer, NULL, ring_setup, ring_before, NULL, NULL);
