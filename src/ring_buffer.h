// Single-producer / single-consumer lock-free ring of int16 samples.
//
// Producer = IMU SPI ISR (drains FIFO into the ring).
// Consumer = main thread (snapshots most-recent N samples for inference).
//
// Sized so the ring holds strictly more than one inference window so the
// snapshot can never race the producer through the wrap point.
#pragma once

#include <stdint.h>
#include <stdbool.h>
#include <zephyr/sys/atomic.h>

#define RING_CAPACITY 8192   // power of 2, > DSCNN_WINDOW_SAMPLES (=6666)

struct sample_ring {
    int16_t  buf[RING_CAPACITY];
    atomic_t head;   // producer writes here, monotonically increasing
    atomic_t tail;   // consumer reads from here (informational only)
};

static inline void ring_init(struct sample_ring* r) {
    atomic_set(&r->head, 0);
    atomic_set(&r->tail, 0);
}

// ISR-safe push (single producer assumption).
static inline void ring_push(struct sample_ring* r, int16_t s) {
    uint32_t h = (uint32_t)atomic_get(&r->head);
    r->buf[h & (RING_CAPACITY - 1)] = s;
    atomic_set(&r->head, (atomic_val_t)(h + 1));
}

static inline void ring_push_block(struct sample_ring* r,
                                   const int16_t* src, uint32_t n) {
    uint32_t h = (uint32_t)atomic_get(&r->head);
    for (uint32_t i = 0; i < n; ++i) {
        r->buf[(h + i) & (RING_CAPACITY - 1)] = src[i];
    }
    atomic_set(&r->head, (atomic_val_t)(h + n));
}

// Returns the absolute head index; consumer uses this to detect "new data
// since last snapshot" without depending on tail.
static inline uint32_t ring_head(const struct sample_ring* r) {
    return (uint32_t)atomic_get((atomic_t*)&r->head);
}

// Snapshot the most-recent `n` samples into `dst`. Returns false if the
// producer hasn't yet written `n` samples total (cold-start).
static inline bool ring_snapshot_last(const struct sample_ring* r,
                                      int16_t* dst, uint32_t n) {
    uint32_t h = ring_head(r);
    if (h < n) return false;
    uint32_t start = h - n;
    for (uint32_t i = 0; i < n; ++i) {
        dst[i] = r->buf[(start + i) & (RING_CAPACITY - 1)];
    }
    // Validate the producer didn't lap us mid-copy. With RING_CAPACITY ≈
    // 2.5× window, a lap requires the ISR to write a full window's worth
    // of samples while we're copying — at 3.3 kHz that's 2 s, while the
    // memcpy runs in well under 1 ms. Still, be defensive.
    uint32_t h2 = ring_head(r);
    return (h2 - start) <= RING_CAPACITY;
}
