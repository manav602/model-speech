// BLE transport for KWS detection events.
//
// Service UUID:        a7f0b5c1-1234-4567-89ab-cdef01234567
// Notification UUID:   a7f0b5c2-1234-4567-89ab-cdef01234567
//
// Payload (2 bytes): [keyword_id (0-3)] [confidence 0-100]
#pragma once
#include "kws_pipeline.h"

// Call once after ring/IMU init. Enables BLE stack and starts advertising.
int  ble_kws_init(void);

// Keyword detection: [kw_id 0-3][conf 0-100]
void ble_kws_notify(keyword_t kw, float conf);

// Quiet gate (no motion): sends fixed byte 55. Call on every quiet window.
void ble_kws_notify_quiet(void);

// IMU heartbeat: [isr_count_lo][isr_count_hi]  (lower 16 bits).
void ble_kws_notify_imu(uint32_t isr_count);
