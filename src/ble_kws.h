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

// Send a keyword detection notification (no-op if no subscriber).
void ble_kws_notify(keyword_t kw, float conf);
