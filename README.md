# Production firmware — nRF52840 + LSM6DSOX KWS

Real-device firmware. No test vectors, no QEMU. Boots, streams Z-axis accel
from an LSM6DSOX at **3.333 kHz** into a lock-free ring buffer, and runs the
DS-CNN-M W8A8 keyword spotter on a sliding 2-second window.

## Hardware

I2C transport (TWIM0 @ 400 kHz, Fast-mode).

| nRF52840-DK | LSM6DSOX |
|---|---|
| P0.27 (SCL)  | SCL  |
| P0.26 (SDA)  | SDA  |
| P0.30 (GPIO) | INT1 |
| 3V3          | VDD/VDDIO |
| GND          | GND  |
| GND          | SA0 (→ I2C addr 0x6A) |

Most LSM6DSOX breakouts ship with on-board 10 kΩ pull-ups on SDA/SCL.
If yours doesn't, add 4.7 kΩ to 3V3 on each line.

If you change pins, edit only `boards/nrf52840dk_nrf52840.overlay`.

## Build & flash (Zephyr / nRF Connect SDK)

```bash
cd deploy
west init -l firmware     # one-time, if not already a Zephyr workspace
west update
west build -b nrf52840dk_nrf52840 firmware
west flash
```

Logs come out over Segger RTT:

```bash
JLinkRTTViewer    # or: west attach
```

## Runtime behavior

- Z-axis accel @ 3.333 kHz, ±4 g full-scale, hardware FIFO with 64-sample watermark.
- Inference fires every 250 ms over the most-recent 2.0 s of samples.
- Output: argmax + softmax confidence; emit only if `conf ≥ 0.70` and class
  is not `unknown`. Debounced to 1 fire / 1.5 s.
- LED0 toggles on each detection; replace `on_keyword()` in `src/main.c`
  with your real downstream (BLE NUS notify, GPIO trigger, MQTT, etc).
- Health log every 30 s: IMU IRQ count, samples pushed, FIFO overruns,
  inference count, average cycles.

## Tunables (`src/main.c`)

| Field | Default | Meaning |
|---|---|---|
| `confidence_threshold` | 0.70 | reject below this softmax max |
| `debounce_ms`          | 1500 | min gap between two emitted detections |
| `slide_ms`             | 250  | inference cadence |
| `IMU_AXIS_Z`           | Z    | axis fed into the model (matches training) |

## Memory

- Sample ring:  32 KB SRAM (16384 × int16, ~4.9 s of audio)
- KWS arena:    137 KB SRAM (W8A8 ping-pong)
- DSP scratch:  46 KB SRAM
- Sample window: 13.3 KB SRAM
- Stack (KWS thread): 4 KB
- **Total: ~233 KB / 256 KB on nRF52840** — leaves ~23 KB for BLE stack etc.

Flash: ~225 KB (model 132 KB INT8 + DSP + Zephyr base ≈ 80 KB).

## Files

```
firmware/
├── CMakeLists.txt           # Zephyr app build
├── prj.conf                 # Zephyr config (FPU, CMSIS-DSP/NN, RTT)
├── boards/
│   └── nrf52840dk_nrf52840.overlay   # SPI + INT1 pin map
└── src/
    ├── main.c               # boot, init, idle health log
    ├── ring_buffer.h        # SPSC lock-free int16 ring
    ├── imu_lsm6dsox.{c,h}   # SPI driver + FIFO IRQ → ring
    └── kws_pipeline.{c,h}   # inference thread, softmax, debounce
```

Reuses `../src/dsp.c`, `../src/kws_int8.c`, `../model/dscnn_m_w8a8.c`,
`../model/mel_filterbank.h` from the model package — those are the same
verified files documented in `deploy/README.md`.

## Critical: must match training

1. **Axis** — Z-only. If training changes, update `IMU_AXIS_Z` in `main.c`.
2. **Sample rate** — exactly 3333 Hz. The HP-filter coefficients in `dsp.c`
   and the mel filterbank in `mel_filterbank.h` are baked for 3333 Hz.
3. **Units** — `dsp.c` divides by 32768.0f, so the IMU must produce raw
   signed int16 LSB. ±4 g range is set in `imu_lsm6dsox.c`; if you change
   it, the training pipeline must change too.
