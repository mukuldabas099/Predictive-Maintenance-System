/*
 * pdm_payload.h -- binary vibration packet (no Arduino dependencies, so it is
 * unit-tested on a PC against src/payload_codec.py; see tests/test_firmware_payload.py).
 *
 * Layout, little-endian (the RA4M1 and x86 are both little-endian):
 *   0  4  'P','D','M','1'
 *   4  2  uint16 sample_rate_hz
 *   6  2  uint16 n_samples
 *   8  4  uint32 t0_ms
 *   12 4  float32 scale_g (g per count)
 *   16 2n int16 samples (raw counts)
 */
#ifndef PDM_PAYLOAD_H
#define PDM_PAYLOAD_H

#include <stdint.h>
#include <string.h>
#include <stddef.h>

#define PDM_HEADER_BYTES 16

static inline void pdm_put_u16(uint8_t* p, uint16_t v) { p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); }
static inline void pdm_put_u32(uint8_t* p, uint32_t v) {
  p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); p[2] = (uint8_t)(v >> 16); p[3] = (uint8_t)(v >> 24);
}

/* Writes header + samples into buf. Returns total bytes, or 0 if cap too small. */
static inline size_t pdm_pack_window(uint8_t* buf, size_t cap, const int16_t* counts,
                                     uint16_t n, uint16_t sample_rate_hz,
                                     uint32_t t0_ms, float scale_g) {
  size_t total = PDM_HEADER_BYTES + (size_t)n * 2;
  if (cap < total) return 0;
  buf[0] = 'P'; buf[1] = 'D'; buf[2] = 'M'; buf[3] = '1';
  pdm_put_u16(buf + 4, sample_rate_hz);
  pdm_put_u16(buf + 6, n);
  pdm_put_u32(buf + 8, t0_ms);
  uint32_t scale_bits;
  memcpy(&scale_bits, &scale_g, 4);
  pdm_put_u32(buf + 12, scale_bits);
  for (uint16_t i = 0; i < n; i++) pdm_put_u16(buf + PDM_HEADER_BYTES + 2 * i, (uint16_t)counts[i]);
  return total;
}

#endif
