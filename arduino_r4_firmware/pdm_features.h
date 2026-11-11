/*
 * pdm_features.h -- time-domain statistics computed ON the Arduino from one
 * window of raw int16 counts (no Arduino dependencies, so it is unit-tested on
 * a PC against NumPy/SciPy; see tests/test_pipeline.py).
 *
 * Two-pass algorithm (mean first, then central moments) -- numerically safe in
 * float32, unlike the sum-of-powers shortcut. DC (gravity) is removed by using
 * the window mean, exactly like the Python bridge.
 */
#ifndef PDM_FEATURES_H
#define PDM_FEATURES_H

#include <stdint.h>
#include <math.h>

typedef struct {
  float rms;        // g, DC removed
  float p2p;        // g
  float crest;      // peak / rms
  float kurtosis;   // excess kurtosis (Fisher), 0 for a Gaussian
  float skewness;
} PdmStats;

static inline PdmStats pdm_compute_stats(const int16_t* c, uint16_t n, float scale_g) {
  PdmStats s = {0, 0, 0, 0, 0};
  if (n < 4) return s;
  double sum = 0;
  int16_t mn = c[0], mx = c[0];
  for (uint16_t i = 0; i < n; i++) { sum += c[i]; if (c[i] < mn) mn = c[i]; if (c[i] > mx) mx = c[i]; }
  const double mean = sum / n;
  double m2 = 0, m3 = 0, m4 = 0, peak = 0;
  for (uint16_t i = 0; i < n; i++) {
    const double d = c[i] - mean;
    const double d2 = d * d;
    m2 += d2; m3 += d2 * d; m4 += d2 * d2;
    if (fabs(d) > peak) peak = fabs(d);
  }
  m2 /= n; m3 /= n; m4 /= n;
  const double rms = sqrt(m2);
  s.rms = (float)(rms * scale_g);
  s.p2p = (float)((mx - mn) * (double)scale_g);
  if (rms > 1e-9) {
    s.crest = (float)(peak / rms);
    s.kurtosis = (float)(m4 / (m2 * m2) - 3.0);
    s.skewness = (float)(m3 / (m2 * rms));
  }
  return s;
}

#endif
