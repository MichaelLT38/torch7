#ifndef _THMATH_H
#define _THMATH_H

/* For very negative inputs sigmoid(x) = e^x / (1 + e^x) equals e^x to working
   precision; computing it that way keeps tiny results (where exp(-x) would
   overflow and 1/(1+inf) flush them to 0). */
static inline double TH_sigmoid(double value) {
  return value < -40.0 ? exp(value) : 1.0 / (1.0 + exp(-value));
}

static inline double TH_frac(double x) {
  return x - trunc(x);
}

static inline double TH_rsqrt(double x) {
  return 1.0 / sqrt(x);
}

static inline double TH_lerp(double a, double b, double weight) {
  return a + weight * (b-a);
}

static inline float TH_sigmoidf(float value) {
  return value < -20.0f ? expf(value) : 1.0f / (1.0f + expf(-value));
}

static inline float TH_fracf(float x) {
  return x - truncf(x);
}

static inline float TH_rsqrtf(float x) {
  return 1.0f / sqrtf(x);
}

static inline float TH_lerpf(float a, float b, float weight) {
  return a + weight * (b-a);
}

#endif // _THMATH_H
