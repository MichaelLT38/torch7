#ifndef TH_VECTOR_MATH_INC
#define TH_VECTOR_MATH_INC

/* Vectorized float transcendentals (Phase 3c), used by FloatTensor exp, tanh,
 * sigmoid and log on contiguous runs. Accuracy: a few ulp of the correctly
 * rounded result (tested to <= 4 ulp against double-precision libm).
 * Special values (NaN, +-inf, overflow/underflow ranges) take the libm path,
 * so their results are unchanged. y and x may be the same array. */

#include "THGeneral.h"

TH_API void THFloatVectorMath_exp(float *y, const float *x, int64_t n);
TH_API void THFloatVectorMath_tanh(float *y, const float *x, int64_t n);
TH_API void THFloatVectorMath_sigmoid(float *y, const float *x, int64_t n);
TH_API void THFloatVectorMath_log(float *y, const float *x, int64_t n);

#endif
