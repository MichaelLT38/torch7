#ifndef TH_HALF_H
#define TH_HALF_H

#include "THGeneral.h"
#include <stdint.h>
#include <stddef.h>

/* IEEE 754 binary16, using GCC's native _Float16 (GCC >= 12 on x86-64 and
 * AArch64). Arithmetic on it is done in float and rounded once to half, so
 * HalfTensor results equal the FloatTensor results rounded to half. The
 * storage layout (2 bytes) is the same as the old struct { unsigned short x; },
 * so files written by upstream Torch read unchanged. */
typedef _Float16 THHalf;

TH_API void TH_float2halfbits(float*, unsigned short*);
TH_API void TH_halfbits2float(unsigned short*, float*);

TH_API THHalf TH_float2half(float);
TH_API float  TH_half2float(THHalf);

/* n contiguous values; F16C-accelerated where available, OpenMP for large n */
TH_API void TH_halfToFloatN(float *dst, const THHalf *src, ptrdiff_t n);
TH_API void TH_floatToHalfN(THHalf *dst, const float *src, ptrdiff_t n);

#define TH_HALF_ZERO 0x0U
#define TH_HALF_INF  0x7C00U
#define TH_HALF_MAX  65504.0f   /* largest finite half */

#endif
