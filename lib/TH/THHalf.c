#include "THHalf.h"
#include <stdlib.h>
#include <string.h>
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

/* Conversions use the compiler's IEEE round-to-nearest-even casts. They agree
 * bit for bit with the old NVIDIA-derived bit-twiddling code on every non-NaN
 * value in both directions (checked exhaustively over all 2^32 floats); NaNs
 * keep their sign and payload instead of becoming 0x7fff. */

THHalf TH_float2half(float f)
{
  return (THHalf)f;
}

float TH_half2float(THHalf h)
{
  return (float)h;
}

void TH_halfbits2float(unsigned short* src, float* res)
{
  THHalf h;
  memcpy(&h, src, sizeof(h));
  *res = (float)h;
}

void TH_float2halfbits(float* src, unsigned short* dest)
{
  THHalf h = (THHalf)*src;
  memcpy(dest, &h, sizeof(h));
}

/* ---------------------------------------------------------------------------
 * Bulk conversions. Without F16C in the baseline ISA every scalar conversion
 * is a libgcc call; on CPUs with F16C (every AVX2 CPU) 8 values convert per
 * instruction. Chosen at run time; TH_NO_AVX2=1 forces the scalar loops (as
 * for the other AVX2 kernels). Both paths round to nearest even, so the
 * results are identical apart from NaN payloads.
 * ------------------------------------------------------------------------- */
static int TH_halfUseF16C(void)
{
  static int cached = -1;
  if (cached < 0) {
#if defined(__x86_64__) || defined(__i386__)
    __builtin_cpu_init();
    cached = __builtin_cpu_supports("avx") && __builtin_cpu_supports("f16c");
    const char *off = getenv("TH_NO_AVX2");
    if (off && strcmp(off, "1") == 0) cached = 0;
#else
    cached = 0;
#endif
  }
  return cached;
}

#if defined(__x86_64__) || defined(__i386__)
__attribute__((target("avx,f16c")))
static void TH_halfToFloatF16C(float *dst, const THHalf *src, ptrdiff_t n)
{
  ptrdiff_t i = 0;
  for (; i + 8 <= n; i += 8)
    _mm256_storeu_ps(dst + i, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(src + i))));
  for (; i < n; i++)
    dst[i] = (float)src[i];
}

__attribute__((target("avx,f16c")))
static void TH_floatToHalfF16C(THHalf *dst, const float *src, ptrdiff_t n)
{
  ptrdiff_t i = 0;
  for (; i + 8 <= n; i += 8)
    _mm_storeu_si128((__m128i *)(dst + i),
                     _mm256_cvtps_ph(_mm256_loadu_ps(src + i), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  for (; i < n; i++)
    dst[i] = (THHalf)src[i];
}
#endif

#define TH_HALF_CONVERT_GRAIN 65536

void TH_halfToFloatN(float *dst, const THHalf *src, ptrdiff_t n)
{
  const int f16c = TH_halfUseF16C();
  if (n <= 4 * TH_HALF_CONVERT_GRAIN) {   /* no OpenMP region for small n (gemm converts column by column) */
#if defined(__x86_64__) || defined(__i386__)
    if (f16c) { TH_halfToFloatF16C(dst, src, n); return; }
#endif
    for (ptrdiff_t i = 0; i < n; i++)
      dst[i] = (float)src[i];
    return;
  }
#pragma omp parallel for schedule(static)
  for (ptrdiff_t c = 0; c < n; c += TH_HALF_CONVERT_GRAIN) {
    const ptrdiff_t len = n - c < TH_HALF_CONVERT_GRAIN ? n - c : TH_HALF_CONVERT_GRAIN;
#if defined(__x86_64__) || defined(__i386__)
    if (f16c) { TH_halfToFloatF16C(dst + c, src + c, len); continue; }
#endif
    (void)f16c;
    for (ptrdiff_t i = 0; i < len; i++)
      dst[c + i] = (float)src[c + i];
  }
}

void TH_floatToHalfN(THHalf *dst, const float *src, ptrdiff_t n)
{
  const int f16c = TH_halfUseF16C();
  if (n <= 4 * TH_HALF_CONVERT_GRAIN) {   /* no OpenMP region for small n (gemm converts column by column) */
#if defined(__x86_64__) || defined(__i386__)
    if (f16c) { TH_floatToHalfF16C(dst, src, n); return; }
#endif
    for (ptrdiff_t i = 0; i < n; i++)
      dst[i] = (THHalf)src[i];
    return;
  }
#pragma omp parallel for schedule(static)
  for (ptrdiff_t c = 0; c < n; c += TH_HALF_CONVERT_GRAIN) {
    const ptrdiff_t len = n - c < TH_HALF_CONVERT_GRAIN ? n - c : TH_HALF_CONVERT_GRAIN;
#if defined(__x86_64__) || defined(__i386__)
    if (f16c) { TH_floatToHalfF16C(dst + c, src + c, len); continue; }
#endif
    (void)f16c;
    for (ptrdiff_t i = 0; i < len; i++)
      dst[c + i] = (THHalf)src[c + i];
  }
}

/* ---------------------------------------------------------------------------
 * bfloat16 bulk conversions: plain integer code on the bit patterns, which
 * GCC vectorizes with the baseline ISA (no dispatch needed). Round to nearest
 * even with NaNs kept quiet: identical to GCC's (__bf16) cast on all 2^32
 * floats (checked exhaustively). bfloat16 -> float is exact (a shift).
 * ------------------------------------------------------------------------- */
/* integer views of float / bfloat16 data (may_alias: no strict-aliasing assumptions) */
typedef uint32_t TH_u32_alias __attribute__((may_alias));
typedef uint16_t TH_u16_alias __attribute__((may_alias));

static void TH_bfloat16ToFloatSerial(float *restrict dst, const THBFloat16 *restrict src, ptrdiff_t n)
{
  const TH_u16_alias *s = (const TH_u16_alias *)src;
  TH_u32_alias *d = (TH_u32_alias *)dst;
  for (ptrdiff_t i = 0; i < n; i++)
    d[i] = (uint32_t)s[i] << 16;
}

static void TH_floatToBFloat16Serial(THBFloat16 *restrict dst, const float *restrict src, ptrdiff_t n)
{
  const TH_u32_alias *s = (const TH_u32_alias *)src;
  TH_u16_alias *d = (TH_u16_alias *)dst;
  for (ptrdiff_t i = 0; i < n; i++) {
    const uint32_t u = s[i];
    const uint16_t rounded = (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
    const uint16_t quiet_nan = (uint16_t)((u >> 16) | 0x0040u);
    d[i] = (u & 0x7fffffffu) > 0x7f800000u ? quiet_nan : rounded;
  }
}

void TH_bfloat16ToFloatN(float *dst, const THBFloat16 *src, ptrdiff_t n)
{
  if (n <= 4 * TH_HALF_CONVERT_GRAIN) { TH_bfloat16ToFloatSerial(dst, src, n); return; }
#pragma omp parallel for schedule(static)
  for (ptrdiff_t c = 0; c < n; c += TH_HALF_CONVERT_GRAIN)
    TH_bfloat16ToFloatSerial(dst + c, src + c, n - c < TH_HALF_CONVERT_GRAIN ? n - c : TH_HALF_CONVERT_GRAIN);
}

void TH_floatToBFloat16N(THBFloat16 *dst, const float *src, ptrdiff_t n)
{
  if (n <= 4 * TH_HALF_CONVERT_GRAIN) { TH_floatToBFloat16Serial(dst, src, n); return; }
#pragma omp parallel for schedule(static)
  for (ptrdiff_t c = 0; c < n; c += TH_HALF_CONVERT_GRAIN)
    TH_floatToBFloat16Serial(dst + c, src + c, n - c < TH_HALF_CONVERT_GRAIN ? n - c : TH_HALF_CONVERT_GRAIN);
}
