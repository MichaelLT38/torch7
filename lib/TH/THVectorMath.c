#include "THVectorMath.h"

/* Algorithms: Cephes (S. L. Moshier) single precision expf / tanhf, written
 * with GCC vector extensions for 8 floats at a time and compiled twice
 * (generic, and target("avx2,fma")), chosen at run time. A chunk of 8 that
 * contains a lane outside the fast path's range (or NaN) is computed with
 * libm instead, so special values behave exactly as before. */

typedef float THv8f __attribute__((vector_size(32)));
typedef int32_t THv8i __attribute__((vector_size(32)));

/* exp fast-path range: keeps 2^n a normal number and the result finite */
#define THVM_EXP_LO (-87.0f)
#define THVM_EXP_HI (88.0f)

static int THVectorMath_hasAvx2Fma(void)
{
  static int cached = -1;
  if (cached < 0) {
#if defined(__x86_64__) || defined(__i386__)
    __builtin_cpu_init();
    cached = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
    /* TH_NO_AVX2=1 forces the generic kernels (used by the tests) */
    const char *off = getenv("TH_NO_AVX2");
    if (off && strcmp(off, "1") == 0) cached = 0;
#else
    cached = 0;
#endif
  }
  return cached;
}

/* For x < -20, sigmoid(x) = e^x / (1 + e^x) equals e^x to float precision;
   computing it that way also keeps tiny results below e^-88.7 (where expf(-x)
   overflows and 1/(1+inf) would give 0) instead of flushing them to zero. */
static inline float THVectorMath_sigmoidScalar(float x)
{
  return x < -20.0f ? expf(x) : 1.0f / (1.0f + expf(-x));
}

/* ---- bodies, instantiated per target ------------------------------------ */

/* all lanes true? (vector comparisons give -1 / 0 per lane) */
#define THVM_ALL(m) (((m)[0] & (m)[1] & (m)[2] & (m)[3] & (m)[4] & (m)[5] & (m)[6] & (m)[7]) != 0)

/* e^v for lanes known to be in [THVM_EXP_LO, THVM_EXP_HI] */
#define THVM_EXP8(v, out)                                                        \
  do {                                                                           \
    THv8f fx_ = (v) * 1.44269504088896341f + 0.5f;     /* x / ln2 + 1/2 */       \
    THv8i n_ = __builtin_convertvector(fx_, THv8i);     /* trunc toward 0 */      \
    THv8f nf_ = __builtin_convertvector(n_, THv8f);                              \
    n_ -= (THv8i)(nf_ > fx_) & 1;                       /* -> floor */            \
    nf_ = __builtin_convertvector(n_, THv8f);                                    \
    THv8f r_ = (v) - nf_ * 0.693359375f;                /* ln2 = C1 + C2 */       \
    r_ = r_ - nf_ * -2.12194440e-4f;                                             \
    THv8f p_ = 1.9875691500E-4f * r_ + 1.3981999507E-3f;                         \
    p_ = p_ * r_ + 8.3334519073E-3f;                                             \
    p_ = p_ * r_ + 4.1665795894E-2f;                                             \
    p_ = p_ * r_ + 1.6666665459E-1f;                                             \
    p_ = p_ * r_ + 5.0000001201E-1f;                                             \
    p_ = p_ * (r_ * r_) + r_ + 1.0f;                                             \
    THv8i e_ = (n_ + 127) << 23;                        /* 2^n as float bits */   \
    THv8f s_;                                                                    \
    memcpy(&s_, &e_, sizeof(s_));                                                \
    (out) = p_ * s_;                                                             \
  } while (0)

#define THVM_EXP_BODY                                                            \
  int64_t i = 0;                                                                 \
  for (; i + 8 <= n; i += 8) {                                                   \
    THv8f v;                                                                     \
    memcpy(&v, x + i, sizeof(v));                                                \
    THv8i ok = (v >= THVM_EXP_LO) & (v <= THVM_EXP_HI);   /* NaN -> false */      \
    if (!THVM_ALL(ok)) {                                                         \
      for (int j = 0; j < 8; j++) y[i + j] = expf(x[i + j]);                     \
      continue;                                                                  \
    }                                                                            \
    THv8f out;                                                                   \
    THVM_EXP8(v, out);                                                           \
    memcpy(y + i, &out, sizeof(out));                                            \
  }                                                                              \
  for (; i < n; i++) y[i] = expf(x[i]);

#define THVM_SIGMOID_BODY                                                        \
  int64_t i = 0;                                                                 \
  for (; i + 8 <= n; i += 8) {                                                   \
    THv8f v;                                                                     \
    memcpy(&v, x + i, sizeof(v));                                                \
    THv8f m = -v;                                                                \
    THv8i ok = (m >= THVM_EXP_LO) & (m <= THVM_EXP_HI);                          \
    if (!THVM_ALL(ok)) {                                                         \
      for (int j = 0; j < 8; j++) y[i + j] = THVectorMath_sigmoidScalar(x[i + j]); \
      continue;                                                                  \
    }                                                                            \
    THv8f e;                                                                     \
    THVM_EXP8(m, e);                                                             \
    THv8f out = 1.0f / (1.0f + e);                                               \
    memcpy(y + i, &out, sizeof(out));                                            \
  }                                                                              \
  for (; i < n; i++) y[i] = THVectorMath_sigmoidScalar(x[i]);

/* tanh: |x| < 0.625: odd polynomial; otherwise sign(x) * (1 - 2/(e^{2|x|}+1)).
   Fast path for |x| <= 44 (e^{2|x|} stays in range; tanh rounds to +-1 well
   before that). */
#define THVM_TANH_BODY                                                           \
  int64_t i = 0;                                                                 \
  for (; i + 8 <= n; i += 8) {                                                   \
    THv8f v;                                                                     \
    memcpy(&v, x + i, sizeof(v));                                                \
    THv8i vi;                                                                    \
    memcpy(&vi, &v, sizeof(vi));                                                 \
    THv8i signbit = vi & (THv8i){0} + (int32_t)0x80000000;                       \
    THv8i ai = vi & (THv8i){0} + 0x7fffffff;                                     \
    THv8f a;                                                                     \
    memcpy(&a, &ai, sizeof(a));                                                  \
    THv8i ok = (a <= 44.0f);                              /* NaN -> false */      \
    if (!THVM_ALL(ok)) {                                                         \
      for (int j = 0; j < 8; j++) y[i + j] = tanhf(x[i + j]);                    \
      continue;                                                                  \
    }                                                                            \
    /* small |x| */                                                              \
    THv8f z = a * a;    /* both branches work on |x|; the sign is restored */     \
    THv8f ps = -5.70498872745E-3f * z + 2.06390887954E-2f;                       \
    ps = ps * z - 5.37397155531E-2f;                                             \
    ps = ps * z + 1.33314422036E-1f;                                             \
    ps = ps * z - 3.33332819422E-1f;                                             \
    THv8f small = ps * z * a + a;                                                \
    /* large |x| (computed on |x|, sign restored) */                             \
    THv8f e;                                                                     \
    THv8f twoA = a + a;                                                          \
    THVM_EXP8(twoA, e);                                                          \
    THv8f big = 1.0f - 2.0f / (e + 1.0f);                                        \
    THv8i bigi;                                                                  \
    memcpy(&bigi, &big, sizeof(bigi));                                           \
    bigi |= signbit;                                                             \
    THv8i smalli;                                                                \
    memcpy(&smalli, &small, sizeof(smalli));                                     \
    smalli |= signbit;                  /* tanh is odd; keeps tanh(-0) = -0 */    \
    THv8i useSmall = (a < 0.625f);                                               \
    THv8i outi = (useSmall & smalli) | (~useSmall & bigi);                       \
    memcpy(y + i, &outi, sizeof(outi));                                          \
  }                                                                              \
  for (; i < n; i++) y[i] = tanhf(x[i]);

/* ---- instantiation & dispatch ------------------------------------------- */

/* Without AVX2 the scalar libm functions are used: 32-byte vectors without
   AVX compile to slow code (measured: exp 62 ms vs libm 24 ms on 10M floats). */
#define THVM_DEFINE(NAME, BODY, SCALAR)                                          \
  static void THVectorMath_##NAME##_generic(float *y, const float *x, int64_t n) \
  { for (int64_t i = 0; i < n; i++) y[i] = SCALAR(x[i]); }                       \
  THVM_DEFINE_AVX2(NAME, BODY)                                                   \
  void THFloatVectorMath_##NAME(float *y, const float *x, int64_t n)             \
  {                                                                              \
    THVM_DISPATCH_AVX2(NAME);                                                    \
    THVectorMath_##NAME##_generic(y, x, n);                                      \
  }

#if defined(__x86_64__) || defined(__i386__)
# define THVM_DEFINE_AVX2(NAME, BODY)                                            \
  __attribute__((target("avx2,fma")))                                            \
  static void THVectorMath_##NAME##_avx2(float *y, const float *x, int64_t n)    \
  { BODY }
# define THVM_DISPATCH_AVX2(NAME)                                                \
  if (THVectorMath_hasAvx2Fma()) { THVectorMath_##NAME##_avx2(y, x, n); return; }
#else
# define THVM_DEFINE_AVX2(NAME, BODY)
# define THVM_DISPATCH_AVX2(NAME)
#endif

THVM_DEFINE(exp, THVM_EXP_BODY, expf)
THVM_DEFINE(sigmoid, THVM_SIGMOID_BODY, THVectorMath_sigmoidScalar)
THVM_DEFINE(tanh, THVM_TANH_BODY, tanhf)
