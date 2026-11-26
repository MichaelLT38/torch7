#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THBlas.c"
#else


#ifdef BLAS_F2C
# define ffloat double
#else
# define ffloat float
#endif

TH_EXTERNC void dswap_(int *n, double *x, int *incx, double *y, int *incy);
TH_EXTERNC void sswap_(int *n, float *x, int *incx, float *y, int *incy);
TH_EXTERNC void dscal_(int *n, double *a, double *x, int *incx);
TH_EXTERNC void sscal_(int *n, float *a, float *x, int *incx);
TH_EXTERNC void dcopy_(int *n, double *x, int *incx, double *y, int *incy);
TH_EXTERNC void scopy_(int *n, float *x, int *incx, float *y, int *incy);
TH_EXTERNC void daxpy_(int *n, double *a, double *x, int *incx, double *y, int *incy);
TH_EXTERNC void saxpy_(int *n, float *a, float *x, int *incx, float *y, int *incy);
TH_EXTERNC double ddot_(int *n, double *x, int *incx, double *y, int *incy);
#ifdef BLAS_USE_CBLAS_DOT
TH_EXTERNC float cblas_sdot(const int n, const float *x, const int incx, const float *y, const int incy);
#ifndef THBlas_C_sdot_
#define THBlas_C_sdot_
inline ffloat sdot_(const int *n, const float *x, const int *incx, const float *y, const int *incy)
{
  return cblas_sdot(*n, x, *incx, y, *incy);
}
#endif
#else
TH_EXTERNC ffloat sdot_(int *n, float *x, int *incx, float *y, int *incy);
#endif
TH_EXTERNC void dgemv_(char *trans, int *m, int *n, double *alpha, double *a, int *lda, double *x, int *incx, double *beta, double *y, int *incy);
TH_EXTERNC void sgemv_(char *trans, int *m, int *n, float *alpha, float *a, int *lda, float *x, int *incx, float *beta, float *y, int *incy);
TH_EXTERNC void dger_(int *m, int *n, double *alpha, double *x, int *incx, double *y, int *incy, double *a, int *lda);
TH_EXTERNC void sger_(int *m, int *n, float *alpha, float *x, int *incx, float *y, int *incy, float *a, int *lda);
TH_EXTERNC void dgemm_(char *transa, char *transb, int *m, int *n, int *k, double *alpha, double *a, int *lda, double *b, int *ldb, double *beta, double *c, int *ldc);
TH_EXTERNC void sgemm_(char *transa, char *transb, int *m, int *n, int *k, float *alpha, float *a, int *lda, float *b, int *ldb, float *beta, float *c, int *ldc);



#ifdef TH_REAL_IS_REDUCED
/* ---------------------------------------------------------------------------
 * Reduced-precision types (half, bfloat16) accumulate in float, like PyTorch:
 * dot and gemv keep float running sums, and gemm converts op(A), op(B) and C
 * to float, runs the float GEMM (built-in kernel or BLAS) and rounds C once.
 * So a HalfTensor product equals the FloatTensor product of the same values,
 * rounded to half.
 * ------------------------------------------------------------------------- */
/* contiguous bulk conversions to and from float */
#undef THBLAS_REDUCED_TO_FLOAT
#undef THBLAS_FLOAT_TO_REDUCED
#if defined(TH_REAL_IS_HALF)
#define THBLAS_REDUCED_TO_FLOAT(dst, src, n) TH_halfToFloatN((dst), (src), (n))
#define THBLAS_FLOAT_TO_REDUCED(dst, src, n) TH_floatToHalfN((dst), (src), (n))
#else
#define THBLAS_REDUCED_TO_FLOAT(dst, src, n) \
  do { for (int64_t q_ = 0; q_ < (n); q_++) (dst)[q_] = (float)(src)[q_]; } while (0)
#define THBLAS_FLOAT_TO_REDUCED(dst, src, n) \
  do { for (int64_t q_ = 0; q_ < (n); q_++) (dst)[q_] = (real)(src)[q_]; } while (0)
#endif

static accreal THBlas_(dotReduced)(int64_t n, const real *x, int64_t incx, const real *y, int64_t incy)
{
  accreal sum = 0;
  for (int64_t i = 0; i < n; i++)
    sum += (accreal)x[i*incx] * (accreal)y[i*incy];
  return sum;
}

static void THBlas_(gemvReduced)(int trans, int64_t m, int64_t n, real alpha, const real *a, int64_t lda,
                                 const real *x, int64_t incx, real beta, real *y, int64_t incy)
{
  const accreal fa = (accreal)alpha, fb = (accreal)beta;
  if (trans) {
    for (int64_t i = 0; i < n; i++) {
      accreal s = 0;
      const real *row = a + lda * i;
      for (int64_t j = 0; j < m; j++)
        s += (accreal)row[j] * (accreal)x[j*incx];
      y[i*incy] = (real)(fb == 0 ? fa * s : fb * (accreal)y[i*incy] + fa * s);
    }
  } else {
    accreal *acc = THAlloc(sizeof(accreal) * (m > 0 ? m : 1));
    for (int64_t i = 0; i < m; i++) acc[i] = 0;
    for (int64_t j = 0; j < n; j++) {
      const accreal z = (accreal)x[j*incx];
      const real *col = a + lda * j;
      for (int64_t i = 0; i < m; i++)
        acc[i] += z * (accreal)col[i];
    }
    for (int64_t i = 0; i < m; i++)
      y[i*incy] = (real)(fb == 0 ? fa * acc[i] : fb * (accreal)y[i*incy] + fa * acc[i]);
    THFree(acc);
  }
}

static void THBlas_(gemmReduced)(int transa, int transb, int64_t m, int64_t n, int64_t k, real alpha,
                                 const real *a, int64_t lda, const real *b, int64_t ldb,
                                 real beta, real *c, int64_t ldc)
{
  if (m <= 0 || n <= 0)
    return;
  /* A and B are converted as stored (column by column, packed), and the
     transposition flags go to the float GEMM unchanged. */
  const int64_t ra = transa ? k : m, ca = transa ? m : k;   /* stored A: ra x ca */
  const int64_t rb = transb ? n : k, cb = transb ? k : n;   /* stored B: rb x cb */
  const int64_t ldaf = ra > 0 ? ra : 1, ldbf = rb > 0 ? rb : 1;
  float *af = THAlloc(sizeof(float) * (ldaf * ca > 0 ? ldaf * ca : 1));
  float *bf = THAlloc(sizeof(float) * (ldbf * cb > 0 ? ldbf * cb : 1));
  float *cf = THAlloc(sizeof(float) * m * n);
  const float fb = (float)beta;
  for (int64_t j = 0; j < ca; j++)
    THBLAS_REDUCED_TO_FLOAT(af + j*ldaf, a + j*lda, ra);
  for (int64_t j = 0; j < cb; j++)
    THBLAS_REDUCED_TO_FLOAT(bf + j*ldbf, b + j*ldb, rb);
  for (int64_t j = 0; j < n; j++) {
    if (fb == 0) {
      for (int64_t i = 0; i < m; i++) cf[i + j*m] = 0.0f;
    } else {
      THBLAS_REDUCED_TO_FLOAT(cf + j*m, c + j*ldc, m);
    }
  }
  THFloatBlas_gemm(transa ? 't' : 'n', transb ? 't' : 'n', m, n, k, (float)alpha,
                   af, ldaf, bf, ldbf, fb, cf, m);
  for (int64_t j = 0; j < n; j++)
    THBLAS_FLOAT_TO_REDUCED(c + j*ldc, cf + j*m, m);
  THFree(af);
  THFree(bf);
  THFree(cf);
}
#endif /* TH_REAL_IS_REDUCED */

void THBlas_(swap)(int64_t n, real *x, int64_t incx, real *y, int64_t incy)
{
  if(n == 1)
  {
    incx = 1;
    incy = 1;
  }

#if defined(USE_BLAS) && (defined(TH_REAL_IS_DOUBLE) || defined(TH_REAL_IS_FLOAT))
  if( (n <= INT_MAX) && (incx <= INT_MAX) && (incy <= INT_MAX) )
  {
    int i_n = (int)n;
    int i_incx = (int)incx;
    int i_incy = (int)incy;

#if defined(TH_REAL_IS_DOUBLE)
    dswap_(&i_n, x, &i_incx, y, &i_incy);
#else
    sswap_(&i_n, x, &i_incx, y, &i_incy);
#endif
    return;
  }
#endif
  {
    int64_t i;
    for(i = 0; i < n; i++)
    {
      real z = x[i*incx];
      x[i*incx] = y[i*incy];
      y[i*incy] = z;
    }
  }
}

void THBlas_(scal)(int64_t n, real a, real *x, int64_t incx)
{
  if(n == 1)
    incx = 1;

#if defined(USE_BLAS) && (defined(TH_REAL_IS_DOUBLE) || defined(TH_REAL_IS_FLOAT))
  if( (n <= INT_MAX) && (incx <= INT_MAX) )
  {
    int i_n = (int)n;
    int i_incx = (int)incx;

#if defined(TH_REAL_IS_DOUBLE)
    dscal_(&i_n, &a, x, &i_incx);
#else
    sscal_(&i_n, &a, x, &i_incx);
#endif
    return;
  }
#endif
  {
    int64_t i;
    for(i = 0; i < n; i++) {
      if (a == 0) {
        x[i*incx] = 0;
      } else {
        x[i*incx] *= a;
      }
    }
  }
}

void THBlas_(copy)(int64_t n, real *x, int64_t incx, real *y, int64_t incy)
{
  if(n == 1)
  {
    incx = 1;
    incy = 1;
  }

#if defined(USE_BLAS) && (defined(TH_REAL_IS_DOUBLE) || defined(TH_REAL_IS_FLOAT))
  if( (n <= INT_MAX) && (incx <= INT_MAX) && (incy <= INT_MAX) )
  {
    int i_n = (int)n;
    int i_incx = (int)incx;
    int i_incy = (int)incy;

#if defined(TH_REAL_IS_DOUBLE)
    dcopy_(&i_n, x, &i_incx, y, &i_incy);
#else
    scopy_(&i_n, x, &i_incx, y, &i_incy);
#endif
    return;
  }
#endif
  {
    int64_t i;
    for(i = 0; i < n; i++)
      y[i*incy] = x[i*incx];
  }
}

void THBlas_(axpy)(int64_t n, real a, real *x, int64_t incx, real *y, int64_t incy)
{
  if(n == 1)
  {
    incx = 1;
    incy = 1;
  }

#if defined(USE_BLAS) && (defined(TH_REAL_IS_DOUBLE) || defined(TH_REAL_IS_FLOAT))
  if( (n <= INT_MAX) && (incx <= INT_MAX) && (incy <= INT_MAX) )
  {
    int i_n = (int)n;
    int i_incx = (int)incx;
    int i_incy = (int)incy;

#if defined(TH_REAL_IS_DOUBLE)
    daxpy_(&i_n, &a, x, &i_incx, y, &i_incy);
#else
    saxpy_(&i_n, &a, x, &i_incx, y, &i_incy);
#endif
    return;
  }
#endif
  {
    int64_t i;
    for(i = 0; i < n; i++)
      y[i*incy] += a*x[i*incx];
  }
}

real THBlas_(dot)(int64_t n, real *x, int64_t incx, real *y, int64_t incy)
{
#ifdef TH_REAL_IS_REDUCED
  return (real)THBlas_(dotReduced)(n, x, incx, y, incy);
#endif
  if(n == 1)
  {
    incx = 1;
    incy = 1;
  }

#if defined(USE_BLAS) && (defined(TH_REAL_IS_DOUBLE) || defined(TH_REAL_IS_FLOAT))
  if( (n <= INT_MAX) && (incx <= INT_MAX) && (incy <= INT_MAX) )
  {
    int i_n = (int)n;
    int i_incx = (int)incx;
    int i_incy = (int)incy;

#if defined(TH_REAL_IS_DOUBLE)
    return (real) ddot_(&i_n, x, &i_incx, y, &i_incy);
#else
    return (real) sdot_(&i_n, x, &i_incx, y, &i_incy);
#endif
  }
#endif
  {
    int64_t i;
    real sum = 0;
    for(i = 0; i < n; i++)
    sum += x[i*incx]*y[i*incy];
    return sum;
  }
}

void THBlas_(gemv)(char trans, int64_t m, int64_t n, real alpha, real *a, int64_t lda, real *x, int64_t incx, real beta, real *y, int64_t incy)
{
  if(n == 1)
    lda = m;
#ifdef TH_REAL_IS_REDUCED
  THBlas_(gemvReduced)((trans == 'T') || (trans == 't'), m, n, alpha, a, lda, x, incx, beta, y, incy);
  return;
#endif

#if defined(USE_BLAS) && (defined(TH_REAL_IS_DOUBLE) || defined(TH_REAL_IS_FLOAT))
  if( (m <= INT_MAX) && (n <= INT_MAX) &&
      (lda > 0) && (lda <= INT_MAX) &&
      (incx > 0) && (incx <= INT_MAX) &&
      (incy > 0) && (incy <= INT_MAX) )
  {
    int i_m = (int)m;
    int i_n = (int)n;
    int i_lda = (int)lda;
    int i_incx = (int)incx;
    int i_incy = (int)incy;

#if defined(TH_REAL_IS_DOUBLE)
    dgemv_(&trans, &i_m, &i_n, &alpha, a, &i_lda, x, &i_incx, &beta, y, &i_incy);
#else
    sgemv_(&trans, &i_m, &i_n, &alpha, a, &i_lda, x, &i_incx, &beta, y, &i_incy);
#endif
    return;
  }
#endif
  {
    int64_t i, j;

    if( (trans == 'T') || (trans == 't') )
    {
      for(i = 0; i < n; i++)
      {
        real sum = 0;
        real *row_ = a+lda*i;
        for(j = 0; j < m; j++)
          sum += x[j*incx]*row_[j];
	if (beta == 0)
	  y[i*incy] = alpha*sum;
	else
	  y[i*incy] = beta*y[i*incy] + alpha*sum;
      }
    }
    else
    {
      if(beta != 1)
        THBlas_(scal)(m, beta, y, incy);

      for(j = 0; j < n; j++)
      {
        real *column_ = a+lda*j;
        real z = alpha*x[j*incx];
        for(i = 0; i < m; i++)
          y[i*incy] += z*column_[i];
      }
    }
  }
}

void THBlas_(ger)(int64_t m, int64_t n, real alpha, real *x, int64_t incx, real *y, int64_t incy, real *a, int64_t lda)
{
  if(n == 1)
    lda = m;

#if defined(USE_BLAS) && (defined(TH_REAL_IS_DOUBLE) || defined(TH_REAL_IS_FLOAT))
  if( (m <= INT_MAX) && (n <= INT_MAX) && (lda <= INT_MAX)  && (incx <= INT_MAX) && (incy <= INT_MAX) )
  {
    int i_m = (int)m;
    int i_n = (int)n;
    int i_lda = (int)lda;
    int i_incx = (int)incx;
    int i_incy = (int)incy;

#if defined(TH_REAL_IS_DOUBLE)
    dger_(&i_m, &i_n, &alpha, x, &i_incx, y, &i_incy, a, &i_lda);
#else
    sger_(&i_m, &i_n, &alpha, x, &i_incx, y, &i_incy, a, &i_lda);
#endif
    return;
  }
#endif
  {
    int64_t i, j;
    for(j = 0; j < n; j++)
    {
      real *column_ = a+j*lda;
#ifdef TH_REAL_IS_REDUCED
      accreal z = (accreal)alpha*(accreal)y[j*incy];   /* not rounded to 16 bits */
#else
      real z = alpha*y[j*incy];
#endif
      for(i = 0; i < m; i++)
        column_[i] += z*x[i*incx] ;
    }
  }
}

#ifndef TH_REAL_IS_REDUCED

/* ---------------------------------------------------------------------------
 * Built-in blocked GEMM (Phase 3): C = beta*C + alpha*op(A)*op(B),
 * column-major like BLAS. Used for floating point when TH is built without
 * BLAS, and always for integer types.
 *
 * Classic packed layout: panels of A (MC x KC) and B (KC x NC) are copied
 * into contiguous micro-panels, and a register-blocked micro-kernel computes
 * an MR x NR tile of C with GCC vector extensions (NR = two 32-byte
 * vectors). The kernel is compiled twice, generic and AVX2+FMA, and chosen
 * at run time (MinGW has no ifunc, so target_clones is not available).
 * Threads split the rows of C; each C element is accumulated in the same
 * order whatever the thread count.
 *
 * Integer types stay bit-exact: wrap-around (two's-complement) addition and
 * multiplication give the same result in any order. Floating-point results
 * differ from the naive loop by rounding only.
 * ------------------------------------------------------------------------- */
#define THGEMM_VL ((int64_t)(THGEMM_VBYTES / sizeof(real)))
#define THGEMM_NR (2 * THGEMM_VL)

typedef real THBlas_(GemmVec) __attribute__((vector_size(THGEMM_VBYTES)));

/* rows [i0, i0+mc) x depth [p0, p0+kc) of op(A), as MR-row micro-panels */
static void THBlas_(gemmPackA)(int transa, const real *a, int64_t lda, int64_t i0, int64_t p0,
                               int64_t mc, int64_t kc, real *buf)
{
  for (int64_t ir = 0; ir < mc; ir += THGEMM_MR) {
    const int64_t mr = mc - ir < THGEMM_MR ? mc - ir : THGEMM_MR;
    for (int64_t p = 0; p < kc; p++) {
      const int64_t l = p0 + p;
      for (int64_t r = 0; r < THGEMM_MR; r++) {
        const int64_t i = i0 + ir + r;
        *buf++ = r < mr ? (transa ? a[l + i * lda] : a[i + l * lda]) : (real)0;
      }
    }
  }
}

/* depth [p0, p0+kc) x columns [j0, j0+nc) of op(B), as NR-column micro-panels */
static void THBlas_(gemmPackB)(int transb, const real *b, int64_t ldb, int64_t p0, int64_t j0,
                               int64_t kc, int64_t nc, real *buf)
{
  for (int64_t jr = 0; jr < nc; jr += THGEMM_NR) {
    const int64_t nr = nc - jr < THGEMM_NR ? nc - jr : THGEMM_NR;
    for (int64_t p = 0; p < kc; p++) {
      const int64_t l = p0 + p;
      for (int64_t cc = 0; cc < THGEMM_NR; cc++) {
        const int64_t j = j0 + jr + cc;
        *buf++ = cc < nr ? (transb ? b[j + l * ldb] : b[l + j * ldb]) : (real)0;
      }
    }
  }
}

/* One packed (mc x kc) x (kc x nc) block into C (c points at C(i0, j0)). */
#define THGEMM_MACRO_KERNEL_BODY                                                 \
  for (int64_t jr = 0; jr < nc; jr += THGEMM_NR) {                               \
    const int64_t nr = nc - jr < THGEMM_NR ? nc - jr : THGEMM_NR;                \
    const real *Bp = Bpack + jr * kc;                                            \
    for (int64_t ir = 0; ir < mc; ir += THGEMM_MR) {                             \
      const int64_t mr = mc - ir < THGEMM_MR ? mc - ir : THGEMM_MR;              \
      const real *Ap = Apack + ir * kc;                                          \
      THBlas_(GemmVec) acc[THGEMM_MR][2];                                        \
      for (int r = 0; r < THGEMM_MR; r++) {                                      \
        acc[r][0] = (THBlas_(GemmVec)){0};                                       \
        acc[r][1] = (THBlas_(GemmVec)){0};                                       \
      }                                                                          \
      for (int64_t p = 0; p < kc; p++) {                                         \
        THBlas_(GemmVec) b0, b1;                                                 \
        memcpy(&b0, Bp + p * THGEMM_NR, THGEMM_VBYTES);                          \
        memcpy(&b1, Bp + p * THGEMM_NR + THGEMM_VL, THGEMM_VBYTES);              \
        for (int r = 0; r < THGEMM_MR; r++) {                                    \
          const THBlas_(GemmVec) av = (THBlas_(GemmVec)){0} + Ap[p * THGEMM_MR + r]; \
          acc[r][0] += av * b0;                                                  \
          acc[r][1] += av * b1;                                                  \
        }                                                                        \
      }                                                                          \
      /* column by column: C is column-major, so the inner loop is contiguous */ \
      for (int64_t cc = 0; cc < nr; cc++)                                        \
        for (int64_t r = 0; r < mr; r++) {                                       \
          const real v = acc[r][cc / THGEMM_VL][cc % THGEMM_VL];                 \
          real *cp = c + (ir + r) + (jr + cc) * ldc;                             \
          *cp = (beta == 0) ? (real)(alpha * v) : (real)(beta * *cp + alpha * v); \
        }                                                                        \
    }                                                                            \
  }

static void THBlas_(gemmMacroGeneric)(int64_t mc, int64_t nc, int64_t kc, real alpha, real beta,
                                      const real *Apack, const real *Bpack, real *c, int64_t ldc)
{
  THGEMM_MACRO_KERNEL_BODY
}

#if defined(__x86_64__) || defined(__i386__)
__attribute__((target("avx2,fma")))
static void THBlas_(gemmMacroAvx2)(int64_t mc, int64_t nc, int64_t kc, real alpha, real beta,
                                   const real *Apack, const real *Bpack, real *c, int64_t ldc)
{
  THGEMM_MACRO_KERNEL_BODY
}
#endif

typedef void (*THBlas_(GemmMacroFn))(int64_t, int64_t, int64_t, real, real,
                                     const real *, const real *, real *, int64_t);

static void THBlas_(gemmBlocked)(int transa, int transb, int64_t m, int64_t n, int64_t k,
                                 real alpha, const real *a, int64_t lda, const real *b, int64_t ldb,
                                 real beta, real *c, int64_t ldc)
{
  if (m <= 0 || n <= 0) return;
  if (k <= 0) {   /* no products: C = beta*C (or alpha*0 when beta == 0) */
    for (int64_t j = 0; j < n; j++)
      for (int64_t i = 0; i < m; i++) {
        real *cp = c + i + j * ldc;
        *cp = (beta == 0) ? (real)(alpha * (real)0) : (real)(beta * *cp + alpha * (real)0);
      }
    return;
  }

  THBlas_(GemmMacroFn) macro = THBlas_(gemmMacroGeneric);
#if defined(__x86_64__) || defined(__i386__)
  if (THBlas_cpuHasAvx2Fma()) macro = THBlas_(gemmMacroAvx2);
#endif

  int nthreads = 1;
#ifdef _OPENMP
  if (!omp_in_parallel() && (double)m * (double)n * (double)k >= 262144.0)
    nthreads = omp_get_max_threads();
#endif
  /* row-block height: split m across threads, a multiple of MR, capped */
  int64_t mc = (m + nthreads - 1) / nthreads;
  mc = ((mc + THGEMM_MR - 1) / THGEMM_MR) * THGEMM_MR;
  if (mc > THGEMM_MC_MAX) mc = THGEMM_MC_MAX;
  if (mc < THGEMM_MR) mc = THGEMM_MR;

  const int64_t ncAlloc = n < THGEMM_NC ? ((n + THGEMM_NR - 1) / THGEMM_NR) * THGEMM_NR : THGEMM_NC;
  real *Bpack = THAlloc(sizeof(real) * THGEMM_KC * ncAlloc);

#pragma omp parallel num_threads(nthreads) if (nthreads > 1)
  {
    real *Apack = THAlloc(sizeof(real) * THGEMM_KC * mc);
    for (int64_t jc = 0; jc < n; jc += THGEMM_NC) {
      const int64_t nc = n - jc < THGEMM_NC ? n - jc : THGEMM_NC;
      for (int64_t pc = 0; pc < k; pc += THGEMM_KC) {
        const int64_t kc = k - pc < THGEMM_KC ? k - pc : THGEMM_KC;
        const real betaEff = pc == 0 ? beta : (real)1;
        /* pack B: its NR-column micro-panels are independent, so split them
           (implicit barrier: everyone then sees the whole packed B) */
#pragma omp for schedule(static)
        for (int64_t jr = 0; jr < nc; jr += THGEMM_NR)
          THBlas_(gemmPackB)(transb, b, ldb, pc, jc + jr, kc,
                             nc - jr < THGEMM_NR ? nc - jr : THGEMM_NR, Bpack + jr * kc);
#pragma omp for schedule(static)
        for (int64_t ic = 0; ic < m; ic += mc) {
          const int64_t mcc = m - ic < mc ? m - ic : mc;
          THBlas_(gemmPackA)(transa, a, lda, ic, pc, mcc, kc, Apack);
          macro(mcc, nc, kc, alpha, betaEff, Apack, Bpack, c + ic + jc * ldc, ldc);
        }
        /* (implicit barrier before B is repacked) */
      }
    }
    THFree(Apack);
  }
  THFree(Bpack);
}

#undef THGEMM_MACRO_KERNEL_BODY
#undef THGEMM_VL
#undef THGEMM_NR
#endif /* TH_REAL_IS_REDUCED */

void THBlas_(gemm)(char transa, char transb, int64_t m, int64_t n, int64_t k, real alpha, real *a, int64_t lda, real *b, int64_t ldb, real beta, real *c, int64_t ldc)
{
  int transa_ = ((transa == 't') || (transa == 'T'));
  int transb_ = ((transb == 't') || (transb == 'T'));

  if(n == 1)
    ldc = m;

  if(transa_)
  {
    if(m == 1)
      lda = k;
  }
  else
  {
    if(k == 1)
      lda = m;
  }

  if(transb_)
  {
    if(k == 1)
      ldb = n;
  }
  else
  {
    if(n == 1)
      ldb = k;
  }

#ifdef TH_REAL_IS_REDUCED
  THBlas_(gemmReduced)(transa_, transb_, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
#else
#if defined(USE_BLAS) && (defined(TH_REAL_IS_DOUBLE) || defined(TH_REAL_IS_FLOAT))
  if( (m <= INT_MAX) && (n <= INT_MAX) && (k <= INT_MAX) && (lda <= INT_MAX)  && (ldb <= INT_MAX) && (ldc <= INT_MAX) )
  {
    int i_m = (int)m;
    int i_n = (int)n;
    int i_k = (int)k;
    int i_lda = (int)lda;
    int i_ldb = (int)ldb;
    int i_ldc = (int)ldc;

#if defined(TH_REAL_IS_DOUBLE)
    dgemm_(&transa, &transb, &i_m, &i_n, &i_k, &alpha, a, &i_lda, b, &i_ldb, &beta, c, &i_ldc);
#else
    sgemm_(&transa, &transb, &i_m, &i_n, &i_k, &alpha, a, &i_lda, b, &i_ldb, &beta, c, &i_ldc);
#endif
    return;
  }
#endif
  if ((double)m * (double)n * (double)k >= THGEMM_MIN_WORK) {
    THBlas_(gemmBlocked)(transa_, transb_, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    return;
  }
  {
    int64_t i, j, l;
    if(!transa_ && !transb_)
    {
      real *a_ = a;
      for(i = 0; i < m; i++)
      {
        real *b_ = b;
        for(j = 0; j < n; j++)
        {
          real sum = 0;
          for(l = 0; l < k; l++)
            sum += a_[l*lda]*b_[l];
          b_ += ldb;
	  if (beta == 0)
	    c[j*ldc+i] = alpha*sum;
	  else
	    c[j*ldc+i] = beta*c[j*ldc+i]+alpha*sum;
        }
        a_++;
      }
    }
    else if(transa_ && !transb_)
    {
      real *a_ = a;
      for(i = 0; i < m; i++)
      {
        real *b_ = b;
        for(j = 0; j < n; j++)
        {
          real sum = 0;
          for(l = 0; l < k; l++)
            sum += a_[l]*b_[l];
          b_ += ldb;
	  if (beta == 0)
	    c[j*ldc+i] = alpha*sum;
	  else
	    c[j*ldc+i] = beta*c[j*ldc+i]+alpha*sum;
        }
        a_ += lda;
      }
    }
    else if(!transa_ && transb_)
    {
      real *a_ = a;
      for(i = 0; i < m; i++)
      {
        real *b_ = b;
        for(j = 0; j < n; j++)
        {
          real sum = 0;
          for(l = 0; l < k; l++)
            sum += a_[l*lda]*b_[l*ldb];
          b_++;
	  if (beta == 0)
	    c[j*ldc+i] = alpha*sum;
	  else
	    c[j*ldc+i] = beta*c[j*ldc+i]+alpha*sum;
        }
        a_++;
      }
    }
    else
    {
      real *a_ = a;
      for(i = 0; i < m; i++)
      {
        real *b_ = b;
        for(j = 0; j < n; j++)
        {
          real sum = 0;
          for(l = 0; l < k; l++)
            sum += a_[l]*b_[l*ldb];
          b_++;
	  if (beta == 0)
	    c[j*ldc+i] = alpha*sum;
	  else
	    c[j*ldc+i] = beta*c[j*ldc+i]+alpha*sum;
        }
        a_ += lda;
      }
    }
  }
#endif /* TH_REAL_IS_REDUCED */
}

#endif
