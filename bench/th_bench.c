/* Micro-benchmarks for TH kernels.
 *
 *   th_bench [threads ...]     default: 1 and omp_get_max_threads()
 *
 * Prints CSV "op,threads,ms" (median of REPS runs after one warm-up), preceded
 * by '#' lines describing the build. Compare two runs with bench/compare.lua.
 * The same ops are timed in PyTorch by bench/bench_pytorch.py.
 */
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "TH.h"

#ifdef _WIN32
#include <windows.h>
static double now_ms(void)
{
  LARGE_INTEGER f, c;
  QueryPerformanceFrequency(&f);
  QueryPerformanceCounter(&c);
  return 1e3 * (double)c.QuadPart / (double)f.QuadPart;
}
#else
#include <time.h>
static double now_ms(void)
{
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return 1e3 * ts.tv_sec + 1e-6 * ts.tv_nsec;
}
#endif

#define REPS 15

static int cmp_double(const void *a, const void *b)
{
  double x = *(const double *)a, y = *(const double *)b;
  return (x > y) - (x < y);
}

static volatile double sink;

#define BENCH(NAME, CODE)                                                     \
  do {                                                                        \
    double t_[REPS];                                                          \
    CODE; /* warm-up */                                                       \
    for (int r_ = 0; r_ < REPS; r_++) {                                       \
      double t0_ = now_ms();                                                  \
      CODE;                                                                   \
      t_[r_] = now_ms() - t0_;                                                \
    }                                                                         \
    qsort(t_, REPS, sizeof(double), cmp_double);                              \
    printf("%s,%d,%.4f\n", NAME, nt, t_[REPS / 2]);                           \
    fflush(stdout);                                                           \
  } while (0)

static void run(int nt, THGenerator *g)
{
  THSetNumThreads(nt);

  /* 1-D contiguous, 10M floats */
  const long N = 10 * 1000 * 1000;
  THFloatTensor *a = THFloatTensor_newWithSize1d(N), *b = THFloatTensor_newWithSize1d(N);
  THFloatTensor *c = THFloatTensor_newWithSize1d(N);
  THFloatTensor_uniform(a, g, -1, 1);
  THFloatTensor_uniform(b, g, -1, 1);

  BENCH("add_10M", THFloatTensor_cadd(c, a, 1.0f, b));
  BENCH("mul_scalar_10M", THFloatTensor_mul(c, a, 2.0f));
  BENCH("cmul_10M", THFloatTensor_cmul(c, a, b));
  BENCH("exp_10M", THFloatTensor_exp(c, a));
  BENCH("tanh_10M", THFloatTensor_tanh(c, a));
  BENCH("sigmoid_10M", THFloatTensor_sigmoid(c, a));
  BENCH("sum_all_10M", sink += THFloatTensor_sumall(a));

  /* 2-D 4096 x 4096: strided operands and per-dim reductions */
  const long M = 4096;
  THFloatTensor *x = THFloatTensor_newWithSize2d(M, M), *y = THFloatTensor_newWithSize2d(M, M);
  THFloatTensor *z = THFloatTensor_newWithSize2d(M, M);
  THFloatTensor_uniform(x, g, -1, 1);
  THFloatTensor_uniform(y, g, -1, 1);
  THFloatTensor *yt = THFloatTensor_newTranspose(y, 0, 1);
  THFloatTensor *rs = THFloatTensor_new();

  BENCH("add_transposed_4096sq", THFloatTensor_cadd(z, x, 1.0f, yt));
  BENCH("copy_transpose_4096sq", THFloatTensor_copy(z, yt));
  BENCH("sum_dim0_4096sq", THFloatTensor_sum(rs, x, 0, 0));
  BENCH("sum_dim1_4096sq", THFloatTensor_sum(rs, x, 1, 0));

  /* matrix multiply (uses BLAS if TH was built with it) */
  const long K = 512;
  THFloatTensor *m1 = THFloatTensor_newWithSize2d(K, K), *m2 = THFloatTensor_newWithSize2d(K, K);
  THFloatTensor *mo = THFloatTensor_newWithSize2d(K, K);
  THFloatTensor_uniform(m1, g, -1, 1);
  THFloatTensor_uniform(m2, g, -1, 1);
  BENCH("matmul_512", THFloatTensor_addmm(mo, 0.0f, mo, 1.0f, m1, m2));

  /* sort 1M */
  THFloatTensor *s = THFloatTensor_newWithSize1d(1000 * 1000), *sv = THFloatTensor_new();
  THLongTensor *si = THLongTensor_new();
  THFloatTensor_uniform(s, g, -1, 1);
  BENCH("sort_1M", THFloatTensor_sort(sv, si, s, 0, 0));

  THFloatTensor_free(a); THFloatTensor_free(b); THFloatTensor_free(c);
  THFloatTensor_free(x); THFloatTensor_free(y); THFloatTensor_free(z);
  THFloatTensor_free(yt); THFloatTensor_free(rs);
  THFloatTensor_free(m1); THFloatTensor_free(m2); THFloatTensor_free(mo);
  THFloatTensor_free(s); THFloatTensor_free(sv); THLongTensor_free(si);
}

int main(int argc, char **argv)
{
  THGenerator *g = THGenerator_new();
  THRandom_manualSeed(g, 0);
  /* read before any run: THSetNumThreads() changes omp_get_max_threads() */
  const int max_threads = omp_get_max_threads();

#ifdef USE_BLAS
  const char *blas = "yes";
#else
  const char *blas = "no (built-in fallback)";
#endif
  printf("# th_bench: gcc %s, OpenMP max threads %d, BLAS %s\n",
         __VERSION__, max_threads, blas);
  printf("op,threads,ms\n");

  if (argc > 1) {
    for (int i = 1; i < argc; i++) run(atoi(argv[i]), g);
  } else {
    run(1, g);
    if (max_threads > 1) run(max_threads, g);
  }
  THGenerator_free(g);
  return 0;
}
