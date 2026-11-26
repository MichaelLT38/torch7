#include "THBlas.h"

#ifdef _OPENMP
#include <omp.h>
#endif

/* Built-in blocked GEMM (see generic/THBlas.c): tile sizes shared by all types.
 * A micro-tile is THGEMM_MR rows x two 32-byte vectors of columns; packed
 * panels are THGEMM_KC deep; columns are processed THGEMM_NC at a time. */
#define THGEMM_VBYTES 32
#define THGEMM_MR 6
#define THGEMM_KC 256
#define THGEMM_NC 2048
#define THGEMM_MC_MAX 144
/* below this many multiply-adds the plain triple loop is used */
#define THGEMM_MIN_WORK 4096

/* 1 if the CPU supports AVX2 and FMA (checked once). */
static int THBlas_cpuHasAvx2Fma(void)
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

#include "generic/THBlas.c"
#include "THGenerateAllTypes.h"

#include "generic/THBlas.c"
#include "THGenerateReducedTypes.h"
