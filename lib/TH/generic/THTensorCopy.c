#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THTensorCopy.c"
#else

int THTensor_(copyTransposeValid)(THTensor *tensor, THTensor *src) {
  const int MIN_SZ = 60 * 60;
  return THTensor_(isContiguous)(tensor) &&
         THTensor_(nDimension)(src) == 2 &&
         THTensor_(stride)(src, 0) == 1 &&
         THTensor_(stride)(src, 1) == THTensor_(size)(src, 0) &&
         THTensor_(nElement)(tensor) >= MIN_SZ;
}

// special case copy where tensor is contiguous and src is a transposed matrix
// This can be generalized to most copies, but it's tricker
void THTensor_(copyTranspose)(THTensor *tensor, THTensor *src) {
  #define MIN(x, y) (((x) < (y)) ? (x) : (y))
  #define MAX(x, y) (((x) > (y)) ? (x) : (y))

#ifdef TH_REAL_IS_BYTE
  const int BLOCK_SZ = 120;
#else
  const int BLOCK_SZ = 60;
#endif

  THTensor *buf = THTensor_(newWithSize2d)(BLOCK_SZ, BLOCK_SZ);
  real *sp = THTensor_(data)(src);
  real *rp = THTensor_(data)(tensor);
  real *bp = THTensor_(data)(buf);

  int64_t NR = THTensor_(size)(src, 0);
  int64_t NC = THTensor_(size)(src, 1);
  for (int64_t R = 0; R < NR; R += BLOCK_SZ) {
    for (int64_t C = 0; C < NC; C += BLOCK_SZ) {
      real *spo = sp + R + C * NR;
      real *rpo = rp + C + R * NC;

      int nr = MIN(NR - R, BLOCK_SZ);
      int nc = MIN(NC - C, BLOCK_SZ);

      // 1. copy columns from src to buf
      for (int c = 0; c < nc; c++) {
        memcpy(bp + c * BLOCK_SZ, spo + c * NR, nr * sizeof(real));
      }

      // 2. transpose buf in place
      int rc_max = MAX(nr, nc);
      int rc_min = MIN(nr, nc);
      for (int r = 0; r < rc_max; r++) {
        int end = MIN(r, rc_min);
        for (int c = 0; c < end; c++) {
          real tmp = bp[r + BLOCK_SZ * c];
          bp[r + BLOCK_SZ * c] = bp[r * BLOCK_SZ + c];
          bp[r * BLOCK_SZ + c] = tmp;
        }
      }

      // 3. copy rows from buf to dst
      for (int r = 0; r < nr; r++) {
        memcpy(rpo + r * NC, bp + r * BLOCK_SZ, nc * sizeof(real));
      }
    }
  }
  THTensor_(free)(buf);
  #undef MIN
  #undef MAX
}

static void THTensor_(copyIterLoop)(char **d, const int64_t *s, int64_t n, void *ctx)
{
  (void)ctx;
  char *rp = d[0], *sp = d[1];
  for (int64_t i = 0; i < n; i++, rp += s[0], sp += s[1])
    *(real *)rp = *(const real *)sp;
}

void THTensor_(copy)(THTensor *tensor, THTensor *src)
{
  if (tensor == src) return;
  if (THTensor_(isContiguous)(tensor) && THTensor_(isContiguous)(src) && THTensor_(nElement)(tensor) == THTensor_(nElement)(src)) {
    real *sp = THTensor_(data)(src);
    real *rp = THTensor_(data)(tensor);
    ptrdiff_t sz = THTensor_(nElement)(tensor);
    THVector_(copy)(rp, sp, sz);
  } else if (THTensor_(copyTransposeValid)(tensor, src)) {
    THTensor_(copyTranspose)(tensor, src);
  } else if (THTensor_(isSameSizeAs)(tensor, src) && THTensor_(iterSafe)(tensor, src)) {
    /* same shape: THIter (coalescing + OpenMP). Copies between different
       shapes with the same number of elements keep the documented flat
       element-order semantics below. */
    THIterOperand ops[2] = {THTensor_(iterOperand)(tensor), THTensor_(iterOperand)(src)};
    THIter it;
    THIter_build(&it, 2, ops);
    THIter_run(&it, THTensor_(copyIterLoop), NULL, 32768);
  } else {
    TH_TENSOR_APPLY2(real, tensor, real, src, *tensor_data = *src_data;)
  }
}

#define IMPLEMENT_THTensor_COPY(TYPENAMESRC, TYPE_SRC) \
void THTensor_(copy##TYPENAMESRC)(THTensor *tensor, TH##TYPENAMESRC##Tensor *src) \
{ \
  TH_TENSOR_APPLY2(real, tensor, TYPE_SRC, src, *tensor_data = (real)(*src_data);) \
}

/* the same, with a bulk conversion routine for contiguous tensors */
#define IMPLEMENT_THTensor_COPY_BULK(TYPENAMESRC, TYPE_SRC, BULK) \
void THTensor_(copy##TYPENAMESRC)(THTensor *tensor, TH##TYPENAMESRC##Tensor *src) \
{ \
  ptrdiff_t n = THTensor_(nElement)(tensor); \
  if (THTensor_(isContiguous)(tensor) && TH##TYPENAMESRC##Tensor_isContiguous(src) && \
      n == TH##TYPENAMESRC##Tensor_nElement(src)) { \
    BULK(THTensor_(data)(tensor), TH##TYPENAMESRC##Tensor_data(src), n); \
    return; \
  } \
  TH_TENSOR_APPLY2(real, tensor, TYPE_SRC, src, *tensor_data = (real)(*src_data);) \
}

/* Every pair converts with a plain C cast: the 16-bit types (THHalf, THBFloat16)
   are native GCC floating types, so each conversion rounds once, to nearest even.
   float <-> 16-bit copies of contiguous tensors use the bulk routines of THHalf.h
   (same results). */
IMPLEMENT_THTensor_COPY(Byte, unsigned char)
IMPLEMENT_THTensor_COPY(Char, char)
IMPLEMENT_THTensor_COPY(Short, short)
IMPLEMENT_THTensor_COPY(Int, int)
IMPLEMENT_THTensor_COPY(Long, int64_t)
IMPLEMENT_THTensor_COPY(Double, double)

#if defined(TH_REAL_IS_FLOAT)
IMPLEMENT_THTensor_COPY_BULK(Half, THHalf, TH_halfToFloatN)
IMPLEMENT_THTensor_COPY_BULK(BFloat16, THBFloat16, TH_bfloat16ToFloatN)
#else
IMPLEMENT_THTensor_COPY(Half, THHalf)
IMPLEMENT_THTensor_COPY(BFloat16, THBFloat16)
#endif

#if defined(TH_REAL_IS_HALF)
IMPLEMENT_THTensor_COPY_BULK(Float, float, TH_floatToHalfN)
#elif defined(TH_REAL_IS_BFLOAT16)
IMPLEMENT_THTensor_COPY_BULK(Float, float, TH_floatToBFloat16N)
#else
IMPLEMENT_THTensor_COPY(Float, float)
#endif

#endif
