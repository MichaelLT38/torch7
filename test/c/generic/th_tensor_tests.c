/* Type-generic TH tests, instantiated for every TH type by THGenerateAllTypes.h
 * (same mechanism TH itself uses). Every result is checked element-by-element
 * against a plain reference loop, over several memory layouts:
 *
 *   contiguous, transposed, narrowed (storage offset + padded rows),
 *   inner stride 2, and stride-0 broadcast views (row / column expanded).
 *
 * These are the layouts the Phase 2 iterator and Phase 3 kernels must keep
 * handling, so this file is their regression guard.
 */
#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/th_tensor_tests.c"
#else

#define TT(name) TH_CONCAT_4(test_, Real, _, name)

#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
# define TT_FLOATING 1
#endif

/* Value ranges keep every integer result in range for the type. */
#if defined(TH_REAL_IS_BYTE)
# define TT_LO 0
# define TT_HI 10
#elif defined(TT_FLOATING)
# define TT_LO -1
# define TT_HI 1
#else
# define TT_LO -5
# define TT_HI 5
#endif

#if defined(TH_REAL_IS_FLOAT)
# define TT_EPS FLT_EPSILON
# define TT_ULPS(a, b) th_test_ulps_float((float)(a), (float)(b))
#elif defined(TH_REAL_IS_DOUBLE)
# define TT_EPS DBL_EPSILON
# define TT_ULPS(a, b) th_test_ulps_double((double)(a), (double)(b))
#endif

/* Layout enum, layout names and test shapes are type-independent and live in
 * test_th.c (L_*, TT_NUM_*_LAYOUTS, tt_layout_name, tt_shapes). */

static real TT(random_value)(uint64_t *rng, double lo, double hi)
{
#ifdef TT_FLOATING
  return (real)th_test_uniform(rng, lo, hi);
#else
  return (real)floor(th_test_uniform(rng, lo, hi + 1));
#endif
}

static void TT(fill_contiguous)(THTensor *t, uint64_t *rng, double lo, double hi)
{
  real *d = THTensor_(data)(t);
  ptrdiff_t n = THTensor_(nElement)(t);
  for (ptrdiff_t i = 0; i < n; i++) d[i] = TT(random_value)(rng, lo, hi);
}

/* A logical m x n tensor in the requested memory layout, filled with values
 * in [lo, hi]. */
static THTensor *TT(make)(int kind, long m, long n, uint64_t *rng, double lo, double hi)
{
  THTensor *base, *view, *tmp;
  THLongStorage *sizes;
  switch (kind) {
  case L_CONTIG:
    base = THTensor_(newWithSize2d)(m, n);
    TT(fill_contiguous)(base, rng, lo, hi);
    return base;
  case L_TRANSPOSED:
    base = THTensor_(newWithSize2d)(n, m);
    TT(fill_contiguous)(base, rng, lo, hi);
    view = THTensor_(newTranspose)(base, 0, 1);
    break;
  case L_NARROWED:
    base = THTensor_(newWithSize2d)(m + 3, n + 5);
    TT(fill_contiguous)(base, rng, lo, hi);
    tmp = THTensor_(newNarrow)(base, 0, 2, m);
    view = THTensor_(newNarrow)(tmp, 1, 3, n);
    THTensor_(free)(tmp);
    break;
  case L_STRIDED:
    base = THTensor_(newWithSize2d)(m, 2 * n);
    TT(fill_contiguous)(base, rng, lo, hi);
    view = THTensor_(newWithStorage2d)(THTensor_(storage)(base), 0, m, 2 * n, n, 2);
    break;
  case L_EXPAND_ROW:
  case L_EXPAND_COL:
    base = kind == L_EXPAND_ROW ? THTensor_(newWithSize2d)(1, n)
                                : THTensor_(newWithSize2d)(m, 1);
    TT(fill_contiguous)(base, rng, lo, hi);
    sizes = THLongStorage_newWithSize2(m, n);
    view = THTensor_(newExpand)(base, sizes);
    THLongStorage_free(sizes);
    break;
  default:
    abort();
  }
  THTensor_(free)(base);
  return view;
}

static real *TT(at)(THTensor *t, long i, long j)
{
  return THTensor_(data)(t) + i * THTensor_(stride)(t, 0) + j * THTensor_(stride)(t, 1);
}

/* got must equal want (integers), or be within a few ulps of the magnitude
 * scale of the operands (floating point: allows FMA / reordering). */
static int TT(close)(real got, double want, double scale)
{
#ifdef TT_FLOATING
  if (isnan(want)) return isnan(got);
  return fabs((double)got - want) <= 4 * TT_EPS * (fabs(want) + scale) + 1e-300;
#else
  (void)scale;
  return (double)got == want;
#endif
}

/* ------------------------------------------------------------------------ */
/* Binary element-wise ops: cadd, csub, cmul, cdiv                          */
/* ------------------------------------------------------------------------ */
static void TT(binary_ops)(void)
{
  static const char *opname[] = {"cadd", "csub", "cmul", "cdiv"};
  uint64_t rng = 1;
  for (int s = 0; s < TT_NUM_SHAPES; s++) {
    long m = tt_shapes[s][0], n = tt_shapes[s][1];
    for (int ka = 0; ka < TT_NUM_IN_LAYOUTS; ka++)
    for (int kb = 0; kb < TT_NUM_IN_LAYOUTS; kb++)
    for (int kr = 0; kr < TT_NUM_OUT_LAYOUTS; kr++) {
      THTensor *a = TT(make)(ka, m, n, &rng, TT_LO, TT_HI);
      /* strictly positive denominators so cdiv is defined for integers */
      THTensor *b = TT(make)(kb, m, n, &rng, 1, TT_HI > 1 ? TT_HI : 2);
      THTensor *r = TT(make)(kr, m, n, &rng, TT_LO, TT_HI);
      for (int op = 0; op < 4; op++) {
        switch (op) {
        case 0: THTensor_(cadd)(r, a, 3, b); break;
        case 1: THTensor_(csub)(r, a, 2, b); break;
        case 2: THTensor_(cmul)(r, a, b); break;
        case 3: THTensor_(cdiv)(r, a, b); break;
        }
        int ok = 1;
        for (long i = 0; i < m && ok; i++)
        for (long j = 0; j < n && ok; j++) {
          real x = *TT(at)(a, i, j), y = *TT(at)(b, i, j), got = *TT(at)(r, i, j);
          double want, scale;
          switch (op) {
          case 0: want = (double)(real)(x + 3 * y); scale = fabs((double)x) + 3 * fabs((double)y); break;
          case 1: want = (double)(real)(x - 2 * y); scale = fabs((double)x) + 2 * fabs((double)y); break;
          case 2: want = (double)(real)(x * y);     scale = fabs((double)x * y); break;
          default:
#ifdef TT_FLOATING
            want = (double)x / y;
#else
            want = (double)(real)(x / y);
#endif
            scale = fabs(want); break;
          }
          ok = TT(close)(got, want, scale);
          CHECK(ok, "%s %ldx%ld a=%s b=%s r=%s at (%ld,%ld): got %g want %g",
                opname[op], m, n, tt_layout_name(ka), tt_layout_name(kb),
                tt_layout_name(kr), i, j, (double)got, want);
        }
      }
      THTensor_(free)(a); THTensor_(free)(b); THTensor_(free)(r);
    }
  }
}

/* r aliases an input (cadd has a dedicated axpy path for r == t). */
static void TT(binary_aliasing)(void)
{
  uint64_t rng = 2;
  for (int k = 0; k < TT_NUM_OUT_LAYOUTS; k++) {
    long m = 37, n = 29;
    THTensor *a = TT(make)(k, m, n, &rng, TT_LO, TT_HI);
    THTensor *b = TT(make)(k, m, n, &rng, TT_LO, TT_HI);
    THTensor *a0 = THTensor_(newClone)(a), *b0 = THTensor_(newClone)(b);

    THTensor_(cadd)(a, a, 2, b);                        /* a = a + 2b  */
    THTensor_(cmul)(b, a0, b);                          /* b = a0 * b  */
    int ok = 1;
    for (long i = 0; i < m && ok; i++)
    for (long j = 0; j < n && ok; j++) {
      real x = *TT(at)(a0, i, j), y = *TT(at)(b0, i, j);
      ok = TT(close)(*TT(at)(a, i, j), (double)(real)(x + 2 * y), fabs((double)x) + 2 * fabs((double)y))
        && TT(close)(*TT(at)(b, i, j), (double)(real)(x * y), fabs((double)x * y));
      CHECK(ok, "aliasing layout=%s at (%ld,%ld)", tt_layout_name(k), i, j);
    }
    THTensor_(free)(a); THTensor_(free)(b); THTensor_(free)(a0); THTensor_(free)(b0);
  }
}

/* ------------------------------------------------------------------------ */
/* Tensor-scalar ops: add, mul, div, clamp                                  */
/* ------------------------------------------------------------------------ */
static void TT(scalar_ops)(void)
{
  static const char *opname[] = {"add", "mul", "div", "clamp"};
  uint64_t rng = 3;
  for (int s = 0; s < TT_NUM_SHAPES; s++) {
    long m = tt_shapes[s][0], n = tt_shapes[s][1];
    for (int ka = 0; ka < TT_NUM_IN_LAYOUTS; ka++)
    for (int kr = 0; kr < TT_NUM_OUT_LAYOUTS; kr++) {
      THTensor *a = TT(make)(ka, m, n, &rng, TT_LO, TT_HI);
      THTensor *r = TT(make)(kr, m, n, &rng, TT_LO, TT_HI);
      const real lo = (real)(TT_LO / 2.0), hi = (real)(TT_HI / 2.0);
      for (int op = 0; op < 4; op++) {
        switch (op) {
        case 0: THTensor_(add)(r, a, 3); break;
        case 1: THTensor_(mul)(r, a, 2); break;
        case 2: THTensor_(div)(r, a, 2); break;
        case 3: THTensor_(clamp)(r, a, lo, hi); break;
        }
        int ok = 1;
        for (long i = 0; i < m && ok; i++)
        for (long j = 0; j < n && ok; j++) {
          real x = *TT(at)(a, i, j), got = *TT(at)(r, i, j);
          double want;
          switch (op) {
          case 0: want = (double)(real)(x + 3); break;
          case 1: want = (double)(real)(x * 2); break;
          case 2: want = (double)(real)(x / 2); break;
          default: want = x < lo ? lo : (x > hi ? hi : x); break;
          }
          ok = TT(close)(got, want, fabs((double)x));
          CHECK(ok, "%s %ldx%ld a=%s r=%s at (%ld,%ld): got %g want %g", opname[op], m, n,
                tt_layout_name(ka), tt_layout_name(kr), i, j, (double)got, want);
        }
      }
      THTensor_(free)(a); THTensor_(free)(r);
    }
  }
}

/* ------------------------------------------------------------------------ */
/* Unary ops. Transcendentals must stay within 4 ulps of libm (double).    */
/* ------------------------------------------------------------------------ */
#ifdef TT_FLOATING
static void TT(unary_ops)(void)
{
  static const char *opname[] = {"exp", "log", "tanh", "sigmoid", "sqrt", "abs", "neg"};
  uint64_t rng = 4;
  for (int s = 0; s < TT_NUM_SHAPES; s++) {
    long m = tt_shapes[s][0], n = tt_shapes[s][1];
    for (int ka = 0; ka < TT_NUM_IN_LAYOUTS; ka++)
    for (int kr = 0; kr < TT_NUM_OUT_LAYOUTS; kr++)
    for (int op = 0; op < 7; op++) {
      /* log and sqrt need positive inputs; exp gets a wide range. */
      double lo = (op == 1 || op == 4) ? 1e-3 : (op == 0 ? -20 : -8);
      double hi = (op == 1 || op == 4) ? 100 : (op == 0 ? 20 : 8);
      THTensor *a = TT(make)(ka, m, n, &rng, lo, hi);
      THTensor *r = TT(make)(kr, m, n, &rng, 0, 1);
      switch (op) {
      case 0: THTensor_(exp)(r, a); break;
      case 1: THTensor_(log)(r, a); break;
      case 2: THTensor_(tanh)(r, a); break;
      case 3: THTensor_(sigmoid)(r, a); break;
      case 4: THTensor_(sqrt)(r, a); break;
      case 5: THTensor_(abs)(r, a); break;
      case 6: THTensor_(neg)(r, a); break;
      }
      int ok = 1;
      for (long i = 0; i < m && ok; i++)
      for (long j = 0; j < n && ok; j++) {
        double x = *TT(at)(a, i, j);
        real got = *TT(at)(r, i, j);
        double want;
        switch (op) {
        case 0: want = exp(x); break;
        case 1: want = log(x); break;
        case 2: want = tanh(x); break;
        case 3: want = 1.0 / (1.0 + exp(-x)); break;
        case 4: want = sqrt(x); break;
        case 5: want = fabs(x); break;
        default: want = -x; break;
        }
        double ulps = TT_ULPS(got, (real)want);
        ok = ulps <= 4;
        CHECK(ok, "%s %ldx%ld a=%s r=%s x=%.9g: got %.9g want %.9g (%g ulps)", opname[op],
              m, n, tt_layout_name(ka), tt_layout_name(kr), x, (double)got, want, ulps);
      }
      THTensor_(free)(a); THTensor_(free)(r);
    }
  }
}
#endif

/* ------------------------------------------------------------------------ */
/* Copy between every layout pair                                           */
/* ------------------------------------------------------------------------ */
static void TT(copy_layouts)(void)
{
  uint64_t rng = 5;
  for (int s = 0; s < TT_NUM_SHAPES; s++) {
    long m = tt_shapes[s][0], n = tt_shapes[s][1];
    for (int ka = 0; ka < TT_NUM_IN_LAYOUTS; ka++)
    for (int kr = 0; kr < TT_NUM_OUT_LAYOUTS; kr++) {
      THTensor *a = TT(make)(ka, m, n, &rng, TT_LO, TT_HI);
      THTensor *r = TT(make)(kr, m, n, &rng, TT_LO, TT_HI);
      THTensor_(copy)(r, a);
      int ok = 1;
      for (long i = 0; i < m && ok; i++)
      for (long j = 0; j < n && ok; j++) {
        ok = *TT(at)(r, i, j) == *TT(at)(a, i, j);
        CHECK(ok, "copy %ldx%ld %s -> %s at (%ld,%ld)", m, n, tt_layout_name(ka),
              tt_layout_name(kr), i, j);
      }
      THTensor_(free)(a); THTensor_(free)(r);
    }
  }
}

/* 3-D permuted (non-coalescable) operand. */
static void TT(binary_3d_permuted)(void)
{
  uint64_t rng = 6;
  THTensor *base = THTensor_(newWithSize3d)(4, 5, 6);
  TT(fill_contiguous)(base, &rng, TT_LO, TT_HI);
  THTensor *a = THTensor_(newTranspose)(base, 0, 2);    /* 6 x 5 x 4, strides 1,6,30 */
  THTensor *b = THTensor_(newWithSize3d)(6, 5, 4);
  TT(fill_contiguous)(b, &rng, TT_LO, TT_HI);
  THTensor *r = THTensor_(new)();
  THTensor_(cadd)(r, a, 1, b);
  CHECK(THTensor_(nDimension)(r) == 3 && THTensor_(size)(r, 0) == 6 &&
        THTensor_(size)(r, 1) == 5 && THTensor_(size)(r, 2) == 4, "3d result shape");
  int ok = 1;
  for (long i = 0; i < 6 && ok; i++)
  for (long j = 0; j < 5 && ok; j++)
  for (long k = 0; k < 4 && ok; k++) {
    real x = THTensor_(data)(a)[i * a->stride[0] + j * a->stride[1] + k * a->stride[2]];
    real y = THTensor_(data)(b)[i * b->stride[0] + j * b->stride[1] + k * b->stride[2]];
    real got = THTensor_(data)(r)[i * r->stride[0] + j * r->stride[1] + k * r->stride[2]];
    ok = TT(close)(got, (double)(real)(x + y), fabs((double)x) + fabs((double)y));
    CHECK(ok, "3d cadd at (%ld,%ld,%ld)", i, j, k);
  }
  THTensor_(free)(base); THTensor_(free)(a); THTensor_(free)(b); THTensor_(free)(r);
}

/* Empty tensors must not crash and must stay empty. */
static void TT(empty)(void)
{
  THTensor *a = THTensor_(new)(), *b = THTensor_(new)(), *r = THTensor_(new)();
  THTensor_(cadd)(r, a, 1, b);
  CHECK(THTensor_(nElement)(r) == 0, "empty cadd result has %ld elements",
        (long)THTensor_(nElement)(r));
  CHECK(THTensor_(sumall)(a) == 0, "sumall of empty tensor is not 0");
  THTensor_(copy)(r, a);
  CHECK(THTensor_(nElement)(r) == 0, "empty copy");
  THTensor_(free)(a); THTensor_(free)(b); THTensor_(free)(r);
}

/* ------------------------------------------------------------------------ */
/* Reductions. Narrow integer types would overflow their own accumulator,   */
/* so reductions are checked for int, long, float and double.              */
/* ------------------------------------------------------------------------ */
#if defined(TH_REAL_IS_INT) || defined(TH_REAL_IS_LONG) || defined(TT_FLOATING)
static void TT(reductions)(void)
{
  uint64_t rng = 7;
#ifdef TT_FLOATING
  /* Summation order may change (pairwise / SIMD), so allow O(eps * log n). */
  const double rel = 64 * TT_EPS;
#else
  const double rel = 0;
#endif
  for (int s = 0; s < TT_NUM_SHAPES; s++) {
    long m = tt_shapes[s][0], n = tt_shapes[s][1];
    for (int ka = 0; ka < TT_NUM_IN_LAYOUTS; ka++) {
      THTensor *a = TT(make)(ka, m, n, &rng, TT_LO, TT_HI);

      /* sumall */
      double want = 0, mag = 0;
      for (long i = 0; i < m; i++)
        for (long j = 0; j < n; j++) { want += *TT(at)(a, i, j); mag += fabs((double)*TT(at)(a, i, j)); }
      double got = (double)THTensor_(sumall)(a);
      CHECK(fabs(got - want) <= rel * mag, "sumall %ldx%ld %s: got %.17g want %.17g",
            m, n, tt_layout_name(ka), got, want);

      /* sum / mean / max / min along each dim, with and without keepdim */
      for (int dim = 0; dim < 2; dim++)
      for (int keepdim = 0; keepdim < 2; keepdim++) {
        long outer = dim == 0 ? n : m, inner = dim == 0 ? m : n;
        THTensor *rs = THTensor_(new)(), *rv = THTensor_(new)();
        THLongTensor *ri = THLongTensor_new();
        THTensor_(sum)(rs, a, dim, keepdim);
        THTensor_(max)(rv, ri, a, dim, keepdim);
        int expect_ndim = keepdim ? 2 : 1;
        CHECK(THTensor_(nDimension)(rs) == expect_ndim && THTensor_(nElement)(rs) == outer,
              "sum dim=%d keepdim=%d shape", dim, keepdim);
        int ok = 1;
        for (long o = 0; o < outer && ok; o++) {
          double ws = 0, wm = 0; real wmax = 0;
          for (long k = 0; k < inner; k++) {
            real v = dim == 0 ? *TT(at)(a, k, o) : *TT(at)(a, o, k);
            ws += v; wm += fabs((double)v);
            if (k == 0 || v > wmax) wmax = v;
          }
          /* reduced tensors are contiguous; index them flat */
          real gs = THTensor_(data)(rs)[o];
          real gm = THTensor_(data)(rv)[o];
          /* Output indices are 0-based at the C level; the Lua binding adds
             1. (Index *inputs*, e.g. index/gather, use TH_INDEX_BASE.) */
          long gi = THLongTensor_data(ri)[o];
          ok = fabs((double)gs - ws) <= rel * wm + (rel > 0 ? 4 * fabs(ws) * rel : 0)
               && gm == wmax && gi >= 0 && gi < inner
               && (dim == 0 ? *TT(at)(a, gi, o) : *TT(at)(a, o, gi)) == wmax;
          CHECK(ok, "sum/max dim=%d keepdim=%d %ldx%ld %s slice %ld: sum %g/%g max %g/%g idx %ld",
                dim, keepdim, m, n, tt_layout_name(ka), o, (double)gs, ws,
                (double)gm, (double)wmax, gi);
        }
#ifdef TT_FLOATING
        THTensor_(mean)(rs, a, dim, keepdim);
        for (long o = 0; o < outer && ok; o++) {
          double ws = 0, wm = 0;
          for (long k = 0; k < inner; k++) {
            double v = dim == 0 ? *TT(at)(a, k, o) : *TT(at)(a, o, k);
            ws += v; wm += fabs(v);
          }
          /* mean accumulates in `real`: error scales with sum|x|, not |mean| */
          ok = TT(close)(THTensor_(data)(rs)[o], ws / inner, wm);
          CHECK(ok, "mean dim=%d %ldx%ld %s slice %ld: got %.9g want %.9g", dim, m, n,
                tt_layout_name(ka), o, (double)THTensor_(data)(rs)[o], ws / inner);
        }
#endif
        THTensor_(free)(rs); THTensor_(free)(rv); THLongTensor_free(ri);
      }
      THTensor_(free)(a);
    }
  }
}

/* sort along the last dim: values ascending/descending, indices consistent */
static void TT(sort)(void)
{
  uint64_t rng = 8;
  for (int ka = 0; ka < TT_NUM_IN_LAYOUTS; ka++)
  for (int desc = 0; desc < 2; desc++) {
    long m = 9, n = 257;
    THTensor *a = TT(make)(ka, m, n, &rng, TT_LO, TT_HI);
    THTensor *rv = THTensor_(new)();
    THLongTensor *ri = THLongTensor_new();
    THTensor_(sort)(rv, ri, a, 1, desc);
    int ok = 1;
    for (long i = 0; i < m && ok; i++)
    for (long j = 0; j < n && ok; j++) {
      real v = *TT(at)(rv, i, j);
      /* 0-based at the C level (see reductions) */
      long idx = THLongTensor_data(ri)[i * THLongTensor_stride(ri, 0) + j * THLongTensor_stride(ri, 1)];
      ok = idx >= 0 && idx < n && *TT(at)(a, i, idx) == v;
      if (ok && j > 0) {
        real prev = *TT(at)(rv, i, j - 1);
        ok = desc ? prev >= v : prev <= v;
      }
      CHECK(ok, "sort desc=%d %s row %ld col %ld", desc, tt_layout_name(ka), i, j);
    }
    THTensor_(free)(a); THTensor_(free)(rv); THLongTensor_free(ri);
  }
}
#endif

/* ------------------------------------------------------------------------ */
/* Matrix products: r = beta*t + alpha*(m1 @ m2), with transposed operands  */
/* ------------------------------------------------------------------------ */
#ifdef TT_FLOATING
static void TT(addmm)(void)
{
  static const long dims[][3] = {{1, 1, 1}, {17, 23, 11}, {64, 64, 64}, {5, 130, 3}};
  uint64_t rng = 9;
  for (int d = 0; d < 4; d++) {
    long M = dims[d][0], K = dims[d][1], N = dims[d][2];
    for (int k1 = 0; k1 < TT_NUM_OUT_LAYOUTS; k1++)
    for (int k2 = 0; k2 < TT_NUM_OUT_LAYOUTS; k2++) {
      THTensor *m1 = TT(make)(k1, M, K, &rng, -1, 1);
      THTensor *m2 = TT(make)(k2, K, N, &rng, -1, 1);
      THTensor *t = TT(make)(L_CONTIG, M, N, &rng, -1, 1);
      THTensor *r = THTensor_(new)();
      THTensor_(addmm)(r, 0.5, t, 2, m1, m2);
      int ok = 1;
      for (long i = 0; i < M && ok; i++)
      for (long j = 0; j < N && ok; j++) {
        double acc = 0, mag = 0;
        for (long k = 0; k < K; k++) {
          double p = (double)*TT(at)(m1, i, k) * *TT(at)(m2, k, j);
          acc += p; mag += fabs(p);
        }
        double want = 0.5 * *TT(at)(t, i, j) + 2 * acc;
        ok = fabs(*TT(at)(r, i, j) - want) <= 8 * TT_EPS * (2 * mag + 1) * (1 + log2((double)K));
        CHECK(ok, "addmm %ldx%ldx%ld m1=%s m2=%s at (%ld,%ld): got %g want %g", M, K, N,
              tt_layout_name(k1), tt_layout_name(k2), i, j, (double)*TT(at)(r, i, j), want);
      }
      THTensor_(free)(m1); THTensor_(free)(m2); THTensor_(free)(t); THTensor_(free)(r);
    }
  }
}

static void TT(addmv_dot)(void)
{
  uint64_t rng = 10;
  long M = 31, K = 47;
  for (int km = 0; km < TT_NUM_OUT_LAYOUTS; km++) {
    THTensor *mat = TT(make)(km, M, K, &rng, -1, 1);
    THTensor *v2 = TT(make)(L_STRIDED, 1, K, &rng, -1, 1);   /* 1 x K, stride 2 */
    THTensor *vec = THTensor_(newSelect)(v2, 0, 0);          /* K, stride 2     */
    THTensor *t = THTensor_(newWithSize1d)(M);
    TT(fill_contiguous)(t, &rng, -1, 1);
    THTensor *r = THTensor_(new)();
    THTensor_(addmv)(r, 0.5, t, 2, mat, vec);
    int ok = 1;
    for (long i = 0; i < M && ok; i++) {
      double acc = 0, mag = 0;
      for (long k = 0; k < K; k++) {
        double p = (double)*TT(at)(mat, i, k) * THTensor_(data)(vec)[k * THTensor_(stride)(vec, 0)];
        acc += p; mag += fabs(p);
      }
      double want = 0.5 * THTensor_(data)(t)[i] + 2 * acc;
      ok = fabs(THTensor_(data)(r)[i * THTensor_(stride)(r, 0)] - want) <= 8 * TT_EPS * (2 * mag + 1) * 8;
      CHECK(ok, "addmv mat=%s row %ld", tt_layout_name(km), i);
    }
    /* dot of a strided vector with a row of mat */
    THTensor *row = THTensor_(newSelect)(mat, 0, 3);
    double want = 0, mag = 0;
    for (long k = 0; k < K; k++) {
      double p = (double)THTensor_(data)(row)[k * THTensor_(stride)(row, 0)] *
                 THTensor_(data)(vec)[k * THTensor_(stride)(vec, 0)];
      want += p; mag += fabs(p);
    }
    double got = (double)THTensor_(dot)(row, vec);
    CHECK(fabs(got - want) <= 8 * TT_EPS * (mag + 1) * 8, "dot mat=%s: got %g want %g",
          tt_layout_name(km), got, want);
    THTensor_(free)(row);
    THTensor_(free)(mat); THTensor_(free)(v2); THTensor_(free)(vec); THTensor_(free)(t); THTensor_(free)(r);
  }
}
#endif

/* ------------------------------------------------------------------------ */
/* Suite entry point for this type                                          */
/* ------------------------------------------------------------------------ */
static void TT(suite)(void)
{
  RUN(TT(binary_ops));
  RUN(TT(binary_aliasing));
  RUN(TT(binary_3d_permuted));
  RUN(TT(scalar_ops));
  RUN(TT(copy_layouts));
  RUN(TT(empty));
#ifdef TT_FLOATING
  RUN(TT(unary_ops));
  RUN(TT(addmm));
  RUN(TT(addmv_dot));
#endif
#if defined(TH_REAL_IS_INT) || defined(TH_REAL_IS_LONG) || defined(TT_FLOATING)
  RUN(TT(reductions));
  RUN(TT(sort));
#endif
}

#undef TT
#undef TT_FLOATING
#undef TT_LO
#undef TT_HI
#undef TT_EPS
#undef TT_ULPS

#endif
