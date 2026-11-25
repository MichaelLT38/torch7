/* C-level test suite for TH.
 *
 *   th_tests            run every suite
 *   th_tests <suite>    run one suite: byte char short int long float double misc
 *
 * Exit status is non-zero if any test failed. Registered with ctest as th.<suite>.
 */
#include "th_test.h"

int th_test_failures = 0;
int th_test_total_failed = 0;
long th_test_checks = 0;
const char *th_test_current = "";
th_jmp_buf *th_test_jmp = NULL;
char th_test_errmsg[1024];

static void th_test_on_error(const char *msg, void *data)
{
  (void)data;
  snprintf(th_test_errmsg, sizeof(th_test_errmsg), "%s", msg);
  if (th_test_jmp) TH_TEST_LONGJMP(*th_test_jmp);
  fprintf(stderr, "    UNEXPECTED TH error in %s: %s\n", th_test_current, msg);
  exit(2);
}

static void th_test_on_arg_error(int arg, const char *msg, void *data)
{
  (void)data;
  char buf[1024];
  snprintf(buf, sizeof(buf), "bad argument #%d: %s", arg, msg);
  th_test_on_error(buf, NULL);
}

void th_test_init(void)
{
  THSetDefaultErrorHandler(th_test_on_error, NULL);
  THSetDefaultArgErrorHandler(th_test_on_arg_error, NULL);
}

/* ---- shared by every type instantiation of generic/th_tensor_tests.c ---- */
enum { L_CONTIG, L_TRANSPOSED, L_NARROWED, L_STRIDED, L_EXPAND_ROW, L_EXPAND_COL };
#define TT_NUM_IN_LAYOUTS 6   /* inputs may use any layout                  */
#define TT_NUM_OUT_LAYOUTS 4  /* outputs must not alias (no stride-0 views) */

static const char *tt_layout_name(int k)
{
  static const char *names[] = {"contig", "transposed", "narrowed", "strided",
                                "expand_row", "expand_col"};
  return names[k];
}

/* {1,1}: degenerate; odd sizes hit SIMD tails; 400x300 = 120000 elements is
 * above TH_OMP_OVERHEAD_THRESHOLD (100000), so OpenMP paths run too. */
static const long tt_shapes[][2] = {{1, 1}, {7, 13}, {64, 33}, {400, 300}};
#define TT_NUM_SHAPES ((int)(sizeof(tt_shapes) / sizeof(tt_shapes[0])))

#include "generic/th_tensor_tests.c"
#include "THGenerateAllTypes.h"

#include "test_iter.c"   /* suite "iter": THIter unit tests */

/* ------------------------------------------------------------------------ */
/* Non-generic tests                                                        */
/* ------------------------------------------------------------------------ */

/* Shape errors (Phase 2c rule, agreed with LatticeAI): incompatible shapes,
 * incompatible shapes with equal element counts (upstream flattened those),
 * and the "same element count, bigger broadcast result" case all raise errors
 * that name the shapes; an in-place result cannot change shape. */
static void misc_shape_errors(void)
{
  THFloatTensor *r = THFloatTensor_new();
  THFloatTensor *a44 = THFloatTensor_newWithSize2d(4, 4), *a28 = THFloatTensor_newWithSize2d(2, 8);
  THFloatTensor *a3 = THFloatTensor_newWithSize1d(3);
  THFloatTensor *c = THFloatTensor_newWithSize2d(784, 1), *v = THFloatTensor_newWithSize1d(784);
  THFloatTensor *row = THFloatTensor_newWithSize2d(1, 4);
  THFloatTensor_fill(a44, 1); THFloatTensor_fill(a28, 1); THFloatTensor_fill(a3, 1);
  THFloatTensor_fill(c, 1); THFloatTensor_fill(v, 1); THFloatTensor_fill(row, 1);

  EXPECT_TH_ERROR(THFloatTensor_cadd(r, a44, 1, a3), "are not broadcastable");
  EXPECT_TH_ERROR(THFloatTensor_cadd(r, a44, 1, a28), "same number of elements");
  EXPECT_TH_ERROR(THFloatTensor_cmul(r, a44, a28), "use :view()");
  EXPECT_TH_ERROR(THFloatTensor_cadd(r, c, 1, v), "would broadcast to [784 x 784]");
  EXPECT_TH_ERROR(THFloatTensor_cadd(r, c, 1, v), "use :view() to make the intent explicit");
  EXPECT_TH_ERROR(THFloatTensor_cadd(r, c, 1, v), "[784 x 1] and [784]");
  EXPECT_TH_ERROR(THFloatTensor_cadd(row, row, 1, a44), "in-place result [1 x 4]");

  THFloatTensor *m1 = THFloatTensor_newWithSize2d(3, 4), *m2 = THFloatTensor_newWithSize2d(5, 2);
  THFloatTensor_zero(m1); THFloatTensor_zero(m2);
  EXPECT_TH_ERROR(THFloatTensor_addmm(r, 0, r, 1, m1, m2), "size mismatch");
  THFloatTensor_free(r); THFloatTensor_free(a44); THFloatTensor_free(a28); THFloatTensor_free(a3);
  THFloatTensor_free(c); THFloatTensor_free(v); THFloatTensor_free(row);
  THFloatTensor_free(m1); THFloatTensor_free(m2);
}

/* Broadcasting results (Phase 2c) against reference loops: (m,1) with (1,n),
 * a 1-D row with a 2-D tensor, in place, integer exactness, ternary ops, and
 * an input that overlaps the output (must use the original values). */
static void misc_broadcasting(void)
{
  uint64_t rng = 31;
  const int64_t m = 37, n = 300;   /* m*n above the OpenMP grain */
  THFloatTensor *col = test_Float_make(L_CONTIG, m, 1, &rng, -1, 1);
  THFloatTensor *row = test_Float_make(L_TRANSPOSED, 1, n, &rng, 0.5, 2);   /* strided */
  THFloatTensor *r = THFloatTensor_new();
  for (int op = 0; op < 5; op++) {
    if (op == 0) THFloatTensor_cadd(r, col, 2, row);
    else if (op == 1) THFloatTensor_cmul(r, col, row);
    else if (op == 2) THFloatTensor_cdiv(r, col, row);
    else if (op == 3) THFloatTensor_cmax(r, col, row);
    else THFloatTensor_lerp(r, col, row, 0.25f);
    CHECK(THFloatTensor_nDimension(r) == 2 && THFloatTensor_size(r, 0) == m && THFloatTensor_size(r, 1) == n,
          "op %d: (m,1) with (1,n) gives (m,n)", op);
    int ok = 1;
    for (int64_t i = 0; i < m && ok; i++)
    for (int64_t j = 0; j < n && ok; j++) {
      float x = THFloatTensor_get2d(col, i, 0), y = THFloatTensor_get2d(row, 0, j);
      float want = op == 0 ? x + 2 * y : op == 1 ? x * y : op == 2 ? x / y
                 : op == 3 ? (x > y ? x : y) : x + 0.25f * (y - x);
      ok = THFloatTensor_get2d(r, i, j) == want;
      CHECK(ok, "broadcast op %d at (%" PRId64 ",%" PRId64 "): got %g want %g", op, i, j,
            (double)THFloatTensor_get2d(r, i, j), (double)want);
    }
  }

  /* a 1-D vector against a 2-D tensor (trailing dimensions align), in place */
  THFloatTensor *x = test_Float_make(L_NARROWED, m, n, &rng, -1, 1);
  THFloatTensor *x0 = THFloatTensor_newClone(x);
  THFloatTensor *vec = THFloatTensor_newWithSize1d(n);
  for (int64_t j = 0; j < n; j++) THFloatTensor_set1d(vec, j, (float)j);
  THFloatTensor_cadd(x, x, 1, vec);
  int ok = 1;
  for (int64_t i = 0; i < m && ok; i++)
  for (int64_t j = 0; j < n && ok; j++) {
    ok = THFloatTensor_get2d(x, i, j) == THFloatTensor_get2d(x0, i, j) + (float)j;
    CHECK(ok, "in-place x:add(vec) at (%" PRId64 ",%" PRId64 ")", i, j);
  }

  /* overlapping input: x += x[0] (row 0 broadcast over x, x[0] itself changes) */
  THFloatTensor *y = THFloatTensor_newClone(x0);
  THFloatTensor *y0 = THFloatTensor_newClone(x0);
  THFloatTensor *first = THFloatTensor_newSelect(y, 0, 0);
  THFloatTensor_cadd(y, y, 1, first);
  ok = 1;
  for (int64_t i = 0; i < m && ok; i++)
  for (int64_t j = 0; j < n && ok; j++) {
    ok = THFloatTensor_get2d(y, i, j) == THFloatTensor_get2d(y0, i, j) + THFloatTensor_get2d(y0, 0, j);
    CHECK(ok, "overlapping x:add(x[1]) at (%" PRId64 ",%" PRId64 ")", i, j);
  }

  /* ternary: addcmul with all three operands broadcast */
  THFloatTensor *t = test_Float_make(L_CONTIG, 1, n, &rng, -1, 1);
  THFloatTensor_addcmul(r, t, 3, col, row);
  ok = THFloatTensor_size(r, 0) == m && THFloatTensor_size(r, 1) == n;
  for (int64_t i = 0; i < m && ok; i++)
  for (int64_t j = 0; j < n && ok; j++) {
    float want = THFloatTensor_get2d(t, 0, j) + 3 * THFloatTensor_get2d(col, i, 0) * THFloatTensor_get2d(row, 0, j);
    ok = THFloatTensor_get2d(r, i, j) == want;
    CHECK(ok, "addcmul broadcast at (%" PRId64 ",%" PRId64 ")", i, j);
  }

  /* integers stay exact under broadcasting (wrap-around) */
  THIntTensor *ic = THIntTensor_newWithSize2d(5, 1), *ir = THIntTensor_newWithSize1d(7), *io = THIntTensor_new();
  for (int i = 0; i < 5; i++) THIntTensor_set2d(ic, i, 0, 2147483600 + i);
  for (int j = 0; j < 7; j++) THIntTensor_set1d(ir, j, 100 * j - 3);
  THIntTensor_cadd(io, ic, 1, ir);
  ok = THIntTensor_size(io, 0) == 5 && THIntTensor_size(io, 1) == 7;
  for (int i = 0; i < 5 && ok; i++)
  for (int j = 0; j < 7 && ok; j++) {
    int want = (int)(uint32_t)((uint32_t)(2147483600 + i) + (uint32_t)(100 * j - 3));
    ok = THIntTensor_get2d(io, i, j) == want;
    CHECK(ok, "int broadcast add at (%d,%d): got %d want %d", i, j, THIntTensor_get2d(io, i, j), want);
  }

  THFloatTensor_free(col); THFloatTensor_free(row); THFloatTensor_free(r);
  THFloatTensor_free(x); THFloatTensor_free(x0); THFloatTensor_free(vec);
  THFloatTensor_free(y); THFloatTensor_free(y0); THFloatTensor_free(first); THFloatTensor_free(t);
  THIntTensor_free(ic); THIntTensor_free(ir); THIntTensor_free(io);
}

/* Cross-type copies over non-contiguous layouts. */
static void misc_copy_cross_type(void)
{
  uint64_t rng = 11;
  THDoubleTensor *base = THDoubleTensor_newWithSize2d(33, 17);
  double *d = THDoubleTensor_data(base);
  for (int i = 0; i < 33 * 17; i++) d[i] = th_test_uniform(&rng, -100, 100);
  THDoubleTensor *src = THDoubleTensor_newTranspose(base, 0, 1);   /* 17 x 33 */
  THFloatTensor *f = THFloatTensor_newWithSize2d(17, 33);
  THLongTensor *l = THLongTensor_newWithSize2d(17, 33);
  THFloatTensor_copyDouble(f, src);
  THLongTensor_copyDouble(l, src);
  int ok = 1;
  for (long i = 0; i < 17 && ok; i++)
  for (long j = 0; j < 33 && ok; j++) {
    double v = THDoubleTensor_get2d(src, i, j);
    ok = THFloatTensor_get2d(f, i, j) == (float)v && THLongTensor_get2d(l, i, j) == (long)v;
    CHECK(ok, "cross-type copy at (%ld,%ld)", i, j);
  }
  THDoubleTensor_free(base); THDoubleTensor_free(src);
  THFloatTensor_free(f); THLongTensor_free(l);
}

/* CONTRACT: max/min along a dim return the LOWEST index among tied values.
 * Downstream code depends on this (e.g. LatticeAI's golden model and its
 * hardware ARGMAX use the same rule), so a parallel or SIMD max/min kernel
 * must preserve it. Checked for IntTensor and FloatTensor, both dims,
 * keepdim 0/1, contiguous and transposed input. */
static void misc_max_min_ties_lowest_index(void)
{
  const long rows = 37, cols = 1000;
  /* row r: all zeros except +5 at columns tie_a < tie_b and -5 at columns
     tie_c < tie_d, so both max and min have exactly two tied candidates */
  for (int transposed = 0; transposed < 2; transposed++)
  for (int dim = 0; dim < 2; dim++)
  for (int keepdim = 0; keepdim < 2; keepdim++) {
    /* logical shape: along `dim` there are `cols` candidates */
    long n_slices = rows, n_cand = cols;
    THIntTensor *ib = transposed ? THIntTensor_newWithSize2d(dim == 1 ? n_cand : n_slices, dim == 1 ? n_slices : n_cand)
                                 : THIntTensor_newWithSize2d(dim == 1 ? n_slices : n_cand, dim == 1 ? n_cand : n_slices);
    THIntTensor *it = transposed ? THIntTensor_newTranspose(ib, 0, 1) : THIntTensor_newClone(ib);
    THIntTensor_zero(it);
    THFloatTensor *ft = THFloatTensor_newWithSize2d(THIntTensor_size(it, 0), THIntTensor_size(it, 1));
    for (long s = 0; s < n_slices; s++) {
      long a = (s * 7) % 400, b = a + 1 + (s * 13) % 500;   /* a < b */
      long c = 450 + s % 50, d = c + 3 + s;                 /* c < d, both < cols */
      long cand[4] = {a, b, c, d};
      int val[4] = {5, 5, -5, -5};
      for (int k = 0; k < 4; k++) {
        long i = dim == 1 ? s : cand[k], j = dim == 1 ? cand[k] : s;
        THIntTensor_set2d(it, i, j, val[k]);
      }
    }
    THFloatTensor_copyInt(ft, it);

    THIntTensor *iv = THIntTensor_new(); THFloatTensor *fv = THFloatTensor_new();
    THLongTensor *ii = THLongTensor_new(), *fi = THLongTensor_new();
    for (int want_max = 0; want_max < 2; want_max++) {
      if (want_max) { THIntTensor_max(iv, ii, it, dim, keepdim); THFloatTensor_max(fv, fi, ft, dim, keepdim); }
      else          { THIntTensor_min(iv, ii, it, dim, keepdim); THFloatTensor_min(fv, fi, ft, dim, keepdim); }
      int ok = 1;
      for (long s = 0; s < n_slices && ok; s++) {
        long a = (s * 7) % 400, c = 450 + s % 50;
        long expect = want_max ? a : c;           /* the lower of the two ties */
        long gi = THLongTensor_data(ii)[s], gf = THLongTensor_data(fi)[s];
        ok = gi == expect && gf == expect;
        CHECK(ok, "%s ties: transposed=%d dim=%d keepdim=%d slice %ld: int idx %ld, float idx %ld, want %ld",
              want_max ? "max" : "min", transposed, dim, keepdim, s, gi, gf, expect);
      }
    }
    THIntTensor_free(ib); THIntTensor_free(it); THFloatTensor_free(ft);
    THIntTensor_free(iv); THFloatTensor_free(fv); THLongTensor_free(ii); THLongTensor_free(fi);
  }
}

/* LongTensor values beyond 32 bits survive arithmetic, reductions, sort and
 * conversions (LongTensor was 32-bit on Windows before Phase 1). */
static void misc_long_values_64bit(void)
{
  const int64_t T40 = (int64_t)1 << 40;
  const int64_t vals[5] = {T40, -T40 + 5, INT64_MAX / 4, 3, -((int64_t)1 << 33)};
  THLongTensor *a = THLongTensor_newWithSize1d(5), *r = THLongTensor_new();
  for (int i = 0; i < 5; i++) THLongTensor_set1d(a, i, vals[i]);

  CHECK(sizeof(int64_t) == 8 && THLongTensor_get1d(a, 0) == T40, "store/load 2^40");
  THLongTensor_mul(r, a, 3);
  CHECK(THLongTensor_get1d(r, 0) == 3 * T40 && THLongTensor_get1d(r, 4) == -3 * ((int64_t)1 << 33),
        "mul by 3: got %" PRId64, THLongTensor_get1d(r, 0));
  THLongTensor_cadd(r, a, 1, a);
  CHECK(THLongTensor_get1d(r, 1) == 2 * (-T40 + 5), "cadd");

  int64_t want = 0;
  for (int i = 0; i < 5; i++) want += vals[i];
  CHECK(THLongTensor_sumall(a) == want, "sumall: got %" PRId64 " want %" PRId64,
        (int64_t)THLongTensor_sumall(a), want);

  THLongTensor *mv = THLongTensor_new(), *mi = THLongTensor_new();
  THLongTensor_max(mv, mi, a, 0, 0);
  CHECK(THLongTensor_get1d(mv, 0) == INT64_MAX / 4 && THLongTensor_get1d(mi, 0) == 2, "max/argmax");

  THLongTensor *sv = THLongTensor_new(), *si = THLongTensor_new();
  THLongTensor_sort(sv, si, a, 0, 0);
  CHECK(THLongTensor_get1d(sv, 0) == -T40 + 5 && THLongTensor_get1d(sv, 4) == INT64_MAX / 4, "sort");

  THDoubleTensor *d = THDoubleTensor_new();
  THDoubleTensor_resize1d(d, 5);
  THDoubleTensor_copyLong(d, a);
  CHECK(THDoubleTensor_get1d(d, 0) == 1099511627776.0, "copy Long -> Double");
  THLongTensor_copyDouble(r, d);
  CHECK(THLongTensor_get1d(r, 4) == -((int64_t)1 << 33), "copy Double -> Long");

  THLongTensor_free(a); THLongTensor_free(r); THLongTensor_free(mv); THLongTensor_free(mi);
  THLongTensor_free(sv); THLongTensor_free(si); THDoubleTensor_free(d);
}

/* Vectorized float exp/tanh/sigmoid (Phase 3c): dense sweeps over each
 * function's whole range, including the fast-path boundaries where chunks
 * switch to libm, must stay within 4 ulp of the double-precision reference;
 * special values must match libm exactly (NaN-ness, +-inf, +-0 with sign). */
static void misc_float_vector_math(void)
{
  static const struct { const char *name; float lo, hi; } ranges[] = {
    {"exp", -104.0f, 89.0f}, {"tanh", -50.0f, 50.0f}, {"sigmoid", -110.0f, 110.0f}};
  const int64_t n = 2000003;   /* odd, so the scalar tail runs too */
  THFloatTensor *x = THFloatTensor_newWithSize1d(n), *y = THFloatTensor_new();
  for (int f = 0; f < 3; f++) {
    float *xd = THFloatTensor_data(x);
    for (int64_t i = 0; i < n; i++)
      xd[i] = ranges[f].lo + (ranges[f].hi - ranges[f].lo) * (float)((double)i / (double)(n - 1));
    if (f == 0) THFloatTensor_exp(y, x);
    else if (f == 1) THFloatTensor_tanh(y, x);
    else THFloatTensor_sigmoid(y, x);
    double worst = 0; float worstX = 0;
    for (int64_t i = 0; i < n; i++) {
      double xv = xd[i];
      double want = f == 0 ? exp(xv) : f == 1 ? tanh(xv) : 1.0 / (1.0 + exp(-xv));
      double u = th_test_ulps_float(THFloatTensor_data(y)[i], (float)want);
      if (u > worst) { worst = u; worstX = xd[i]; }
    }
    CHECK(worst <= 4, "%s: worst error %g ulp at x = %.9g", ranges[f].name, worst, (double)worstX);
    printf("    %-8s max error %g ulp over [%g, %g]\n", ranges[f].name, worst,
           (double)ranges[f].lo, (double)ranges[f].hi);
  }

  /* special values, each placed inside an otherwise ordinary chunk of 8 */
  const float specials[] = {NAN, INFINITY, -INFINITY, 0.0f, -0.0f, 1e-30f, -1e-30f, 1e-45f,
                            88.0f, -87.0f, 88.5f, -87.5f, 44.0f, -44.0f, 0.625f, -0.625f};
  const int ns = (int)(sizeof(specials) / sizeof(specials[0]));
  THFloatTensor *s = THFloatTensor_newWithSize1d(8 * ns);
  for (int k = 0; k < ns; k++)
    for (int j = 0; j < 8; j++) THFloatTensor_data(s)[8 * k + j] = j == 3 ? specials[k] : 0.5f * (float)j;
  for (int f = 0; f < 3; f++) {
    if (f == 0) THFloatTensor_exp(y, s);
    else if (f == 1) THFloatTensor_tanh(y, s);
    else THFloatTensor_sigmoid(y, s);
    for (int k = 0; k < ns; k++) {
      float xv = specials[k], got = THFloatTensor_data(y)[8 * k + 3];
      float want = f == 0 ? expf(xv) : f == 1 ? tanhf(xv) : 1.0f / (1.0f + expf(-xv));
      int ok = isnan(want) ? isnan(got)
             : (want == 0 || isinf(want)) ? (got == want && signbit(got) == signbit(want))
             : th_test_ulps_float(got, want) <= 4;
      CHECK(ok, "%s(%g): got %.9g want %.9g", ranges[f].name, (double)xv, (double)got, (double)want);
    }
  }
  THFloatTensor_free(x); THFloatTensor_free(y); THFloatTensor_free(s);
}

/* Vectorized float log: geometric sweep over every positive normal float
 * exponent (<= 4 ulp vs double log), and special values exactly as libm. */
static void misc_float_vector_log(void)
{
  const int64_t n = 2000003;
  THFloatTensor *x = THFloatTensor_newWithSize1d(n), *y = THFloatTensor_new();
  float *xd = THFloatTensor_data(x);
  const double lo = log(1.17549435e-38), hi = log(3.0e38);
  for (int64_t i = 0; i < n; i++) xd[i] = (float)exp(lo + (hi - lo) * (double)i / (double)(n - 1));
  THFloatTensor_log(y, x);
  double worst = 0; float worstX = 0;
  for (int64_t i = 0; i < n; i++) {
    double u = th_test_ulps_float(THFloatTensor_data(y)[i], (float)log((double)xd[i]));
    if (u > worst) { worst = u; worstX = xd[i]; }
  }
  CHECK(worst <= 4, "log: worst error %g ulp at x = %.9g", worst, (double)worstX);
  printf("    log      max error %g ulp over [1.2e-38, 3e38]\n", worst);

  const float specials[] = {NAN, INFINITY, -INFINITY, 0.0f, -0.0f, -1.0f, 1e-45f, 1e-39f,
                            1.0f, 2.0f, 0.5f, 0.70710677f, 0.70710683f, 3.4028235e38f, 1.17549435e-38f};
  const int ns = (int)(sizeof(specials) / sizeof(specials[0]));
  THFloatTensor *s = THFloatTensor_newWithSize1d(8 * ns);
  for (int k = 0; k < ns; k++)
    for (int j = 0; j < 8; j++) THFloatTensor_data(s)[8 * k + j] = j == 5 ? specials[k] : 1.5f + (float)j;
  THFloatTensor_log(y, s);
  for (int k = 0; k < ns; k++) {
    float xv = specials[k], got = THFloatTensor_data(y)[8 * k + 5], want = logf(xv);
    int ok = isnan(want) ? isnan(got)
           : (want == 0 || isinf(want)) ? (got == want && signbit(got) == signbit(want))
           : th_test_ulps_float(got, want) <= 4;
    CHECK(ok, "log(%g): got %.9g want %.9g", (double)xv, (double)got, (double)want);
  }
  THFloatTensor_free(x); THFloatTensor_free(y); THFloatTensor_free(s);
}

static void misc_suite(void)
{
  RUN(misc_shape_errors);
  RUN(misc_broadcasting);
  RUN(misc_copy_cross_type);
  RUN(misc_max_min_ties_lowest_index);
  RUN(misc_long_values_64bit);
  RUN(misc_float_vector_math);
  RUN(misc_float_vector_log);
}

/* ------------------------------------------------------------------------ */
/* 64-bit sizes: more than 2^31 elements (impossible on Windows before,    */
/* where sizes were 32-bit C long). Needs ~2 GB; skipped if allocation     */
/* fails.                                                                   */
/* ------------------------------------------------------------------------ */
static void large_over_2g_elements(void)
{
  const int64_t n = ((int64_t)1 << 31) + 17;          /* > INT32_MAX */
  THByteTensor *volatile t = NULL;
  th_jmp_buf jb;
  th_jmp_buf *prev = th_test_jmp;
  th_test_jmp = &jb;
  if (TH_TEST_SETJMP(jb) == 0) t = THByteTensor_newWithSize1d(n);
  th_test_jmp = prev;
  if (!t) {
    printf("    SKIP: could not allocate %" PRId64 " bytes (%s)\n", n, th_test_errmsg);
    return;
  }

  CHECK(THByteTensor_nElement(t) == n && THByteTensor_size(t, 0) == n,
        "nElement %" PRId64 ", size %" PRId64 ", want %" PRId64,
        (int64_t)THByteTensor_nElement(t), THByteTensor_size(t, 0), n);
  THByteTensor_fill(t, 1);
  THByteTensor_set1d(t, n - 1, 3);
  CHECK(THByteTensor_get1d(t, n - 1) == 3, "last element");

  /* accreal for Byte is int64_t: the sum no longer wraps at 2^31 */
  int64_t s = THByteTensor_sumall(t);
  CHECK(s == n + 2, "sumall: got %" PRId64 ", want %" PRId64, s, n + 2);

  /* a view whose storage offset is beyond 2^31 */
  THByteTensor *v = THByteTensor_newNarrow(t, 0, n - 10, 10);
  CHECK(THByteTensor_get1d(v, 9) == 3 && THByteTensor_sumall(v) == 12,
        "narrow view at offset %" PRId64, n - 10);

  /* a stride beyond 2^31 */
  const int64_t big = (int64_t)1 << 31;
  THByteTensor *w = THByteTensor_newWithStorage2d(THByteTensor_storage(t), 0, 2, big, 4, 1);
  THByteTensor_set2d(w, 1, 2, 7);
  CHECK(THByteTensor_data(t)[big + 2] == 7 && THByteTensor_stride(w, 0) == big,
        "2-D view with stride 2^31");

  THByteTensor_free(v);
  THByteTensor_free(w);
  THByteTensor_free(t);
}

static void large_suite(void)
{
  RUN(large_over_2g_elements);
}

/* ------------------------------------------------------------------------ */

typedef struct { const char *name; void (*run)(void); } th_suite;

static const th_suite suites[] = {
  {"byte", test_Byte_suite},     {"char", test_Char_suite},
  {"short", test_Short_suite},   {"int", test_Int_suite},
  {"long", test_Long_suite},     {"float", test_Float_suite},
  {"double", test_Double_suite}, {"misc", misc_suite},
  {"large", large_suite},
  {"iter", iter_suite},
};

int main(int argc, char **argv)
{
  /* unbuffered, so a crash still shows which test was running */
  setvbuf(stdout, NULL, _IONBF, 0);
  th_test_init();
  int ran = 0;
  for (size_t i = 0; i < sizeof(suites) / sizeof(suites[0]); i++) {
    if (argc > 1 && strcmp(argv[1], suites[i].name) != 0) continue;
    printf("[%s]\n", suites[i].name);
    suites[i].run();
    ran++;
  }
  if (!ran) {
    fprintf(stderr, "unknown suite '%s'\n", argv[1]);
    return 2;
  }
  printf("%ld checks, %d failed test(s)\n", th_test_checks, th_test_total_failed);
  return th_test_total_failed ? 1 : 0;
}
