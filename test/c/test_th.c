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

/* Mismatched element counts must raise a TH error, not crash. */
static void misc_shape_errors(void)
{
  THFloatTensor *a = THFloatTensor_newWithSize2d(4, 4);
  THFloatTensor *b = THFloatTensor_newWithSize2d(1, 4);
  THFloatTensor *r = THFloatTensor_new();
  THFloatTensor_fill(a, 1); THFloatTensor_fill(b, 1);
  EXPECT_TH_ERROR(THFloatTensor_cadd(r, a, 1, b), "inconsistent tensor size");
  EXPECT_TH_ERROR(THFloatTensor_cmul(r, a, b), "inconsistent tensor size");

  THFloatTensor *m1 = THFloatTensor_newWithSize2d(3, 4), *m2 = THFloatTensor_newWithSize2d(5, 2);
  THFloatTensor_zero(m1); THFloatTensor_zero(m2);
  EXPECT_TH_ERROR(THFloatTensor_addmm(r, 0, r, 1, m1, m2), "size mismatch");
  THFloatTensor_free(a); THFloatTensor_free(b); THFloatTensor_free(r);
  THFloatTensor_free(m1); THFloatTensor_free(m2);
}

/* KNOWN ISSUE (to be fixed in Phase 2, broadcasting): operands with the same
 * number of elements but different shapes are silently treated as flat
 * arrays. This test pins today's behavior so the change is deliberate; when
 * broadcasting lands, flip it to EXPECT_TH_ERROR. */
static void misc_known_issue_same_numel_flattens(void)
{
  THFloatTensor *a = THFloatTensor_newWithSize2d(4, 4);
  THFloatTensor *b = THFloatTensor_newWithSize2d(2, 8);
  THFloatTensor *r = THFloatTensor_new();
  THFloatTensor_fill(a, 1);
  for (int i = 0; i < 16; i++) THFloatTensor_data(b)[i] = (float)i;
  THFloatTensor_cadd(r, a, 1, b);
  CHECK(THFloatTensor_size(r, 0) == 4 && THFloatTensor_size(r, 1) == 4, "result takes a's shape");
  CHECK(THFloatTensor_get2d(r, 1, 0) == 5, "b read as flat: r[1][0] = 1 + b.flat[4]");
  THFloatTensor_free(a); THFloatTensor_free(b); THFloatTensor_free(r);
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

static void misc_suite(void)
{
  RUN(misc_shape_errors);
  RUN(misc_known_issue_same_numel_flattens);
  RUN(misc_copy_cross_type);
  RUN(misc_max_min_ties_lowest_index);
  RUN(misc_long_values_64bit);
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
