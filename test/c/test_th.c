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
jmp_buf *th_test_jmp = NULL;
char th_test_errmsg[1024];

static void th_test_on_error(const char *msg, void *data)
{
  (void)data;
  snprintf(th_test_errmsg, sizeof(th_test_errmsg), "%s", msg);
  if (th_test_jmp) longjmp(*th_test_jmp, 1);
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

static void misc_suite(void)
{
  RUN(misc_shape_errors);
  RUN(misc_known_issue_same_numel_flattens);
  RUN(misc_copy_cross_type);
  RUN(misc_max_min_ties_lowest_index);
}

/* ------------------------------------------------------------------------ */

typedef struct { const char *name; void (*run)(void); } th_suite;

static const th_suite suites[] = {
  {"byte", test_Byte_suite},     {"char", test_Char_suite},
  {"short", test_Short_suite},   {"int", test_Int_suite},
  {"long", test_Long_suite},     {"float", test_Float_suite},
  {"double", test_Double_suite}, {"misc", misc_suite},
};

int main(int argc, char **argv)
{
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
