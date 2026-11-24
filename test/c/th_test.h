/* Minimal, dependency-free test harness for the TH C library.
 *
 *   CHECK(cond, fmt, ...)          record a failure (with file:line) if !cond
 *   EXPECT_TH_ERROR(stmt, substr)  stmt must raise a TH error whose message
 *                                  contains substr (NULL: any message)
 *   RUN(fn)                        run one test function and report it
 *
 * TH errors are turned into longjmp()s by the error handlers installed in
 * th_test_init(), so a failing argument check aborts only the current test.
 */
#ifndef TH_TEST_H
#define TH_TEST_H

#include <math.h>
#include <setjmp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "TH.h"

extern int th_test_failures;        /* failed checks in the current test */
extern int th_test_total_failed;    /* failed tests overall */
extern long th_test_checks;         /* checks performed overall */
extern const char *th_test_current; /* name of the running test */
extern jmp_buf *th_test_jmp;        /* where a TH error jumps to, or NULL */
extern char th_test_errmsg[1024];   /* message of the last TH error */

void th_test_init(void);

#define CHECK(cond, ...)                                                    \
  do {                                                                      \
    th_test_checks++;                                                       \
    if (!(cond)) {                                                          \
      if (++th_test_failures <= 10) {                                       \
        fprintf(stderr, "    FAIL %s:%d: ", __FILE__, __LINE__);            \
        fprintf(stderr, __VA_ARGS__);                                       \
        fputc('\n', stderr);                                                \
      }                                                                     \
    }                                                                       \
  } while (0)

#define EXPECT_TH_ERROR(stmt, substr)                                       \
  do {                                                                      \
    jmp_buf th_jb_;                                                         \
    jmp_buf *th_prev_ = th_test_jmp;                                        \
    volatile int th_raised_ = 0;                                            \
    th_test_errmsg[0] = '\0';                                               \
    th_test_jmp = &th_jb_;                                                  \
    if (setjmp(th_jb_) == 0) { stmt; } else { th_raised_ = 1; }             \
    th_test_jmp = th_prev_;                                                 \
    CHECK(th_raised_, "expected a TH error from: %s", #stmt);               \
    if (th_raised_ && (substr) != NULL)                                     \
      CHECK(strstr(th_test_errmsg, (substr)) != NULL,                       \
            "error \"%s\" does not mention \"%s\"", th_test_errmsg, (substr)); \
  } while (0)

/* Indirection so that RUN(TT(x)) reports the expanded function name. */
#define RUN(fn) RUN_(fn)
#define RUN_(fn)                                                            \
  do {                                                                      \
    th_test_current = #fn;                                                  \
    th_test_failures = 0;                                                   \
    fn();                                                                   \
    printf("  %-40s %s\n", #fn, th_test_failures ? "FAIL" : "ok");         \
    if (th_test_failures) th_test_total_failed++;                           \
  } while (0)

/* Deterministic pseudo-random numbers (no global libc state). */
static inline double th_test_uniform(uint64_t *state, double lo, double hi)
{
  *state = *state * 6364136223846793005ULL + 1442695040888963407ULL;
  return lo + (hi - lo) * ((double)(*state >> 11) / 9007199254740992.0);
}

/* Distance in units-in-the-last-place between two floats / doubles. */
static inline double th_test_ulps_float(float a, float b)
{
  if (isnan(a) || isnan(b)) return (isnan(a) && isnan(b)) ? 0 : INFINITY;
  int32_t ia, ib;
  memcpy(&ia, &a, 4); memcpy(&ib, &b, 4);
  if (ia < 0) ia = INT32_MIN - ia;
  if (ib < 0) ib = INT32_MIN - ib;
  return fabs((double)ia - (double)ib);
}

static inline double th_test_ulps_double(double a, double b)
{
  if (isnan(a) || isnan(b)) return (isnan(a) && isnan(b)) ? 0 : INFINITY;
  int64_t ia, ib;
  memcpy(&ia, &a, 8); memcpy(&ib, &b, 8);
  if (ia < 0) ia = INT64_MIN - ia;
  if (ib < 0) ib = INT64_MIN - ib;
  return fabs((double)ia - (double)ib);
}

#endif
