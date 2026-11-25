#include "THIter.h"

#ifdef _OPENMP
#include <omp.h>
#endif

static int64_t THIter_numel(const THIterOperand *op)
{
  if (op->ndim == 0) return 0;
  int64_t n = 1;
  for (int d = 0; d < op->ndim; d++) n *= op->size[d];
  return n;
}

static void THIter_describe(const THIterOperand *op, char *buf, size_t len)
{
  size_t pos = 0;
  pos += (size_t)snprintf(buf + pos, len - pos, "[");
  for (int d = 0; d < op->ndim && pos < len; d++)
    pos += (size_t)snprintf(buf + pos, len - pos, d ? " x %" PRId64 : "%" PRId64, op->size[d]);
  if (pos < len) snprintf(buf + pos, len - pos, "]");
}

int THIter_broadcastShape(int nops, const THIterOperand *ops,
                          int *ndim, int64_t *shape, char *err, size_t errlen)
{
  int nd = 0;
  for (int k = 0; k < nops; k++) {
    if (ops[k].ndim > TH_ITER_MAX_DIMS) {
      snprintf(err, errlen, "tensor has %d dimensions; at most %d are supported",
               ops[k].ndim, TH_ITER_MAX_DIMS);
      return 0;
    }
    if (ops[k].ndim > nd) nd = ops[k].ndim;
  }
  for (int i = 0; i < nd; i++) shape[i] = 1;
  /* align trailing dimensions: shape index i <-> op dim i - (nd - op.ndim) */
  for (int k = 0; k < nops; k++) {
    const int off = nd - ops[k].ndim;
    for (int d = 0; d < ops[k].ndim; d++) {
      int64_t s = ops[k].size[d];
      int64_t *t = &shape[off + d];
      if (s == *t || s == 1) continue;
      if (*t == 1) { *t = s; continue; }
      char a[256], b[256];
      THIter_describe(&ops[0], a, sizeof(a));
      THIter_describe(&ops[k], b, sizeof(b));
      snprintf(err, errlen, "shapes %s and %s are not broadcastable (dimension %d: %"
               PRId64 " vs %" PRId64 ")", a, b, off + d, *t, s);
      return 0;
    }
  }
  *ndim = nd;
  return 1;
}

void THIter_build(THIter *it, int nops, const THIterOperand *ops)
{
  THArgCheck(nops >= 1 && nops <= TH_ITER_MAX_OPERANDS, 2,
             "THIter supports 1 to %d operands, got %d", TH_ITER_MAX_OPERANDS, nops);
  const THIterOperand *out = &ops[0];
  THArgCheck(out->ndim <= TH_ITER_MAX_DIMS, 3, "output has %d dimensions; at most %d are supported",
             out->ndim, TH_ITER_MAX_DIMS);

  /* every input must broadcast to exactly the output's shape */
  for (int k = 1; k < nops; k++) {
    const THIterOperand *in = &ops[k];
    int ok = in->ndim <= out->ndim;
    const int off = out->ndim - in->ndim;
    for (int d = 0; ok && d < in->ndim; d++)
      ok = in->size[d] == out->size[off + d] || in->size[d] == 1;
    /* an empty input (no elements) cannot be broadcast to a non-empty output */
    if (THIter_numel(in) == 0 && THIter_numel(out) != 0) ok = 0;
    if (!ok) {
      char a[256], b[256];
      THIter_describe(out, a, sizeof(a));
      THIter_describe(in, b, sizeof(b));
      THError("operand %d of shape %s cannot be broadcast to the output shape %s", k, b, a);
    }
  }

  it->nops = nops;
  it->numel = THIter_numel(out);
  for (int k = 0; k < nops; k++) it->data[k] = ops[k].data;
  if (it->numel == 0) {
    it->ndim = 0;
    return;
  }

  /* Collect the output's dimensions innermost-first, skipping size-1 dims.
     Input strides are 0 along broadcast dimensions. */
  int nd = 0;
  for (int d = out->ndim - 1; d >= 0; d--) {
    if (out->size[d] == 1) continue;
    it->shape[nd] = out->size[d];
    for (int k = 0; k < nops; k++) {
      const THIterOperand *op = &ops[k];
      const int od = d - (out->ndim - op->ndim);
      int64_t st = 0;
      if (od >= 0 && op->size[od] != 1) st = op->stride[od] * op->elsize;
      it->stride[nd][k] = st;
    }
    nd++;
  }

  /* Order dimensions so the output's smallest |stride| is innermost (stable
     insertion sort; ties broken by the inputs' strides). Pure re-nesting of the
     loops: every element is still paired with the same input elements. */
  for (int i = 1; i < nd; i++) {
    for (int j = i; j > 0; j--) {
      int swap = 0;
      for (int k = 0; k < nops; k++) {
        int64_t a = it->stride[j - 1][k], b = it->stride[j][k];
        if (a < 0) a = -a;
        if (b < 0) b = -b;
        if (a == b || a == 0 || b == 0) continue;  /* undecided: look at next operand */
        swap = b < a;
        break;
      }
      if (!swap) break;
      int64_t ts = it->shape[j]; it->shape[j] = it->shape[j - 1]; it->shape[j - 1] = ts;
      for (int k = 0; k < nops; k++) {
        int64_t t = it->stride[j][k]; it->stride[j][k] = it->stride[j - 1][k]; it->stride[j - 1][k] = t;
      }
    }
  }

  /* Coalesce: merge dim i+1 into dim i when, for every operand, stepping once
     along i+1 equals stepping size[i] times along i. */
  int w = 0;
  for (int i = 1; i < nd; i++) {
    int mergeable = 1;
    for (int k = 0; k < nops && mergeable; k++)
      mergeable = it->stride[i][k] == it->stride[w][k] * it->shape[w];
    if (mergeable) {
      it->shape[w] *= it->shape[i];
    } else {
      w++;
      it->shape[w] = it->shape[i];
      for (int k = 0; k < nops; k++) it->stride[w][k] = it->stride[i][k];
    }
  }
  it->ndim = nd ? w + 1 : 1;
  if (nd == 0) {   /* every dimension had size 1: a single element */
    it->shape[0] = 1;
    for (int k = 0; k < nops; k++) it->stride[0][k] = 0;
  }
}

/* Run elements [begin, end) in iteration order. */
static void THIter_runRange(const THIter *it, THIterLoop loop, void *ctx,
                            int64_t begin, int64_t end)
{
  const int nd = it->ndim, nops = it->nops;
  int64_t idx[TH_ITER_MAX_DIMS];
  char *ptr[TH_ITER_MAX_OPERANDS];

  int64_t rem = begin;
  for (int d = 0; d < nd; d++) {
    idx[d] = rem % it->shape[d];
    rem /= it->shape[d];
  }
  for (int k = 0; k < nops; k++) {
    char *p = it->data[k];
    for (int d = 0; d < nd; d++) p += idx[d] * it->stride[d][k];
    ptr[k] = p;
  }

  int64_t pos = begin;
  while (pos < end) {
    int64_t n = it->shape[0] - idx[0];
    if (n > end - pos) n = end - pos;
    loop(ptr, it->stride[0], n, ctx);
    pos += n;
    if (pos >= end) break;
    /* advance: finish the inner run, then carry into outer dimensions */
    for (int k = 0; k < nops; k++) ptr[k] += n * it->stride[0][k];
    idx[0] += n;
    for (int d = 0; d < nd - 1 && idx[d] == it->shape[d]; d++) {
      idx[d] = 0;
      idx[d + 1]++;
      for (int k = 0; k < nops; k++)
        ptr[k] += it->stride[d + 1][k] - it->shape[d] * it->stride[d][k];
    }
  }
}

void THIter_run(const THIter *it, THIterLoop loop, void *ctx, int64_t grain)
{
  const int64_t numel = it->numel;
  if (numel == 0) return;
#ifdef _OPENMP
  if (numel >= grain && grain > 0 && !omp_in_parallel() && omp_get_max_threads() > 1) {
    int64_t maxThreads = (numel + grain - 1) / grain;
    if (maxThreads > omp_get_max_threads()) maxThreads = omp_get_max_threads();
    #pragma omp parallel num_threads((int)maxThreads)
    {
      const int64_t nt = omp_get_num_threads(), tid = omp_get_thread_num();
      const int64_t chunk = (numel + nt - 1) / nt;
      const int64_t b = tid * chunk;
      const int64_t e = b + chunk < numel ? b + chunk : numel;
      if (b < e) THIter_runRange(it, loop, ctx, b, e);
    }
    return;
  }
#else
  (void)grain;
#endif
  THIter_runRange(it, loop, ctx, 0, numel);
}

/* [lo, hi) byte range touched by an operand */
static void THIter_span(const THIterOperand *op, char **lo, char **hi)
{
  char *a = op->data, *b = op->data;
  if (THIter_numel(op) > 0) {
    for (int d = 0; d < op->ndim; d++) {
      int64_t ext = (op->size[d] - 1) * op->stride[d] * op->elsize;
      if (ext < 0) a += ext; else b += ext;
    }
    b += op->elsize;
  }
  *lo = a;
  *hi = b;
}

int THIter_overlaps(const THIterOperand *a, const THIterOperand *b)
{
  if (THIter_numel(a) == 0 || THIter_numel(b) == 0) return 0;
  char *alo, *ahi, *blo, *bhi;
  THIter_span(a, &alo, &ahi);
  THIter_span(b, &blo, &bhi);
  return alo < bhi && blo < ahi;
}

int THIter_sameLayout(const THIterOperand *a, const THIterOperand *b)
{
  if (a->data != b->data || a->elsize != b->elsize || a->ndim != b->ndim) return 0;
  for (int d = 0; d < a->ndim; d++) {
    if (a->size[d] != b->size[d]) return 0;
    if (a->size[d] != 1 && a->stride[d] != b->stride[d]) return 0;
  }
  return 1;
}
