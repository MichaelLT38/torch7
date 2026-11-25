#ifndef TH_ITER_INC
#define TH_ITER_INC

/* THIter: one strided iteration engine for element-wise operations.
 *
 * Given an output and up to TH_ITER_MAX_OPERANDS-1 inputs (each a data pointer,
 * element size and strided shape), it
 *   - checks shapes (inputs broadcast to the output shape, NumPy rules),
 *   - drops size-1 dimensions, orders dimensions by the output's strides so the
 *     innermost loop walks memory in order, and merges ("coalesces") adjacent
 *     dimensions that are contiguous for every operand,
 *   - runs a typed inner-loop function over the remaining elements, splitting
 *     the work across OpenMP threads when it is large enough.
 *
 * The inner loop sees one contiguous run of the innermost dimension at a time:
 *     loop(data, strides, n, ctx)
 * where data[k] points at operand k's first element of the run and strides[k]
 * is operand k's stride in BYTES along it (0 for a broadcast operand).
 *
 * Operand 0 is the output. It must already have the broadcast shape (callers
 * resize it first; see THIter_broadcastShape). Element-wise kernels are
 * order-independent, so traversal order and parallel splitting do not change
 * results, provided the output does not partially overlap an input
 * (THIter_overlaps lets callers detect that and fall back).
 */

#include "THGeneral.h"

#define TH_ITER_MAX_DIMS 32
#define TH_ITER_MAX_OPERANDS 4

typedef struct {
  char *data;            /* first element */
  int64_t elsize;        /* bytes per element */
  int ndim;              /* 0 means empty (legacy TH convention) */
  const int64_t *size;   /* ndim sizes */
  const int64_t *stride; /* ndim strides, in elements */
} THIterOperand;

typedef struct {
  int nops;                                  /* operand 0 is the output */
  int ndim;                                  /* iteration dims; dim 0 is innermost */
  int64_t numel;
  int64_t shape[TH_ITER_MAX_DIMS];
  char *data[TH_ITER_MAX_OPERANDS];
  int64_t stride[TH_ITER_MAX_DIMS][TH_ITER_MAX_OPERANDS];  /* bytes */
} THIter;

typedef void (*THIterLoop)(char **data, const int64_t *strides, int64_t n, void *ctx);

/* Broadcast shape of ops[0..nops-1] (NumPy rules: align trailing dimensions;
 * each pair of sizes must be equal or one of them 1). On success returns 1 and
 * writes *ndim and shape[] (outermost first, like THTensor). On failure returns
 * 0 and writes a description to err. */
TH_API int THIter_broadcastShape(int nops, const THIterOperand *ops,
                                 int *ndim, int64_t *shape, char *err, size_t errlen);

/* Build an iterator. ops[0] (the output) must have exactly the broadcast shape
 * of the inputs ops[1..nops-1]; raises a TH error otherwise. */
TH_API void THIter_build(THIter *it, int nops, const THIterOperand *ops);

/* Run `loop` over every element. Uses OpenMP threads when numel >= grain and
 * we are not already inside a parallel region. */
TH_API void THIter_run(const THIter *it, THIterLoop loop, void *ctx, int64_t grain);

/* 1 if the memory spanned by a and b overlaps (conservative: compares the
 * address ranges, not the exact elements). */
TH_API int THIter_overlaps(const THIterOperand *a, const THIterOperand *b);

/* 1 if both operands describe exactly the same elements in the same layout. */
TH_API int THIter_sameLayout(const THIterOperand *a, const THIterOperand *b);

#endif
