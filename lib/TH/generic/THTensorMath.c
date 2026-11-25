#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THTensorMath.c"
#else

#ifndef NAN
  #define NAN (nan(NULL))
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#define TH_OMP_OVERHEAD_THRESHOLD 100000

#ifdef _OPENMP

#define PRAGMA(P) _Pragma(#P)

#define TH_TENSOR_APPLY_CONTIG(TYPE, TENSOR, CODE) \
{ \
  ptrdiff_t TH_TENSOR_size = THTensor_(nElement)(TENSOR); \
  PRAGMA(omp parallel if (TH_TENSOR_size > TH_OMP_OVERHEAD_THRESHOLD)) \
  { \
    size_t num_threads = omp_get_num_threads(); \
    size_t tid = omp_get_thread_num(); \
    ptrdiff_t TH_TENSOR_offset = tid * (TH_TENSOR_size / num_threads); \
    ptrdiff_t TH_TENSOR_end = tid == num_threads - 1 ? TH_TENSOR_size : \
      TH_TENSOR_offset + TH_TENSOR_size / num_threads; \
    ptrdiff_t TENSOR##_len = TH_TENSOR_end - TH_TENSOR_offset; \
    TYPE *TENSOR##_data = THTensor_(data)(TENSOR) + TH_TENSOR_offset; \
    CODE \
  } \
}
#else
#define TH_TENSOR_APPLY_CONTIG(TYPE, TENSOR, CODE) \
{ \
  TYPE *TENSOR##_data = THTensor_(data)(TENSOR); \
  ptrdiff_t TENSOR##_len = THTensor_(nElement)(TENSOR); \
  CODE \
}
#endif

#ifdef _OPENMP
#define TH_TENSOR_APPLY2_CONTIG(TYPE1, TENSOR1, TYPE2, TENSOR2, CODE) \
{ \
  ptrdiff_t TH_TENSOR_size = THTensor_(nElement)(TENSOR1); \
  PRAGMA(omp parallel if (TH_TENSOR_size > TH_OMP_OVERHEAD_THRESHOLD)) \
  { \
    size_t num_threads = omp_get_num_threads(); \
    size_t tid = omp_get_thread_num(); \
    ptrdiff_t TH_TENSOR_offset = tid * (TH_TENSOR_size / num_threads); \
    ptrdiff_t TH_TENSOR_end = tid == num_threads - 1 ? TH_TENSOR_size : \
      TH_TENSOR_offset + TH_TENSOR_size / num_threads; \
    ptrdiff_t TENSOR1##_len = TH_TENSOR_end - TH_TENSOR_offset; \
    TYPE1 *TENSOR1##_data = THTensor_(data)(TENSOR1) + TH_TENSOR_offset; \
    TYPE2 *TENSOR2##_data = THTensor_(data)(TENSOR2) + TH_TENSOR_offset; \
    CODE \
  } \
}
#else
#define TH_TENSOR_APPLY2_CONTIG(TYPE1, TENSOR1, TYPE2, TENSOR2, CODE) \
{ \
  TYPE1 *TENSOR1##_data = THTensor_(data)(TENSOR1); \
  TYPE2 *TENSOR2##_data = THTensor_(data)(TENSOR2); \
  ptrdiff_t TENSOR1##_len = THTensor_(nElement)(TENSOR1); \
  CODE \
}
#endif

#ifdef _OPENMP
#define TH_TENSOR_APPLY3_CONTIG(TYPE1, TENSOR1, TYPE2, TENSOR2, TYPE3, TENSOR3, CODE) \
{ \
  ptrdiff_t TH_TENSOR_size = THTensor_(nElement)(TENSOR1); \
  PRAGMA(omp parallel if (TH_TENSOR_size > TH_OMP_OVERHEAD_THRESHOLD)) \
  { \
    size_t num_threads = omp_get_num_threads(); \
    size_t tid = omp_get_thread_num(); \
    ptrdiff_t TH_TENSOR_offset = tid * (TH_TENSOR_size / num_threads); \
    ptrdiff_t TH_TENSOR_end = tid == num_threads - 1 ? TH_TENSOR_size : \
      TH_TENSOR_offset + TH_TENSOR_size / num_threads; \
    ptrdiff_t TENSOR1##_len = TH_TENSOR_end - TH_TENSOR_offset; \
    TYPE1 *TENSOR1##_data = THTensor_(data)(TENSOR1) + TH_TENSOR_offset; \
    TYPE2 *TENSOR2##_data = THTensor_(data)(TENSOR2) + TH_TENSOR_offset; \
    TYPE3 *TENSOR3##_data = THTensor_(data)(TENSOR3) + TH_TENSOR_offset; \
    CODE \
  } \
}
#else
#define TH_TENSOR_APPLY3_CONTIG(TYPE1, TENSOR1, TYPE2, TENSOR2, TYPE3, TENSOR3, CODE) \
{ \
  TYPE1 *TENSOR1##_data = THTensor_(data)(TENSOR1); \
  TYPE2 *TENSOR2##_data = THTensor_(data)(TENSOR2); \
  TYPE3 *TENSOR3##_data = THTensor_(data)(TENSOR3); \
  ptrdiff_t TENSOR1##_len = THTensor_(nElement)(TENSOR1); \
  CODE \
}
#endif

/* ---------------------------------------------------------------------------
 * THIter paths for element-wise ops (Phase 2b).
 *
 * Used when every operand has the output's shape (legacy calls with equal
 * element counts but different shapes keep the old path) and the output does
 * not partially overlap an input (then traversal order would be observable).
 * The kernels evaluate exactly the same expression as the legacy
 * TH_TENSOR_APPLY code, so results are bit-identical; what changes is speed
 * on strided tensors (dimension coalescing + OpenMP). Fully contiguous calls
 * keep their existing vectorized paths.
 * ------------------------------------------------------------------------- */
#define TH_ITER_GRAIN 32768

static int THTensor_(iterate2)(THTensor *r_, THTensor *a, THIterLoop loop, void *ctx)
{
  if (!THTensor_(isSameSizeAs)(r_, a) || !THTensor_(iterSafe)(r_, a)) return 0;
  THIterOperand ops[2] = {THTensor_(iterOperand)(r_), THTensor_(iterOperand)(a)};
  THIter it;
  THIter_build(&it, 2, ops);
  THIter_run(&it, loop, ctx, TH_ITER_GRAIN);
  return 1;
}

static int THTensor_(iterate3)(THTensor *r_, THTensor *a, THTensor *b, THIterLoop loop, void *ctx)
{
  if (!THTensor_(isSameSizeAs)(r_, a) || !THTensor_(isSameSizeAs)(r_, b) ||
      !THTensor_(iterSafe)(r_, a) || !THTensor_(iterSafe)(r_, b)) return 0;
  THIterOperand ops[3] = {THTensor_(iterOperand)(r_), THTensor_(iterOperand)(a),
                          THTensor_(iterOperand)(b)};
  THIter it;
  THIter_build(&it, 3, ops);
  THIter_run(&it, loop, ctx, TH_ITER_GRAIN);
  return 1;
}

/* Kernel generators: EXPR may use x (and y for binary ops) and value, the
   real scalar passed through ctx (0 when ctx is NULL). When the run is
   contiguous for every operand a typed loop is used, which the compiler can
   vectorize; the per-element arithmetic is the same (no FMA contraction and
   no reassociation in these flags), so results are identical either way. */
#define TH_ITER_CONTIG(S) ((S) == (int64_t)sizeof(real))

#define TH_ITER_KERNEL2(NAME, EXPR)                                            \
  static void THTensor_(NAME)(char **d_, const int64_t *s_, int64_t n_, void *ctx_) \
  {                                                                           \
    const real value = ctx_ ? *(const real *)ctx_ : (real)0;                  \
    (void)value;                                                              \
    char *rp_ = d_[0], *xp_ = d_[1];                                          \
    if (TH_ITER_CONTIG(s_[0]) && TH_ITER_CONTIG(s_[1])) {                      \
      real *r_ = (real *)rp_;                                                 \
      const real *xa_ = (const real *)xp_;                                    \
      for (int64_t i_ = 0; i_ < n_; i_++) {                                   \
        const real x = xa_[i_];                                               \
        r_[i_] = (EXPR);                                                      \
      }                                                                       \
      return;                                                                 \
    }                                                                         \
    for (int64_t i_ = 0; i_ < n_; i_++, rp_ += s_[0], xp_ += s_[1]) {         \
      const real x = *(const real *)xp_;                                      \
      *(real *)rp_ = (EXPR);                                                  \
    }                                                                         \
  }

#define TH_ITER_KERNEL3(NAME, EXPR)                                            \
  static void THTensor_(NAME)(char **d_, const int64_t *s_, int64_t n_, void *ctx_) \
  {                                                                           \
    const real value = ctx_ ? *(const real *)ctx_ : (real)0;                  \
    (void)value;                                                              \
    char *rp_ = d_[0], *xp_ = d_[1], *yp_ = d_[2];                            \
    if (TH_ITER_CONTIG(s_[0]) && TH_ITER_CONTIG(s_[1]) && TH_ITER_CONTIG(s_[2])) { \
      real *r_ = (real *)rp_;                                                 \
      const real *xa_ = (const real *)xp_, *ya_ = (const real *)yp_;          \
      for (int64_t i_ = 0; i_ < n_; i_++) {                                   \
        const real x = xa_[i_], y = ya_[i_];                                  \
        r_[i_] = (EXPR);                                                      \
      }                                                                       \
      return;                                                                 \
    }                                                                         \
    for (int64_t i_ = 0; i_ < n_; i_++, rp_ += s_[0], xp_ += s_[1], yp_ += s_[2]) { \
      const real x = *(const real *)xp_, y = *(const real *)yp_;              \
      *(real *)rp_ = (EXPR);                                                  \
    }                                                                         \
  }

TH_ITER_KERNEL2(addLoop, x + value)
TH_ITER_KERNEL2(mulLoop, x * value)
TH_ITER_KERNEL2(divLoop, x / value)
TH_ITER_KERNEL3(caddLoop, x + value * y)
TH_ITER_KERNEL3(cmulLoop, x * y)
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
TH_ITER_KERNEL3(cdivLoop, x / y)
#endif

void THTensor_(fill)(THTensor *r_, real value)
{
  if (THTensor_(isContiguous)(r_) || THTensor_(isTransposed)(r_)) {
    TH_TENSOR_APPLY_CONTIG(real, r_, THVector_(fill)(r__data, value, r__len););
  } else {
    TH_TENSOR_APPLY(real, r_,
      if (r__stride == 1) {
        THVector_(fill)(r__data, value, r__size);
	r__i = r__size;
	r__data += r__stride * r__size;
	break;
      } else {
        *r__data = value;
      }
      );
  }
}

void THTensor_(zero)(THTensor *r_)
{
  THTensor_(fill)(r_, 0);
}

void THTensor_(maskedFill)(THTensor *tensor, THByteTensor *mask, real value)
{
  TH_TENSOR_APPLY2(real, tensor, unsigned char, mask,
                   if (*mask_data > 1)
                   {
                     THFree(mask_counter);
                     THFree(tensor_counter);
                     THError("Mask tensor can take 0 and 1 values only");
                   }
                   else if (*mask_data == 1)
                   {
                     *tensor_data = value;
                   });
}

void THTensor_(maskedCopy)(THTensor *tensor, THByteTensor *mask, THTensor* src )
{
  THTensor *srct = THTensor_(newContiguous)(src);
  real *src_data = THTensor_(data)(srct);
  ptrdiff_t cntr = 0;
  ptrdiff_t nelem = THTensor_(nElement)(srct);
  if (THTensor_(nElement)(tensor) != THByteTensor_nElement(mask))
  {
    THTensor_(free)(srct);
    THError("Number of elements of destination tensor != Number of elements in mask");
  }
  TH_TENSOR_APPLY2(real, tensor, unsigned char, mask,
                   if (*mask_data > 1)
                   {
                     THTensor_(free)(srct);
                     THFree(mask_counter);
                     THFree(tensor_counter);
                     THError("Mask tensor can take 0 and 1 values only");
                   }
                   else if (*mask_data == 1)
                   {
                     if (cntr == nelem)
                     {
                       THTensor_(free)(srct);
                       THFree(mask_counter);
                       THFree(tensor_counter);
                       THError("Number of elements of src < number of ones in mask");
                     }
                     *tensor_data = *src_data;
                     src_data++;
                     cntr++;
                   });
  THTensor_(free)(srct);
}

void THTensor_(maskedSelect)(THTensor *tensor, THTensor *src, THByteTensor *mask)
{
  ptrdiff_t numel = THByteTensor_sumall(mask);
  real *tensor_data;

#ifdef DEBUG
  THAssert(numel <= INT64_MAX);
#endif
  THTensor_(resize1d)(tensor,numel);
  tensor_data = THTensor_(data)(tensor);
  TH_TENSOR_APPLY2(real, src, unsigned char, mask,
                   if (*mask_data > 1)
                   {
                     THFree(mask_counter);
                     THFree(src_counter);
                     THError("Mask tensor can take 0 and 1 values only");
                   }
                   else if (*mask_data == 1)
                   {
                     *tensor_data = *src_data;
                     tensor_data++;
                   });
}

// Finds non-zero elements of a tensor and returns their subscripts
void THTensor_(nonzero)(THLongTensor *subscript, THTensor *tensor)
{
  ptrdiff_t numel = 0;
  int64_t *subscript_data;
  int64_t i = 0;
  int64_t dim;
  int64_t div = 1;
#ifdef TH_REAL_IS_HALF
#define IS_NONZERO(val) ((val.x & 0x7fff) != 0)
#else
#define IS_NONZERO(val) ((val)!=0)
#endif

  /* First Pass to determine size of subscripts */
  TH_TENSOR_APPLY(real, tensor,
                  if IS_NONZERO(*tensor_data) {
                    ++numel;
                  });
#ifdef DEBUG
  THAssert(numel <= INT64_MAX);
#endif
  THLongTensor_resize2d(subscript, numel, tensor->nDimension);

  /* Second pass populates subscripts */
  subscript_data = THLongTensor_data(subscript);
  TH_TENSOR_APPLY(real, tensor,
                  if IS_NONZERO(*tensor_data) {
                    div = 1;

                    for (dim = tensor->nDimension - 1; dim >= 0; dim--) {
                      *(subscript_data + dim) = (i/div) % tensor->size[dim];
                      div *= tensor->size[dim];
                    }

                    subscript_data += tensor->nDimension;
                  }
                  ++i;);
}

void THTensor_(indexSelect)(THTensor *tensor, THTensor *src, int dim, THLongTensor *index)
{
  ptrdiff_t i, numel;
  THLongStorage *newSize;
  THTensor *tSlice, *sSlice;
  int64_t *index_data;
  real *tensor_data, *src_data;

  THArgCheck(index->nDimension == 1, 3, "Index is supposed to be a vector");
  THArgCheck(dim < src->nDimension, 4,"Indexing dim %d is out of bounds of tensor", dim + TH_INDEX_BASE);
  THArgCheck(src->nDimension > 0,2,"Source tensor is empty");

  numel = THLongTensor_nElement(index);

  newSize = THLongStorage_newWithSize(src->nDimension);
  THLongStorage_rawCopy(newSize,src->size);
#ifdef DEBUG
  THAssert(numel <= INT64_MAX);
#endif
  newSize->data[dim] = numel;
  THTensor_(resize)(tensor,newSize,NULL);
  THLongStorage_free(newSize);

  index = THLongTensor_newContiguous(index);
  index_data = THLongTensor_data(index);

  if (dim == 0 && THTensor_(isContiguous)(src) && THTensor_(isContiguous)(tensor))
  {
    tensor_data = THTensor_(data)(tensor);
    src_data = THTensor_(data)(src);
    ptrdiff_t rowsize = THTensor_(nElement)(src) / src->size[0];

    // check that the indices are within range
    int64_t max = src->size[0] - 1 + TH_INDEX_BASE;
    for (i=0; i<numel; i++) {
      if (index_data[i] < TH_INDEX_BASE || index_data[i] > max) {
        THLongTensor_free(index);
        THError("index out of range");
      }
    }

    if (src->nDimension == 1) {
      #pragma omp parallel for if(numel > TH_OMP_OVERHEAD_THRESHOLD) private(i)
      for (i=0; i<numel; i++)
        tensor_data[i] = src_data[index_data[i] - TH_INDEX_BASE];
    } else {
      #pragma omp parallel for if(numel*rowsize > TH_OMP_OVERHEAD_THRESHOLD) private(i)
      for (i=0; i<numel; i++)
        memcpy(tensor_data + i*rowsize, src_data + (index_data[i] - TH_INDEX_BASE)*rowsize, rowsize*sizeof(real));
    }
  }
  else if (src->nDimension == 1)
  {
    for (i=0; i<numel; i++)
      THTensor_(set1d)(tensor,i,THTensor_(get1d)(src,index_data[i] - TH_INDEX_BASE));
  }
  else
  {
    for (i=0; i<numel; i++)
    {
      tSlice = THTensor_(new)();
      sSlice = THTensor_(new)();
      THTensor_(select)(tSlice, tensor, dim, i);
      THTensor_(select)(sSlice, src, dim, index_data[i] - TH_INDEX_BASE);
      THTensor_(copy)(tSlice, sSlice);
      THTensor_(free)(tSlice);
      THTensor_(free)(sSlice);
    }
  }

  THLongTensor_free(index);
}

void THTensor_(indexCopy)(THTensor *tensor, int dim, THLongTensor *index, THTensor *src)
{
  ptrdiff_t i, numel;
  THTensor *tSlice, *sSlice;
  int64_t *index_data;

  numel = THLongTensor_nElement(index);
  THArgCheck(index->nDimension == 1, 3, "Index is supposed to be a vector");
  THArgCheck(dim < src->nDimension, 4, "Indexing dim %d is out of bounds of tensor", dim + TH_INDEX_BASE);
  THArgCheck(numel == src->size[dim],4,"Number of indices should be equal to source:size(dim)");

  index = THLongTensor_newContiguous(index);
  index_data = THLongTensor_data(index);

  if (tensor->nDimension > 1 )
  {
    tSlice = THTensor_(new)();
    sSlice = THTensor_(new)();

    for (i=0; i<numel; i++)
    {
      THTensor_(select)(tSlice, tensor, dim, index_data[i] - TH_INDEX_BASE);
      THTensor_(select)(sSlice, src, dim, i);
      THTensor_(copy)(tSlice, sSlice);
    }

    THTensor_(free)(tSlice);
    THTensor_(free)(sSlice);
  }
  else
  {
    for (i=0; i<numel; i++)
    {
      THTensor_(set1d)(tensor, index_data[i] - TH_INDEX_BASE, THTensor_(get1d)(src,i));
    }
  }
  THLongTensor_free(index);
}

void THTensor_(indexAdd)(THTensor *tensor, int dim, THLongTensor *index, THTensor *src)
{
  ptrdiff_t i, numel;
  THTensor *tSlice, *sSlice;
  int64_t *index_data;

  numel = THLongTensor_nElement(index);
  THArgCheck(index->nDimension == 1, 3, "Index is supposed to be a vector");
  THArgCheck(dim < src->nDimension, 4,"Indexing dim %d is out of bounds of tensor", dim + TH_INDEX_BASE);
  THArgCheck(numel == src->size[dim],4,"Number of indices should be equal to source:size(dim)");

  index = THLongTensor_newContiguous(index);
  index_data = THLongTensor_data(index);

  if (tensor->nDimension > 1)
  {
    tSlice = THTensor_(new)();
    sSlice = THTensor_(new)();

    for (i=0; i<numel; i++)
    {
      THTensor_(select)(tSlice, tensor, dim, index_data[i] - TH_INDEX_BASE);
      THTensor_(select)(sSlice, src, dim, i);
      THTensor_(cadd)(tSlice, tSlice, 1.0, sSlice);
    }

    THTensor_(free)(tSlice);
    THTensor_(free)(sSlice);
  }
  else
  {
    for (i=0; i<numel; i++)
    {
      THTensor_(set1d)(tensor,
              index_data[i] - TH_INDEX_BASE,
              THTensor_(get1d)(src,i) + THTensor_(get1d)(tensor,index_data[i] - TH_INDEX_BASE));
    }
  }
  THLongTensor_free(index);
}

void THTensor_(indexFill)(THTensor *tensor, int dim, THLongTensor *index, real val)
{
  ptrdiff_t i, numel;
  THTensor *tSlice;
  int64_t *index_data;

  numel = THLongTensor_nElement(index);
  THArgCheck(index->nDimension == 1, 3, "Index is supposed to be a vector");
  THArgCheck(dim < tensor->nDimension, 4,"Indexing dim %d is out of bounds of tensor", dim + TH_INDEX_BASE);

  index = THLongTensor_newContiguous(index);
  index_data = THLongTensor_data(index);

  for (i=0; i<numel; i++)
  {
    if (tensor->nDimension > 1)
    {
      tSlice = THTensor_(new)();
      THTensor_(select)(tSlice, tensor,dim,index_data[i] - TH_INDEX_BASE);
      THTensor_(fill)(tSlice, val);
      THTensor_(free)(tSlice);
    }
    else
    {
      THTensor_(set1d)(tensor, index_data[i] - TH_INDEX_BASE, val);
    }
  }
  THLongTensor_free(index);
}

void THTensor_(gather)(THTensor *tensor, THTensor *src, int dim, THLongTensor *index)
{
  int64_t elems_per_row, i, idx;

  THArgCheck(THTensor_(nDimension)(src) == THTensor_(nDimension)(tensor), 2,
             "Input tensor must have same dimensions as output tensor");
  THArgCheck(dim < THTensor_(nDimension)(tensor), 3, "Index dimension is out of bounds");
  THArgCheck(THLongTensor_nDimension(index) == THTensor_(nDimension)(src), 4,
             "Index tensor must have same dimensions as input tensor");

  elems_per_row = THLongTensor_size(index, dim);

  TH_TENSOR_DIM_APPLY3(real, tensor, real, src, int64_t, index, dim,
                       for (i = 0; i < elems_per_row; ++i)
                       {
                         idx = *(index_data + i*index_stride);
                         if (idx < TH_INDEX_BASE || idx >= src_size + TH_INDEX_BASE)
                         {
                           THFree(TH_TENSOR_DIM_APPLY_counter);
                           THError("Invalid index in gather");
                         }
                         *(tensor_data + i*tensor_stride) = src_data[(idx - TH_INDEX_BASE) * src_stride];
                       })
}

void THTensor_(scatter)(THTensor *tensor, int dim, THLongTensor *index, THTensor *src)
{
  int64_t elems_per_row, i, idx;

  THArgCheck(dim < THTensor_(nDimension)(tensor), 2, "Index dimension is out of bounds");
  THArgCheck(THLongTensor_nDimension(index) == THTensor_(nDimension)(tensor), 3,
             "Index tensor must have same dimensions as output tensor");
  THArgCheck(THTensor_(nDimension)(src) == THTensor_(nDimension)(tensor), 4,
             "Input tensor must have same dimensions as output tensor");

  elems_per_row = THLongTensor_size(index, dim);

  TH_TENSOR_DIM_APPLY3(real, tensor, real, src, int64_t, index, dim,
                       for (i = 0; i < elems_per_row; ++i)
                       {
                         idx = *(index_data + i*index_stride);
                         if (idx < TH_INDEX_BASE || idx >= tensor_size + TH_INDEX_BASE)
                         {
                           THFree(TH_TENSOR_DIM_APPLY_counter);
                           THError("Invalid index in scatter");
                         }
                         tensor_data[(idx - TH_INDEX_BASE) * tensor_stride] = *(src_data + i*src_stride);
                       })
}

void THTensor_(scatterAdd)(THTensor *tensor, int dim, THLongTensor *index, THTensor *src)
{
  int64_t elems_per_row, i, idx;

  THArgCheck(dim < THTensor_(nDimension)(tensor), 2, "Index dimension is out of bounds");
  THArgCheck(THLongTensor_nDimension(index) == THTensor_(nDimension)(tensor), 3,
             "Index tensor must have same dimensions as output tensor");
  THArgCheck(THTensor_(nDimension)(src) == THTensor_(nDimension)(tensor), 4,
             "Input tensor must have same dimensions as output tensor");

  elems_per_row = THLongTensor_size(index, dim);

  TH_TENSOR_DIM_APPLY3(real, tensor, real, src, int64_t, index, dim,
                       for (i = 0; i < elems_per_row; ++i)
                       {
                         idx = *(index_data + i*index_stride);
                         if (idx < TH_INDEX_BASE || idx >= tensor_size + TH_INDEX_BASE)
                         {
                           THFree(TH_TENSOR_DIM_APPLY_counter);
                           THError("Invalid index in scatterAdd");
                         }
                         tensor_data[(idx - TH_INDEX_BASE) * tensor_stride] += *(src_data + i*src_stride);
                       })
}

void THTensor_(scatterFill)(THTensor *tensor, int dim, THLongTensor *index, real val)
{
  int64_t elems_per_row, i, idx;

  THArgCheck(dim < THTensor_(nDimension)(tensor), 2, "Index dimension is out of bounds");
  THArgCheck(THLongTensor_nDimension(index) == THTensor_(nDimension)(tensor), 3,
             "Index tensor must have same dimensions as output tensor");

  elems_per_row = THLongTensor_size(index, dim);

  TH_TENSOR_DIM_APPLY2(real, tensor, int64_t, index, dim,
                       for (i = 0; i < elems_per_row; ++i)
                       {
                         idx = *(index_data + i*index_stride);
                         if (idx < TH_INDEX_BASE || idx >= tensor_size + TH_INDEX_BASE)
                         {
                           THFree(TH_TENSOR_DIM_APPLY_counter);
                           THError("Invalid index in scatter");
                         }
                         tensor_data[(idx - TH_INDEX_BASE) * tensor_stride] = val;
                       })
}

accreal THTensor_(dot)(THTensor *tensor, THTensor *src)
{
  accreal sum = 0;
  /* we use a trick here. careful with that. */
  TH_TENSOR_APPLY2(real, tensor, real, src,
                   int64_t sz = (tensor_size-tensor_i < src_size-src_i ? tensor_size-tensor_i : src_size-src_i);
                   sum += THBlas_(dot)(sz, src_data, src_stride, tensor_data, tensor_stride);
                   tensor_i += sz;
                   src_i += sz;
                   tensor_data += sz*tensor_stride;
                   src_data += sz*src_stride;
                   break;);
  return sum;
}


#undef th_isnan
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
#define th_isnan(val) \
(isnan(val))
#else
#define th_isnan(val) (0)
#endif

#undef th_isnan_break
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
#define th_isnan_break(val) \
if (isnan(val)) break;
#else
#define th_isnan_break(val)
#endif

real THTensor_(minall)(THTensor *tensor)
{
  real theMin;
  real value;

  THArgCheck(tensor->nDimension > 0, 1, "tensor must have one dimension");
  theMin = THTensor_(data)(tensor)[0];
  TH_TENSOR_APPLY(real, tensor,
                  value = *tensor_data;
                  /* This is not the same as value<theMin in the case of NaNs */
                  if(!(value >= theMin))
                  {
                    theMin = value;
                    th_isnan_break(value)
                  });
  return theMin;
}

real THTensor_(maxall)(THTensor *tensor)
{
  real theMax;
  real value;

  THArgCheck(tensor->nDimension > 0, 1, "tensor must have one dimension");
  theMax = THTensor_(data)(tensor)[0];
  TH_TENSOR_APPLY(real, tensor,
                  value = *tensor_data;
                  /* This is not the same as value>theMax in the case of NaNs */
                  if(!(value <= theMax))
                  {
                    theMax = value;
                    th_isnan_break(value)
                  });
  return theMax;
}

static void THTensor_(quickselectnoidx)(real *arr, int64_t k, int64_t elements, int64_t stride);

real THTensor_(medianall)(THTensor *tensor)
{
  THArgCheck(tensor->nDimension > 0, 1, "tensor must have one dimension");

  real theMedian;
  ptrdiff_t numel;
  int64_t k;
  THTensor *temp_;
  real *temp__data;

  numel = THTensor_(nElement)(tensor);
  k = (numel-1) >> 1;

  temp_ = THTensor_(newClone)(tensor);
  temp__data = THTensor_(data)(temp_);

  THTensor_(quickselectnoidx)(temp__data, k, numel, 1);

  theMedian = temp__data[k];

  THTensor_(free)(temp_);

  return theMedian;
}

/* sumall (Phase 3b): every thread keeps its own partial sum (8 independent
   accumulators within a run, so the additions can overlap), and partials are
   combined in thread order. Integer sums are exact as before (wrap-around
   addition does not depend on order). Floating-point sums change by rounding
   only: a different, and for long sums more accurate, summation order. For a
   given thread count the result is deterministic. */
#define TH_SUMALL_PAD 8   /* accreals per thread slot: keeps slots on separate cache lines */

static void THTensor_(sumAllLoop)(char **d, const int64_t *s, int64_t n, void *ctx)
{
  int tid = 0;
#ifdef _OPENMP
  tid = omp_get_thread_num();
#endif
  accreal *slot = (accreal *)ctx + TH_SUMALL_PAD * tid;
  if (TH_ITER_CONTIG(s[0])) {
    const real *x = (const real *)d[0];
    accreal a0 = 0, a1 = 0, a2 = 0, a3 = 0, a4 = 0, a5 = 0, a6 = 0, a7 = 0;
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
      a0 += x[i];     a1 += x[i + 1]; a2 += x[i + 2]; a3 += x[i + 3];
      a4 += x[i + 4]; a5 += x[i + 5]; a6 += x[i + 6]; a7 += x[i + 7];
    }
    for (; i < n; i++) a0 += x[i];
    *slot += ((a0 + a1) + (a2 + a3)) + ((a4 + a5) + (a6 + a7));
  } else {
    accreal a = 0;
    const char *p = d[0];
    for (int64_t i = 0; i < n; i++, p += s[0]) a += *(const real *)p;
    *slot += a;
  }
}

accreal THTensor_(sumall)(THTensor *tensor)
{
  if (THTensor_(nElement)(tensor) == 0) return 0;
  int maxThreads = 1;
#ifdef _OPENMP
  maxThreads = omp_get_max_threads();
#endif
  accreal *partials = THAlloc(sizeof(accreal) * TH_SUMALL_PAD * maxThreads);
  for (int t = 0; t < TH_SUMALL_PAD * maxThreads; t++) partials[t] = 0;
  THIterOperand op = THTensor_(iterOperand)(tensor);
  THIter it;
  THIter_build(&it, 1, &op);
  THIter_run(&it, THTensor_(sumAllLoop), partials, TH_ITER_GRAIN);
  accreal sum = 0;
  for (int t = 0; t < maxThreads; t++) sum += partials[TH_SUMALL_PAD * t];
  THFree(partials);
  return sum;
}

accreal THTensor_(prodall)(THTensor *tensor)
{
  accreal prod = 1;
  TH_TENSOR_APPLY(real, tensor, prod *= *tensor_data;);
  return prod;
}

void THTensor_(add)(THTensor *r_, THTensor *t, real value)
{
  THTensor_(resizeAs)(r_, t);
  if (THTensor_(isContiguous)(r_) && THTensor_(isContiguous)(t) && THTensor_(nElement)(r_) == THTensor_(nElement)(t)) {
    TH_TENSOR_APPLY2_CONTIG(real, r_, real, t, THVector_(adds)(r__data, t_data, value, r__len););
  } else {
    if (!THTensor_(iterate2)(r_, t, THTensor_(addLoop), &value))
      TH_TENSOR_APPLY2(real, r_, real, t, *r__data = *t_data + value;);
  }
}

void THTensor_(sub)(THTensor *r_, THTensor *t, real value)
{
  THTensor_(add)(r_, t, -value);
}

void THTensor_(mul)(THTensor *r_, THTensor *t, real value)
{
  THTensor_(resizeAs)(r_, t);
  if (THTensor_(isContiguous)(r_) && THTensor_(isContiguous)(t) && THTensor_(nElement)(r_) == THTensor_(nElement)(t)) {
    TH_TENSOR_APPLY2_CONTIG(real, r_, real, t, THVector_(muls)(r__data, t_data, value, r__len););
  } else {
    if (!THTensor_(iterate2)(r_, t, THTensor_(mulLoop), &value))
      TH_TENSOR_APPLY2(real, r_, real, t, *r__data = *t_data * value;);
  }
}

/* Unsigned counterpart of `real`, used by the (logical) shift operators and the
   wrapping negation below. `unsigned real` is not valid C when real is the
   int64_t typedef. */
#if defined(TH_REAL_IS_LONG)
# define TH_UNSIGNED_REAL uint64_t
#else
# define TH_UNSIGNED_REAL unsigned real
#endif

#if !defined(TH_REAL_IS_FLOAT) && !defined(TH_REAL_IS_DOUBLE) && !defined(TH_REAL_IS_HALF)
/* ---------------------------------------------------------------------------
 * Integer division helpers, shared by div/cdiv, fmod/cfmod (truncating) and
 * floordiv/cfloordiv, remainder/cremainder (flooring). In C, x / 0 and
 * MIN / -1 are undefined (both trap on x86 and kill the process), so:
 *   - a zero divisor raises a TH error (checked before any parallel work);
 *   - b == -1 is special-cased: quotient = wrapping negation, remainder = 0.
 * ------------------------------------------------------------------------- */
static inline real THTensor_(divTruncInt)(real a, real b)
{
#if !defined(TH_REAL_IS_BYTE)
  if (b == -1) return (real)(0 - (TH_UNSIGNED_REAL)a);
#endif
  return a / b;
}

static inline real THTensor_(modTruncInt)(real a, real b)
{
#if !defined(TH_REAL_IS_BYTE)
  if (b == -1) return 0;
#endif
  return a % b;
}

static inline real THTensor_(divFloorInt)(real a, real b)
{
#if !defined(TH_REAL_IS_BYTE)
  if (b == -1) return (real)(0 - (TH_UNSIGNED_REAL)a);
#endif
  real q = a / b;
  /* compare signs instead of multiplying, which could overflow */
  if ((a % b != 0) && ((a < 0) != (b < 0))) q = q - 1;
  return q;
}

static inline real THTensor_(modFloorInt)(real a, real b)
{
#if !defined(TH_REAL_IS_BYTE)
  if (b == -1) return 0;
#endif
  real m = a % b;
  if ((m != 0) && ((m < 0) != (b < 0))) m = m + b;
  return m;
}

static void THTensor_(checkNoZeroDivisor)(THTensor *src, int argNumber)
{
  int zero = 0;
  TH_TENSOR_APPLY(real, src, if (*src_data == 0) zero = 1;);
  THArgCheck(!zero, argNumber, "integer division by zero");
}
#endif

void THTensor_(div)(THTensor *r_, THTensor *t, real value)
{
#if !defined(TH_REAL_IS_FLOAT) && !defined(TH_REAL_IS_DOUBLE) && !defined(TH_REAL_IS_HALF)
  THArgCheck(value != 0, 3, "integer division by zero");
#if !defined(TH_REAL_IS_BYTE)
  if (value == -1) {  /* MIN / -1 traps on x86: negate (wrapping) instead */
    THTensor_(resizeAs)(r_, t);
    TH_TENSOR_APPLY2(real, r_, real, t, *r__data = THTensor_(divTruncInt)(*t_data, value););
    return;
  }
#endif
#endif
  THTensor_(resizeAs)(r_, t);
  if (THTensor_(isContiguous)(r_) && THTensor_(isContiguous)(t) && THTensor_(nElement)(r_) == THTensor_(nElement)(t)) {
    TH_TENSOR_APPLY2_CONTIG(real, r_, real, t, THVector_(divs)(r__data, t_data, value, r__len););
  } else {
    if (!THTensor_(iterate2)(r_, t, THTensor_(divLoop), &value))
      TH_TENSOR_APPLY2(real, r_, real, t, *r__data = *t_data / value;);
  }
}

#if !defined(TH_REAL_IS_FLOAT) && !defined(TH_REAL_IS_DOUBLE) && !defined(TH_REAL_IS_HALF)
/* Integer shift amounts must lie in [0, bits-1]: anything else is undefined
   behavior in C (on x86 the count is silently masked, elsewhere it differs).
   A shift by 0 is the identity. Checked before any (parallel) work starts. */
#define TH_SHIFT_BITS ((int64_t)(8 * sizeof(real)))

static void THTensor_(checkShiftAmount)(real value, int argNumber)
{
  int64_t s = (int64_t)value;
  THArgCheck(s >= 0 && s < TH_SHIFT_BITS, argNumber,
             "shift amount %" PRId64 " is out of range for %s: must be in [0, %" PRId64 "]",
             s, TH_CONCAT_STRING_3(torch., Real, Tensor), TH_SHIFT_BITS - 1);
}

static void THTensor_(checkShiftAmounts)(THTensor *src, int argNumber)
{
  int found = 0;
  int64_t bad = 0;
  TH_TENSOR_APPLY(real, src,
                  int64_t s = (int64_t)*src_data;
                  if (!found && (s < 0 || s >= TH_SHIFT_BITS)) { found = 1; bad = s; });
  THArgCheck(!found, argNumber,
             "shift amount %" PRId64 " is out of range for %s: must be in [0, %" PRId64 "]",
             bad, TH_CONCAT_STRING_3(torch., Real, Tensor), TH_SHIFT_BITS - 1);
}
#endif

void THTensor_(lshift)(THTensor *r_, THTensor *t, real value)
{
#if defined(TH_REAL_IS_FLOAT)
  return THTensor_(mul)(r_, t, powf(2, value));
#elif defined(TH_REAL_IS_DOUBLE)
  return THTensor_(mul)(r_, t, pow(2, value));
#elif defined(TH_REAL_IS_HALF)
  return THError("lshift is not supported for torch.HalfTensor");
#else
  THTensor_(checkShiftAmount)(value, 3);
  THTensor_(resizeAs)(r_, t);
  if (THTensor_(isContiguous)(r_) &&
      THTensor_(isContiguous)(t) &&
      THTensor_(nElement)(r_) == THTensor_(nElement)(t)) {
      real *tp = THTensor_(data)(t);
      real *rp = THTensor_(data)(r_);
      int64_t sz = THTensor_(nElement)(t);
      int64_t i;
      #pragma omp parallel for if(sz > TH_OMP_OVERHEAD_THRESHOLD * 100) private(i)
      for (i=0; i<sz; i++) {
#if defined(TH_REAL_IS_BYTE)
          rp[i] = ((real) tp[i]) << value;
#else
          rp[i] = ((TH_UNSIGNED_REAL) tp[i]) << value;
#endif
      }
  } else {
#if defined(TH_REAL_IS_BYTE)
      TH_TENSOR_APPLY2(real, r_, real, t, *r__data = (((real) *t_data) << value););
#else
      TH_TENSOR_APPLY2(real, r_, real, t, *r__data = (((TH_UNSIGNED_REAL) *t_data) << value););
#endif
  }
#endif
}

void THTensor_(rshift)(THTensor *r_, THTensor *t, real value)
{
#if defined(TH_REAL_IS_FLOAT)
  return THTensor_(div)(r_, t, powf(2, value));
#elif defined(TH_REAL_IS_DOUBLE)
  return THTensor_(div)(r_, t, pow(2, value));
#elif defined(TH_REAL_IS_HALF)
  return THError("rshift is not supported for torch.HalfTensor");
#else
  /* NOTE: on signed types this is a LOGICAL shift (zero-filling, via the
     unsigned type). For an arithmetic shift (floor toward -inf) use arshift. */
  THTensor_(checkShiftAmount)(value, 3);
  THTensor_(resizeAs)(r_, t);
  if (THTensor_(isContiguous)(r_) &&
      THTensor_(isContiguous)(t) &&
      THTensor_(nElement)(r_) == THTensor_(nElement)(t)) {
      real *tp = THTensor_(data)(t);
      real *rp = THTensor_(data)(r_);
      int64_t sz = THTensor_(nElement)(t);
      int64_t i;
      #pragma omp parallel for if(sz > TH_OMP_OVERHEAD_THRESHOLD * 100) private(i)
      for (i=0; i<sz; i++) {
#if defined(TH_REAL_IS_BYTE)
          rp[i] = ((real) tp[i]) >> value;
#else
          rp[i] = ((TH_UNSIGNED_REAL) tp[i]) >> value;
#endif
      }
  } else {
#if defined(TH_REAL_IS_BYTE)
      TH_TENSOR_APPLY2(real, r_, real, t, *r__data = (((real) *t_data) >> value););
#else
      TH_TENSOR_APPLY2(real, r_, real, t, *r__data = (((TH_UNSIGNED_REAL) *t_data) >> value););
#endif
  }
#endif
}

void THTensor_(fmod)(THTensor *r_, THTensor *t, real value)
{
#if !defined(TH_REAL_IS_FLOAT) && !defined(TH_REAL_IS_DOUBLE)
  THArgCheck(value != 0, 3, "integer division by zero");
#endif
  THTensor_(resizeAs)(r_, t);
  if (THTensor_(isContiguous)(r_) && THTensor_(isContiguous)(t) && THTensor_(nElement)(r_) == THTensor_(nElement)(t)) {

      real *tp = THTensor_(data)(t);
      real *rp = THTensor_(data)(r_);
      ptrdiff_t sz = THTensor_(nElement)(t);
      ptrdiff_t i;
      #pragma omp parallel for if(sz > TH_OMP_OVERHEAD_THRESHOLD) private(i)
      for (i=0; i<sz; i++) {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
          rp[i] = fmod(tp[i], value);
#else
          rp[i] = THTensor_(modTruncInt)(tp[i], value);
#endif
      }
  } else {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
      TH_TENSOR_APPLY2(real, r_, real, t, *r__data = fmod(*t_data, value););
#else
      TH_TENSOR_APPLY2(real, r_, real, t, *r__data = THTensor_(modTruncInt)(*t_data, value););
#endif
  }
}

void THTensor_(remainder)(THTensor *r_, THTensor *t, real value)
{
#if !defined(TH_REAL_IS_FLOAT) && !defined(TH_REAL_IS_DOUBLE)
  THArgCheck(value != 0, 3, "integer division by zero");
#endif
  THTensor_(resizeAs)(r_, t);
  if (THTensor_(isContiguous)(r_) && THTensor_(isContiguous)(t) && THTensor_(nElement)(r_) == THTensor_(nElement)(t)) {
      real *tp = THTensor_(data)(t);
      real *rp = THTensor_(data)(r_);
      ptrdiff_t sz = THTensor_(nElement)(t);
      ptrdiff_t i;
      #pragma omp parallel for if(sz > TH_OMP_OVERHEAD_THRESHOLD) private(i)
      for (i=0; i<sz; i++) {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
          rp[i] = (value == 0)? NAN : tp[i] - value * floor(tp[i] / value);
#else
          rp[i] = THTensor_(modFloorInt)(tp[i], value);
#endif
      }
  } else {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
      TH_TENSOR_APPLY2(real, r_, real, t, *r__data = (value == 0)? NAN : *t_data - value * floor(*t_data / value););
#else
      TH_TENSOR_APPLY2(real, r_, real, t, *r__data = THTensor_(modFloorInt)(*t_data, value););
#endif
  }
}

void THTensor_(bitand)(THTensor *r_, THTensor *t, real value)
{
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || defined(TH_REAL_IS_HALF)
  return THError("bitand is only supported for integer type tensors");
#else
  THTensor_(resizeAs)(r_, t);
  if (THTensor_(isContiguous)(r_) &&
      THTensor_(isContiguous)(t) &&
      THTensor_(nElement)(r_) == THTensor_(nElement)(t)) {
      real *tp = THTensor_(data)(t);
      real *rp = THTensor_(data)(r_);
      int64_t sz = THTensor_(nElement)(t);
      int64_t i;
      #pragma omp parallel for if(sz > TH_OMP_OVERHEAD_THRESHOLD * 100) private(i)
      for (i=0; i<sz; i++) {
          rp[i] = tp[i] & value;
      }
  } else {
      TH_TENSOR_APPLY2(real, r_, real, t, *r__data = *t_data & value;);
  }
#endif
}

void THTensor_(bitor)(THTensor *r_, THTensor *t, real value)
{
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || defined(TH_REAL_IS_HALF)
  return THError("bitor is only supported for integer type tensors");
#else
  THTensor_(resizeAs)(r_, t);
  if (THTensor_(isContiguous)(r_) &&
      THTensor_(isContiguous)(t) &&
      THTensor_(nElement)(r_) == THTensor_(nElement)(t)) {
      real *tp = THTensor_(data)(t);
      real *rp = THTensor_(data)(r_);
      int64_t sz = THTensor_(nElement)(t);
      int64_t i;
      #pragma omp parallel for if(sz > TH_OMP_OVERHEAD_THRESHOLD * 100) private(i)
      for (i=0; i<sz; i++) {
          rp[i] = tp[i] | value;
      }
  } else {
      TH_TENSOR_APPLY2(real, r_, real, t, *r__data = *t_data | value;);
  }
#endif
}

void THTensor_(bitxor)(THTensor *r_, THTensor *t, real value)
{
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || defined(TH_REAL_IS_HALF)
  return THError("bitxor is only supported for integer type tensors");
#else
  THTensor_(resizeAs)(r_, t);
  if (THTensor_(isContiguous)(r_) &&
      THTensor_(isContiguous)(t) &&
      THTensor_(nElement)(r_) == THTensor_(nElement)(t)) {
      real *tp = THTensor_(data)(t);
      real *rp = THTensor_(data)(r_);
      int64_t sz = THTensor_(nElement)(t);
      int64_t i;
      #pragma omp parallel for if(sz > TH_OMP_OVERHEAD_THRESHOLD * 100) private(i)
      for (i=0; i<sz; i++) {
          rp[i] = tp[i] ^ value;
      }
  } else {
      TH_TENSOR_APPLY2(real, r_, real, t, *r__data = *t_data ^ value;);
  }
#endif
}

void THTensor_(clamp)(THTensor *r_, THTensor *t, real min_value, real max_value)
{
  THTensor_(resizeAs)(r_, t);
  if (THTensor_(isContiguous)(r_) && THTensor_(isContiguous)(t) && THTensor_(nElement)(r_) == THTensor_(nElement)(t)) {
    real *tp = THTensor_(data)(t);
    real *rp = THTensor_(data)(r_);
    /* real t_val; */
    ptrdiff_t sz = THTensor_(nElement)(t);
    ptrdiff_t i;
    #pragma omp parallel for if(sz > TH_OMP_OVERHEAD_THRESHOLD) private(i)
    for (i=0; i<sz; i++)
      rp[i] = (tp[i] < min_value) ? min_value : (tp[i] > max_value ? max_value : tp[i]);
  } else {
    TH_TENSOR_APPLY2(real, r_, real, t, *r__data = (*t_data < min_value) ? min_value : (*t_data > max_value ? max_value : *t_data););
  }
}

void THTensor_(cadd)(THTensor *r_, THTensor *t, real value, THTensor *src)
{
  THTensor_(resizeAs)(r_, t);
  if (THTensor_(isContiguous)(r_) && THTensor_(isContiguous)(t) && THTensor_(isContiguous)(src) && THTensor_(nElement)(r_) == THTensor_(nElement)(src)) {
    if(r_ == t) {
      THBlas_(axpy)(THTensor_(nElement)(t), value, THTensor_(data)(src), 1, THTensor_(data)(r_), 1);
    } else {
      TH_TENSOR_APPLY3_CONTIG(real, r_, real, t, real, src, THVector_(cadd)(r__data, t_data, src_data, value, r__len););
    }
  } else {
    if (!THTensor_(iterate3)(r_, t, src, THTensor_(caddLoop), &value))
      TH_TENSOR_APPLY3(real, r_, real, t, real, src, *r__data = *t_data + value * *src_data;);
  }
}

void THTensor_(csub)(THTensor *r_, THTensor *t, real value,THTensor *src)
{
  THTensor_(cadd)(r_, t, -value, src);
}

void THTensor_(cmul)(THTensor *r_, THTensor *t, THTensor *src)
{
  THTensor_(resizeAs)(r_, t);
  if (THTensor_(isContiguous)(r_) && THTensor_(isContiguous)(t) && THTensor_(isContiguous)(src) && THTensor_(nElement)(r_) == THTensor_(nElement)(src)) {
    TH_TENSOR_APPLY3_CONTIG(real, r_, real, t, real, src, THVector_(cmul)(r__data, t_data, src_data, r__len););
  } else {
    if (!THTensor_(iterate3)(r_, t, src, THTensor_(cmulLoop), NULL))
      TH_TENSOR_APPLY3(real, r_, real, t, real, src, *r__data = *t_data * *src_data;);
  }
}

void THTensor_(cpow)(THTensor *r_, THTensor *t, THTensor *src)
{
  THTensor_(resizeAs)(r_, t);
  if (THTensor_(isContiguous)(r_) && THTensor_(isContiguous)(t) && THTensor_(isContiguous)(src) && THTensor_(nElement)(r_) == THTensor_(nElement)(src)) {
    real *tp = THTensor_(data)(t);
    real *sp = THTensor_(data)(src);
    real *rp = THTensor_(data)(r_);
    ptrdiff_t sz = THTensor_(nElement)(t);
    ptrdiff_t i;
    #pragma omp parallel for if(sz > TH_OMP_OVERHEAD_THRESHOLD) private(i)
    for (i=0; i<sz; i++)
      rp[i] = pow(tp[i], sp[i]);
  } else {
    TH_TENSOR_APPLY3(real, r_, real, t, real, src, *r__data = pow(*t_data, *src_data););
  }
}

void THTensor_(cdiv)(THTensor *r_, THTensor *t, THTensor *src)
{
#if !defined(TH_REAL_IS_FLOAT) && !defined(TH_REAL_IS_DOUBLE) && !defined(TH_REAL_IS_HALF)
  /* integer: reject zero divisors up front; MIN / -1 wraps (see divTruncInt) */
  THTensor_(checkNoZeroDivisor)(src, 3);
  THTensor_(resizeAs)(r_, t);
  TH_TENSOR_APPLY3(real, r_, real, t, real, src, *r__data = THTensor_(divTruncInt)(*t_data, *src_data););
  return;
#endif
  THTensor_(resizeAs)(r_, t);
  if (THTensor_(isContiguous)(r_) && THTensor_(isContiguous)(t) && THTensor_(isContiguous)(src) && THTensor_(nElement)(r_) == THTensor_(nElement)(src)) {
    TH_TENSOR_APPLY3_CONTIG(real, r_, real, t, real, src, THVector_(cdiv)(r__data, t_data, src_data, r__len););
  } else {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
    if (!THTensor_(iterate3)(r_, t, src, THTensor_(cdivLoop), NULL))
#endif
      TH_TENSOR_APPLY3(real, r_, real, t, real, src, *r__data = *t_data / *src_data;);
  }
}

void THTensor_(clshift)(THTensor *r_, THTensor *t, THTensor *src)
{
#if defined(TH_REAL_IS_HALF)
  return THError("clshift is not supported for torch.HalfTensor");
#endif
#if !defined(TH_REAL_IS_FLOAT) && !defined(TH_REAL_IS_DOUBLE) && !defined(TH_REAL_IS_HALF)
  THTensor_(checkShiftAmounts)(src, 3);
#endif
  THTensor_(resizeAs)(r_, t);
  if (THTensor_(isContiguous)(r_) &&
      THTensor_(isContiguous)(t) &&
      THTensor_(isContiguous)(src) &&
      THTensor_(nElement)(r_) == THTensor_(nElement)(src)) {
      real *tp = THTensor_(data)(t);
      real *sp = THTensor_(data)(src);
      real *rp = THTensor_(data)(r_);
      ptrdiff_t sz = THTensor_(nElement)(t);
      ptrdiff_t i;
      #pragma omp parallel for if(sz > TH_OMP_OVERHEAD_THRESHOLD) private(i)
    for (i=0; i<sz; i++) {
#if defined(TH_REAL_IS_FLOAT)
      rp[i] = tp[i] * powf(2, sp[i]);
#elif defined(TH_REAL_IS_DOUBLE)
      rp[i] = tp[i] * pow(2, sp[i]);
#elif defined(TH_REAL_IS_BYTE)
      rp[i] = ((real) tp[i]) << sp[i];
#else
      rp[i] = ((TH_UNSIGNED_REAL) tp[i]) << sp[i];
#endif
    }
  } else {
#if defined(TH_REAL_IS_FLOAT)
      TH_TENSOR_APPLY3(real, r_, real, t, real, src, *r__data = *t_data * powf(2, *src_data););
#elif defined(TH_REAL_IS_DOUBLE)
      TH_TENSOR_APPLY3(real, r_, real, t, real, src, *r__data = *t_data * pow(2, *src_data););
#elif defined(TH_REAL_IS_BYTE)
      TH_TENSOR_APPLY3(real, r_, real, t, real, src, *r__data = ((real)*t_data) << *src_data;);
#else
      TH_TENSOR_APPLY3(real, r_, real, t, real, src, *r__data = ((TH_UNSIGNED_REAL)*t_data) << *src_data;);
#endif
  }
}

void THTensor_(crshift)(THTensor *r_, THTensor *t, THTensor *src)
{
#if defined(TH_REAL_IS_HALF)
  return THError("crshift is not supported for torch.HalfTensor");
#endif
#if !defined(TH_REAL_IS_FLOAT) && !defined(TH_REAL_IS_DOUBLE) && !defined(TH_REAL_IS_HALF)
  THTensor_(checkShiftAmounts)(src, 3);
#endif
  THTensor_(resizeAs)(r_, t);
  if (THTensor_(isContiguous)(r_) &&
      THTensor_(isContiguous)(t) &&
      THTensor_(isContiguous)(src) &&
      THTensor_(nElement)(r_) == THTensor_(nElement)(src)) {
      real *tp = THTensor_(data)(t);
      real *sp = THTensor_(data)(src);
      real *rp = THTensor_(data)(r_);
      ptrdiff_t sz = THTensor_(nElement)(t);
      ptrdiff_t i;
      #pragma omp parallel for if(sz > TH_OMP_OVERHEAD_THRESHOLD) private(i)
    for (i=0; i<sz; i++) {
#if defined(TH_REAL_IS_FLOAT)
      rp[i] = tp[i] / powf(2, sp[i]);
#elif defined(TH_REAL_IS_DOUBLE)
      rp[i] = tp[i] / pow(2, sp[i]);
#elif defined(TH_REAL_IS_BYTE)
      rp[i] = ((real) tp[i]) >> sp[i];
#else
      rp[i] = ((TH_UNSIGNED_REAL) tp[i]) >> sp[i];
#endif
    }
  } else {
#if defined(TH_REAL_IS_FLOAT)
      TH_TENSOR_APPLY3(real, r_, real, t, real, src, *r__data = *t_data / powf(2, *src_data););
#elif defined(TH_REAL_IS_DOUBLE)
      TH_TENSOR_APPLY3(real, r_, real, t, real, src, *r__data = *t_data / pow(2, *src_data););
#elif defined(TH_REAL_IS_BYTE)
      TH_TENSOR_APPLY3(real, r_, real, t, real, src, *r__data = ((real)*t_data) >> *src_data;);
#else
      TH_TENSOR_APPLY3(real, r_, real, t, real, src, *r__data = ((TH_UNSIGNED_REAL)*t_data) >> *src_data;);
#endif
  }
}

/* ---------------------------------------------------------------------------
 * arshift / carshift: ARITHMETIC right shift (sign-filling), i.e. floor(x / 2^s)
 * rounding toward -inf. Integer tensors only. GCC defines >> on negative
 * signed integers as arithmetic, so this is exact over the whole range.
 * (rshift / crshift stay LOGICAL on signed types.)
 * ------------------------------------------------------------------------- */
void THTensor_(arshift)(THTensor *r_, THTensor *t, real value)
{
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || defined(TH_REAL_IS_HALF)
  (void)r_; (void)t; (void)value;
  THError("arshift is only defined for integer tensors, not %s",
          TH_CONCAT_STRING_3(torch., Real, Tensor));
#else
  THTensor_(checkShiftAmount)(value, 3);
  const int s = (int)value;
  THTensor_(resizeAs)(r_, t);
  TH_TENSOR_APPLY2(real, r_, real, t, *r__data = (real)(*t_data >> s););
#endif
}

void THTensor_(carshift)(THTensor *r_, THTensor *t, THTensor *src)
{
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || defined(TH_REAL_IS_HALF)
  (void)r_; (void)t; (void)src;
  THError("carshift is only defined for integer tensors, not %s",
          TH_CONCAT_STRING_3(torch., Real, Tensor));
#else
  THTensor_(checkShiftAmounts)(src, 3);
  THTensor_(resizeAs)(r_, t);
  TH_TENSOR_APPLY3(real, r_, real, t, real, src, *r__data = (real)(*t_data >> (int)*src_data););
#endif
}

/* ---------------------------------------------------------------------------
 * floordiv / cfloordiv: division rounding toward -inf. Pairs with remainder /
 * cremainder (floor-mod):  a == floordiv(a, b) * b + remainder(a, b).
 * Integer division by zero raises an error. For floating point: floor(a / b).
 * ------------------------------------------------------------------------- */
void THTensor_(floordiv)(THTensor *r_, THTensor *t, real value)
{
#if defined(TH_REAL_IS_HALF)
  (void)r_; (void)t; (void)value;
  THError("floordiv is not supported for torch.HalfTensor");
#elif defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
  THTensor_(resizeAs)(r_, t);
  TH_TENSOR_APPLY2(real, r_, real, t, *r__data = (real)floor((double)*t_data / (double)value););
#else
  THArgCheck(value != 0, 3, "integer division by zero");
  THTensor_(resizeAs)(r_, t);
  TH_TENSOR_APPLY2(real, r_, real, t, *r__data = THTensor_(divFloorInt)(*t_data, value););
#endif
}

void THTensor_(cfloordiv)(THTensor *r_, THTensor *t, THTensor *src)
{
#if defined(TH_REAL_IS_HALF)
  (void)r_; (void)t; (void)src;
  THError("cfloordiv is not supported for torch.HalfTensor");
#elif defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
  THTensor_(resizeAs)(r_, t);
  TH_TENSOR_APPLY3(real, r_, real, t, real, src,
                   *r__data = (real)floor((double)*t_data / (double)*src_data););
#else
  THTensor_(checkNoZeroDivisor)(src, 3);
  THTensor_(resizeAs)(r_, t);
  TH_TENSOR_APPLY3(real, r_, real, t, real, src,
                   *r__data = THTensor_(divFloorInt)(*t_data, *src_data););
#endif
}

#if !defined(TH_REAL_IS_FLOAT) && !defined(TH_REAL_IS_DOUBLE) && !defined(TH_REAL_IS_HALF)
#undef TH_SHIFT_BITS
#endif
#undef TH_UNSIGNED_REAL

void THTensor_(cfmod)(THTensor *r_, THTensor *t, THTensor *src)
{
#if !defined(TH_REAL_IS_FLOAT) && !defined(TH_REAL_IS_DOUBLE)
  THTensor_(checkNoZeroDivisor)(src, 3);
#endif
  THTensor_(resizeAs)(r_, t);
  if (THTensor_(isContiguous)(r_) && THTensor_(isContiguous)(t) && THTensor_(isContiguous)(src) && THTensor_(nElement)(r_) == THTensor_(nElement)(src)) {
      real *tp = THTensor_(data)(t);
      real *sp = THTensor_(data)(src);
      real *rp = THTensor_(data)(r_);
      ptrdiff_t sz = THTensor_(nElement)(t);
      ptrdiff_t i;
      #pragma omp parallel for if(sz > TH_OMP_OVERHEAD_THRESHOLD) private(i)
      for (i=0; i<sz; i++) {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
          rp[i] = fmod(tp[i], sp[i]);
#else
          rp[i] = THTensor_(modTruncInt)(tp[i], sp[i]);
#endif
      }
  } else {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
      TH_TENSOR_APPLY3(real, r_, real, t, real, src, *r__data = fmod(*t_data, *src_data););
#else
      TH_TENSOR_APPLY3(real, r_, real, t, real, src, *r__data = THTensor_(modTruncInt)(*t_data, *src_data););
#endif

  }
}

void THTensor_(cremainder)(THTensor *r_, THTensor *t, THTensor *src)
{
#if !defined(TH_REAL_IS_FLOAT) && !defined(TH_REAL_IS_DOUBLE)
  THTensor_(checkNoZeroDivisor)(src, 3);
#endif
  THTensor_(resizeAs)(r_, t);
  if (THTensor_(isContiguous)(r_) && THTensor_(isContiguous)(t) && THTensor_(isContiguous)(src) && THTensor_(nElement)(r_) == THTensor_(nElement)(src)) {
      real *tp = THTensor_(data)(t);
      real *sp = THTensor_(data)(src);
      real *rp = THTensor_(data)(r_);
      ptrdiff_t sz = THTensor_(nElement)(t);
      ptrdiff_t i;
      #pragma omp parallel for if(sz > TH_OMP_OVERHEAD_THRESHOLD) private(i)
      for (i=0; i<sz; i++) {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
          rp[i] = (sp[i] == 0)? NAN : tp[i] - sp[i] * floor(tp[i] / sp[i]);
#else
          rp[i] = THTensor_(modFloorInt)(tp[i], sp[i]);
#endif
      }
  } else {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
      TH_TENSOR_APPLY3(real, r_, real, t, real, src, *r__data = (*src_data == 0)? NAN : *t_data - *src_data * floor(*t_data / *src_data););
#else
      TH_TENSOR_APPLY3(real, r_, real, t, real, src, *r__data = THTensor_(modFloorInt)(*t_data, *src_data););
#endif

  }
}

void THTensor_(cbitand)(THTensor *r_, THTensor *t, THTensor *src)
{
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || defined(TH_REAL_IS_HALF)
  return THError("cbitand is only supported for integer type tensors");
#else
  THTensor_(resizeAs)(r_, t);
  if (THTensor_(isContiguous)(r_) &&
      THTensor_(isContiguous)(t) &&
      THTensor_(isContiguous)(src) &&
      THTensor_(nElement)(r_) == THTensor_(nElement)(src)) {
      real *tp = THTensor_(data)(t);
      real *sp = THTensor_(data)(src);
      real *rp = THTensor_(data)(r_);
      ptrdiff_t sz = THTensor_(nElement)(t);
      ptrdiff_t i;
      #pragma omp parallel for if(sz > TH_OMP_OVERHEAD_THRESHOLD) private(i)
    for (i=0; i<sz; i++) {
      rp[i] = tp[i] & sp[i];
    }
  } else {
      TH_TENSOR_APPLY3(real, r_, real, t, real, src, *r__data = *t_data & *src_data;);
  }
#endif
}

void THTensor_(cbitor)(THTensor *r_, THTensor *t, THTensor *src)
{
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || defined(TH_REAL_IS_HALF)
  return THError("cbitor is only supported for integer type tensors");
#else
  THTensor_(resizeAs)(r_, t);
  if (THTensor_(isContiguous)(r_) &&
      THTensor_(isContiguous)(t) &&
      THTensor_(isContiguous)(src) &&
      THTensor_(nElement)(r_) == THTensor_(nElement)(src)) {
      real *tp = THTensor_(data)(t);
      real *sp = THTensor_(data)(src);
      real *rp = THTensor_(data)(r_);
      ptrdiff_t sz = THTensor_(nElement)(t);
      ptrdiff_t i;
      #pragma omp parallel for if(sz > TH_OMP_OVERHEAD_THRESHOLD) private(i)
    for (i=0; i<sz; i++) {
      rp[i] = tp[i] | sp[i];
    }
  } else {
      TH_TENSOR_APPLY3(real, r_, real, t, real, src, *r__data = *t_data | *src_data;);
  }
#endif
}

void THTensor_(cbitxor)(THTensor *r_, THTensor *t, THTensor *src)
{
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || defined(TH_REAL_IS_HALF)
  return THError("cbitxor is only supported for integer type tensors");
#else
  THTensor_(resizeAs)(r_, t);
  if (THTensor_(isContiguous)(r_) &&
      THTensor_(isContiguous)(t) &&
      THTensor_(isContiguous)(src) &&
      THTensor_(nElement)(r_) == THTensor_(nElement)(src)) {
      real *tp = THTensor_(data)(t);
      real *sp = THTensor_(data)(src);
      real *rp = THTensor_(data)(r_);
      ptrdiff_t sz = THTensor_(nElement)(t);
      ptrdiff_t i;
      #pragma omp parallel for if(sz > TH_OMP_OVERHEAD_THRESHOLD) private(i)
    for (i=0; i<sz; i++) {
      rp[i] = tp[i] ^ sp[i];
    }
  } else {
      TH_TENSOR_APPLY3(real, r_, real, t, real, src, *r__data = *t_data ^ *src_data;);
  }
#endif
}

void THTensor_(tpow)(THTensor *r_, real value, THTensor *t)
{
  THTensor_(resizeAs)(r_, t);
  if (THTensor_(isContiguous)(r_) && THTensor_(isContiguous)(t) && THTensor_(nElement)(r_) == THTensor_(nElement)(t)) {
    real *tp = THTensor_(data)(t);
    real *rp = THTensor_(data)(r_);
    ptrdiff_t sz = THTensor_(nElement)(t);
    ptrdiff_t i;
    #pragma omp parallel for if(sz > TH_OMP_OVERHEAD_THRESHOLD) private(i)
    for (i=0; i<sz; i++)
      rp[i] = pow(value, tp[i]);
  } else {
    TH_TENSOR_APPLY2(real, r_, real, t, *r__data = pow(value, *t_data););
  }
}

void THTensor_(addcmul)(THTensor *r_, THTensor *t, real value, THTensor *src1, THTensor *src2)
{
  if(r_ != t)
  {
    THTensor_(resizeAs)(r_, t);
    THTensor_(copy)(r_, t);
  }

  TH_TENSOR_APPLY3(real, r_, real, src1, real, src2, *r__data += value * *src1_data * *src2_data;);
}


void THTensor_(addcdiv)(THTensor *r_, THTensor *t, real value, THTensor *src1, THTensor *src2)
{
  if(r_ != t)
  {
    THTensor_(resizeAs)(r_, t);
    THTensor_(copy)(r_, t);
  }

  TH_TENSOR_APPLY3(real, r_, real, src1, real, src2, *r__data += value * *src1_data / *src2_data;);
}

void THTensor_(addmv)(THTensor *r_, real beta, THTensor *t, real alpha, THTensor *mat, THTensor *vec)
{
  if( (mat->nDimension != 2) || (vec->nDimension != 1) )
    THError("matrix and vector expected, got %dD, %dD",
      mat->nDimension, vec->nDimension);

  if( mat->size[1] != vec->size[0] ) {
    THDescBuff bm = THTensor_(sizeDesc)(mat);
    THDescBuff bv = THTensor_(sizeDesc)(vec);
    THError("size mismatch, %s, %s", bm.str, bv.str);
  }

  if(t->nDimension != 1)
    THError("vector expected, got t: %dD", t->nDimension);

  if(t->size[0] != mat->size[0]) {
    THDescBuff bt = THTensor_(sizeDesc)(t);
    THDescBuff bm = THTensor_(sizeDesc)(mat);
    THError("size mismatch, t: %s, mat: %s", bt.str, bm.str);
  }

  if(r_ != t)
  {
    THTensor_(resizeAs)(r_, t);
    THTensor_(copy)(r_, t);
  }

  if(mat->stride[0] == 1)
  {
    THBlas_(gemv)('n', mat->size[0], mat->size[1],
                  alpha, THTensor_(data)(mat), mat->stride[1],
                  THTensor_(data)(vec), vec->stride[0],
                  beta, THTensor_(data)(r_), r_->stride[0]);
  }
  else if(mat->stride[1] == 1)
  {
    THBlas_(gemv)('t',  mat->size[1], mat->size[0],
                  alpha, THTensor_(data)(mat), mat->stride[0],
                  THTensor_(data)(vec), vec->stride[0],
                  beta, THTensor_(data)(r_), r_->stride[0]);
  }
  else
  {
    THTensor *cmat = THTensor_(newContiguous)(mat);

    THBlas_(gemv)('t',  mat->size[1], mat->size[0],
                  alpha, THTensor_(data)(cmat), cmat->stride[0],
                  THTensor_(data)(vec), vec->stride[0],
                  beta, THTensor_(data)(r_), r_->stride[0]);

    THTensor_(free)(cmat);
  }
}

void THTensor_(match)(THTensor *r_, THTensor *m1, THTensor *m2, real gain)
{
  int64_t N1 = m1->size[0];
  int64_t N2 = m2->size[0];
  int64_t dim;
  real *m1_p;
  real *m2_p;
  real *r_p;
  int64_t i;

  THTensor_(resize2d)(r_, N1, N2);

  m1 = THTensor_(newContiguous)(m1);
  m2 = THTensor_(newContiguous)(m2);

  THTensor_(resize2d)(m1, N1, THTensor_(nElement)(m1) / N1);
  THTensor_(resize2d)(m2, N2, THTensor_(nElement)(m2) / N2);

  dim = m1->size[1];
  THArgCheck(m1->size[1] == m2->size[1], 3, "m1 and m2 must have the same inner vector dim");

  m1_p = THTensor_(data)(m1);
  m2_p = THTensor_(data)(m2);
  r_p = THTensor_(data)(r_);

#pragma omp parallel for private(i)
  for (i=0; i<N1; i++) {
    int64_t j,k;
    for (j=0; j<N2; j++) {
      real sum = 0;
      for (k=0; k<dim; k++) {
        real term = m1_p[ i*dim + k ] - m2_p[ j*dim + k ];
        sum += term*term;
      }
      r_p[ i*N2 + j ] = gain * sum;
    }
  }

  THTensor_(free)(m1);
  THTensor_(free)(m2);
}

void THTensor_(addmm)(THTensor *r_, real beta, THTensor *t, real alpha, THTensor *m1, THTensor *m2)
{
  char transpose_r, transpose_m1, transpose_m2;
  THTensor *r__, *m1_, *m2_;

  if( (m1->nDimension != 2) || (m2->nDimension != 2))
    THError("matrices expected, got %dD, %dD tensors", m1->nDimension, m2->nDimension);

  if(m1->size[1] != m2->size[0]) {
    THDescBuff bm1 = THTensor_(sizeDesc)(m1);
    THDescBuff bm2 = THTensor_(sizeDesc)(m2);
    THError("size mismatch, m1: %s, m2: %s", bm1.str, bm2.str);
  }

  if( t->nDimension != 2 )
    THError("matrix expected, got %dD tensor for t", t->nDimension);

  if( (t->size[0] != m1->size[0]) || (t->size[1] != m2->size[1]) ) {
    THDescBuff bt  = THTensor_(sizeDesc)(t);
    THDescBuff bm1 = THTensor_(sizeDesc)(m1);
    THDescBuff bm2 = THTensor_(sizeDesc)(m2);
    THError("size mismatch, t: %s, m1: %s, m2: %s", bt.str, bm1.str, bm2.str);
  }

  if(t != r_)
  {
    THTensor_(resizeAs)(r_, t);
    if (beta != 0.0) {
      THTensor_(copy)(r_, t);
    }
  }

  /* r_ */
  if(r_->stride[0] == 1 &&
     r_->stride[1] != 0)
  {
    transpose_r = 'n';
    r__ = r_;
  }
  else if(r_->stride[1] == 1 &&
          r_->stride[0] != 0)
  {
    THTensor *swap = m2;
    m2 = m1;
    m1 = swap;
    transpose_r = 't';
    r__ = r_;
  }
  else
  {
    transpose_r = 'n';

    THTensor *transp_r_ = THTensor_(newTranspose)(r_, 0, 1);
    r__ = THTensor_(newClone)(transp_r_);
    THTensor_(free)(transp_r_);
    THTensor_(transpose)(r__, NULL, 0, 1);
  }

  /* m1 */
  if(m1->stride[(transpose_r == 'n' ? 0 : 1)] == 1 &&
     m1->stride[(transpose_r == 'n' ? 1 : 0)] != 0)
  {
    transpose_m1 = 'n';
    m1_ = m1;
  }
  else if(m1->stride[(transpose_r == 'n' ? 1 : 0)] == 1 &&
          m1->stride[(transpose_r == 'n' ? 0 : 1)] != 0)
  {
    transpose_m1 = 't';
    m1_ = m1;
  }
  else
  {
    transpose_m1 = (transpose_r == 'n' ? 't' : 'n');
    m1_ = THTensor_(newContiguous)(m1);
  }

  /* m2 */
  if(m2->stride[(transpose_r == 'n' ? 0 : 1)] == 1 &&
     m2->stride[(transpose_r == 'n' ? 1 : 0)] != 0)
  {
    transpose_m2 = 'n';
    m2_ = m2;
  }
  else if(m2->stride[(transpose_r == 'n' ? 1 : 0)] == 1 &&
          m2->stride[(transpose_r == 'n' ? 0 : 1)] != 0)
  {
    transpose_m2 = 't';
    m2_ = m2;
  }
  else
  {
    transpose_m2 = (transpose_r == 'n' ? 't' : 'n');
    m2_ = THTensor_(newContiguous)(m2);
  }

#pragma omp critical(blasgemm)
  /* do the operation */
  THBlas_(gemm)(transpose_m1,
                transpose_m2,
                r__->size[(transpose_r == 'n' ? 0 : 1)],
                r__->size[(transpose_r == 'n' ? 1 : 0)],
                m1_->size[(transpose_r == 'n' ? 1 : 0)],
                alpha,
                THTensor_(data)(m1_),
                (transpose_m1 == 'n' ? m1_->stride[(transpose_r == 'n' ? 1 : 0)] : m1_->stride[(transpose_r == 'n' ? 0 : 1)]),
                THTensor_(data)(m2_),
                (transpose_m2 == 'n' ? m2_->stride[(transpose_r == 'n' ? 1 : 0)] : m2_->stride[(transpose_r == 'n' ? 0 : 1)]),
                beta,
                THTensor_(data)(r__),
                r__->stride[(transpose_r == 'n' ? 1 : 0)]);

  /* free intermediate variables */
  if(m1_ != m1)
    THTensor_(free)(m1_);

  if(m2_ != m2)
    THTensor_(free)(m2_);

  if(r__ != r_)
    THTensor_(freeCopyTo)(r__, r_);
}

void THTensor_(addr)(THTensor *r_, real beta, THTensor *t, real alpha, THTensor *vec1, THTensor *vec2)
{
  if( (vec1->nDimension != 1) || (vec2->nDimension != 1) )
    THError("vector and vector expected, got %dD, %dD tensors",
        vec1->nDimension, vec2->nDimension);

  if(t->nDimension != 2)
    THError("expected matrix, got %dD tensor for t", t->nDimension);

  if( (t->size[0] != vec1->size[0]) || (t->size[1] != vec2->size[0]) ) {
    THDescBuff bt  = THTensor_(sizeDesc)(t);
    THDescBuff bv1 = THTensor_(sizeDesc)(vec1);
    THDescBuff bv2 = THTensor_(sizeDesc)(vec2);
    THError("size mismatch, t: %s, vec1: %s, vec2: %s", bt.str, bv1.str, bv2.str);
  }

  if(r_ != t)
  {
    THTensor_(resizeAs)(r_, t);
    THTensor_(copy)(r_, t);
  }

  if(beta == 0) {
    THTensor_(zero)(r_);
  }
  else if(beta != 1)
    THTensor_(mul)(r_, r_, beta);

  if(r_->stride[0] == 1)
  {
    THBlas_(ger)(vec1->size[0], vec2->size[0],
                 alpha, THTensor_(data)(vec1), vec1->stride[0],
                 THTensor_(data)(vec2), vec2->stride[0],
                 THTensor_(data)(r_), r_->stride[1]);
  }
  else if(r_->stride[1] == 1)
  {
    THBlas_(ger)(vec2->size[0], vec1->size[0],
                 alpha, THTensor_(data)(vec2), vec2->stride[0],
                 THTensor_(data)(vec1), vec1->stride[0],
                 THTensor_(data)(r_), r_->stride[0]);
  }
  else
  {
    THTensor *cr = THTensor_(newClone)(r_);

    THBlas_(ger)(vec2->size[0], vec1->size[0],
                 alpha, THTensor_(data)(vec2), vec2->stride[0],
                 THTensor_(data)(vec1), vec1->stride[0],
                 THTensor_(data)(cr), cr->stride[0]);

    THTensor_(freeCopyTo)(cr, r_);
  }
}

void THTensor_(addbmm)(THTensor *result, real beta, THTensor *t, real alpha, THTensor *batch1, THTensor *batch2)
{
  int64_t batch;

  THArgCheck(THTensor_(nDimension)(batch1) == 3, 1, "expected 3D tensor");
  THArgCheck(THTensor_(nDimension)(batch2) == 3, 2, "expected 3D tensor");
  THArgCheck(THTensor_(size)(batch1, 0) == THTensor_(size)(batch2, 0), 2,
             "equal number of batches expected, got %" PRId64 ", %" PRId64,
             THTensor_(size)(batch1, 0), THTensor_(size)(batch2, 0));
  THArgCheck(THTensor_(size)(batch1, 2) == THTensor_(size)(batch2, 1), 2,
             "wrong matrix size, batch1: %" PRId64 "x%" PRId64 ", batch2: %" PRId64 "x%" PRId64,
             THTensor_(size)(batch1, 1), THTensor_(size)(batch1,2),
             THTensor_(size)(batch2, 1), THTensor_(size)(batch2,2));

  int64_t dim1 = THTensor_(size)(batch1, 1);
  int64_t dim2 = THTensor_(size)(batch2, 2);
  THArgCheck(THTensor_(size)(t, 0) == dim1, 1, "output tensor of incorrect size");
  THArgCheck(THTensor_(size)(t, 1) == dim2, 1, "output tensor of incorrect size");

  if (t != result) {
    THTensor_(resizeAs)(result, t);
    if (beta != 0.0) {
      THTensor_(copy)(result, t);
    }
  }

  THTensor *matrix1 = THTensor_(new)();
  THTensor *matrix2 = THTensor_(new)();

  for (batch = 0; batch < THTensor_(size)(batch1, 0); ++batch) {
    THTensor_(select)(matrix1, batch1, 0, batch);
    THTensor_(select)(matrix2, batch2, 0, batch);

    THTensor_(addmm)(result, beta, result, alpha, matrix1, matrix2);
    beta = 1; // accumulate output once
  }

  THTensor_(free)(matrix1);
  THTensor_(free)(matrix2);
}

void THTensor_(baddbmm)(THTensor *result, real beta, THTensor *t, real alpha, THTensor *batch1, THTensor *batch2)
{
  int64_t batch;

  THArgCheck(THTensor_(nDimension)(batch1) == 3, 1, "expected 3D tensor, got %dD", THTensor_(nDimension)(batch1));
  THArgCheck(THTensor_(nDimension)(batch2) == 3, 2, "expected 3D tensor, got %dD", THTensor_(nDimension)(batch2));
  THArgCheck(THTensor_(size)(batch1, 0) == THTensor_(size)(batch2, 0), 2,
             "equal number of batches expected, got %" PRId64 ", %" PRId64,
             THTensor_(size)(batch1, 0), THTensor_(size)(batch2, 0));
  THArgCheck(THTensor_(size)(batch1, 2) == THTensor_(size)(batch2, 1), 2,
             "wrong matrix size, batch1: %" PRId64 "x%" PRId64 ", batch2: %" PRId64 "x%" PRId64,
             THTensor_(size)(batch1, 1), THTensor_(size)(batch1, 2),
             THTensor_(size)(batch2, 1), THTensor_(size)(batch2, 2));

  int64_t bs = THTensor_(size)(batch1, 0);
  int64_t dim1 = THTensor_(size)(batch1, 1);
  int64_t dim2 = THTensor_(size)(batch2, 2);
  THArgCheck(THTensor_(size)(t, 0) == bs, 1,   "output tensor of incorrect size");
  THArgCheck(THTensor_(size)(t, 1) == dim1, 1, "output tensor of incorrect size");
  THArgCheck(THTensor_(size)(t, 2) == dim2, 1, "output tensor of incorrect size");

  if (t != result) {
    THTensor_(resizeAs)(result, t);
    if (beta != 0.0) {
      THTensor_(copy)(result, t);
    }
  }

  THTensor *matrix1 = THTensor_(new)();
  THTensor *matrix2 = THTensor_(new)();
  THTensor *result_matrix = THTensor_(new)();

  for (batch = 0; batch < THTensor_(size)(batch1, 0); ++batch) {
    THTensor_(select)(matrix1, batch1, 0, batch);
    THTensor_(select)(matrix2, batch2, 0, batch);
    THTensor_(select)(result_matrix, result, 0, batch);

    THTensor_(addmm)(result_matrix, beta, result_matrix, alpha, matrix1, matrix2);
  }

  THTensor_(free)(matrix1);
  THTensor_(free)(matrix2);
  THTensor_(free)(result_matrix);
}

ptrdiff_t THTensor_(numel)(THTensor *t)
{
  return THTensor_(nElement)(t);
}

/* ---------------------------------------------------------------------------
 * Reductions along one dimension through THIter (Phase 2b).
 *
 * The iterator runs over the output (the input's shape with size 1 at the
 * reduced dimension) paired with the first element of each input slice, so
 * work is split across threads by output element. Each output element is
 * computed with exactly the same arithmetic, in the same order, as the
 * legacy code it replaces, so results are bit-identical.
 * ------------------------------------------------------------------------- */
typedef struct {
  int64_t n;        /* size of the reduced dimension */
  int64_t stride;   /* input stride along it, in bytes */
} THTensor_(ReduceCtx);

/* legacy layout 1 (reduced dim has stride 1): one accreal sum per slice */
static void THTensor_(sumSliceLoop)(char **d, const int64_t *s, int64_t n_, void *ctx_)
{
  const THTensor_(ReduceCtx) *c = ctx_;
  char *rp = d[0], *xp = d[1];
  for (int64_t j = 0; j < n_; j++, rp += s[0], xp += s[1]) {
    accreal sum = 0;
    for (int64_t i = 0; i < c->n; i++)
      sum += *(const real *)(xp + i * c->stride);
    *(real *)rp = (real)sum;
  }
}

/* legacy layout 2 (otherwise): out = ((0 + x_0) + x_1) + ... in `real`,
   computed row by row over a run of outputs (memory-friendly, same order) */
static void THTensor_(sumRowsLoop)(char **d, const int64_t *s, int64_t n_, void *ctx_)
{
  const THTensor_(ReduceCtx) *c = ctx_;
  char *rp = d[0];
  if (TH_ITER_CONTIG(s[0]) && TH_ITER_CONTIG(s[1])) {   /* vectorizable, same order */
    real *r = (real *)rp;
    for (int64_t j = 0; j < n_; j++) r[j] = 0;
    for (int64_t i = 0; i < c->n; i++) {
      const real *x = (const real *)(d[1] + i * c->stride);
      for (int64_t j = 0; j < n_; j++) r[j] = r[j] + x[j];
    }
    return;
  }
  for (int64_t j = 0; j < n_; j++) *(real *)(rp + j * s[0]) = 0;
  for (int64_t i = 0; i < c->n; i++) {
    const char *xp = d[1] + i * c->stride;
    for (int64_t j = 0; j < n_; j++)
      *(real *)(rp + j * s[0]) = *(real *)(rp + j * s[0]) + *(const real *)(xp + j * s[1]);
  }
}

/* Run a slice kernel over nout outputs (keepdim shape) and t; the input is
   the last operand. Returns 0 (caller falls back to the legacy code) if an
   output overlaps t or t has too many dimensions. */
static int THTensor_(reduceDimN)(int nout, const THIterOperand *outs, THTensor *t,
                                 int dimension, THIterLoop loop)
{
  if (t->nDimension > TH_ITER_MAX_DIMS) return 0;
  THIterOperand ops[TH_ITER_MAX_OPERANDS];
  THIterOperand in = THTensor_(iterOperand)(t);
  for (int k = 0; k < nout; k++) {
    if (THIter_overlaps(&outs[k], &in)) return 0;
    ops[k] = outs[k];
  }
  int64_t sliceSize[TH_ITER_MAX_DIMS];
  for (int d = 0; d < t->nDimension; d++) sliceSize[d] = t->size[d];
  sliceSize[dimension] = 1;   /* each output element sees the first element of its slice */
  in.size = sliceSize;
  ops[nout] = in;
  THTensor_(ReduceCtx) ctx = {t->size[dimension], t->stride[dimension] * (int64_t)sizeof(real)};
  THIter it;
  THIter_build(&it, nout + 1, ops);
  int64_t grain = TH_ITER_GRAIN / (ctx.n > 0 ? ctx.n : 1);
  THIter_run(&it, loop, &ctx, grain > 0 ? grain : 1);
  return 1;
}

static int THTensor_(reduceDim)(THTensor *r_, THTensor *t, int dimension, THIterLoop loop)
{
  THIterOperand out = THTensor_(iterOperand)(r_);
  return THTensor_(reduceDimN)(1, &out, t, dimension, loop);
}

/* prod: same two legacy layouts as sum, with * and initial value 1 */
static void THTensor_(prodSliceLoop)(char **d, const int64_t *s, int64_t n_, void *ctx_)
{
  const THTensor_(ReduceCtx) *c = ctx_;
  char *rp = d[0], *xp = d[1];
  for (int64_t j = 0; j < n_; j++, rp += s[0], xp += s[1]) {
    accreal prod = 1;
    for (int64_t i = 0; i < c->n; i++)
      prod *= *(const real *)(xp + i * c->stride);
    *(real *)rp = (real)prod;
  }
}

static void THTensor_(prodRowsLoop)(char **d, const int64_t *s, int64_t n_, void *ctx_)
{
  const THTensor_(ReduceCtx) *c = ctx_;
  char *rp = d[0];
  if (TH_ITER_CONTIG(s[0]) && TH_ITER_CONTIG(s[1])) {   /* vectorizable, same order */
    real *r = (real *)rp;
    for (int64_t j = 0; j < n_; j++) r[j] = 1;
    for (int64_t i = 0; i < c->n; i++) {
      const real *x = (const real *)(d[1] + i * c->stride);
      for (int64_t j = 0; j < n_; j++) r[j] = r[j] * x[j];
    }
    return;
  }
  for (int64_t j = 0; j < n_; j++) *(real *)(rp + j * s[0]) = 1;
  for (int64_t i = 0; i < c->n; i++) {
    const char *xp = d[1] + i * c->stride;
    for (int64_t j = 0; j < n_; j++)
      *(real *)(rp + j * s[0]) = *(real *)(rp + j * s[0]) * *(const real *)(xp + j * s[1]);
  }
}

/* max / min with indices. Both legacy layouts keep the FIRST extreme in index
   order (ties keep the lowest index) and stop at the first NaN; the slice
   kernel scans each slice like legacy layout 1, the rows kernel walks rows
   like legacy layout 2 (start from row 0, update only while not NaN). */
#define TH_MAXMIN_KERNELS(NAME, KEEP)                                          \
static void THTensor_(NAME##SliceLoop)(char **d, const int64_t *s, int64_t n_, void *ctx_) \
{                                                                             \
  const THTensor_(ReduceCtx) *c = ctx_;                                       \
  char *vp = d[0], *ip = d[1], *xp = d[2];                                    \
  for (int64_t j = 0; j < n_; j++, vp += s[0], ip += s[1], xp += s[2]) {      \
    real best = *(const real *)xp;                                            \
    int64_t bi = 0;                                                           \
    for (int64_t i = 0; i < c->n; i++) {                                      \
      real value = *(const real *)(xp + i * c->stride);                       \
      if (!(value KEEP best)) {                                               \
        bi = i;                                                               \
        best = value;                                                         \
        th_isnan_break(value)                                                 \
      }                                                                       \
    }                                                                         \
    *(real *)vp = best;                                                       \
    *(int64_t *)ip = bi;                                                      \
  }                                                                           \
}                                                                             \
static void THTensor_(NAME##RowsLoop)(char **d, const int64_t *s, int64_t n_, void *ctx_) \
{                                                                             \
  const THTensor_(ReduceCtx) *c = ctx_;                                       \
  for (int64_t j = 0; j < n_; j++) {                                          \
    *(real *)(d[0] + j * s[0]) = *(const real *)(d[2] + j * s[2]);            \
    *(int64_t *)(d[1] + j * s[1]) = 0;                                        \
  }                                                                           \
  for (int64_t i = 0; i < c->n; i++) {                                        \
    const char *xp = d[2] + i * c->stride;                                    \
    for (int64_t j = 0; j < n_; j++) {                                        \
      real x = *(const real *)(xp + j * s[2]);                                \
      real *v = (real *)(d[0] + j * s[0]);                                    \
      if (!(x KEEP *v) && !th_isnan(*v)) {                                    \
        *v = x;                                                               \
        *(int64_t *)(d[1] + j * s[1]) = i;                                    \
      }                                                                       \
    }                                                                         \
  }                                                                           \
}

TH_MAXMIN_KERNELS(max, <=)
TH_MAXMIN_KERNELS(min, >=)

static int THTensor_(reduceDimMaxMin)(THTensor *values_, THLongTensor *indices_, THTensor *t,
                                      int dimension, THIterLoop sliceLoop, THIterLoop rowsLoop)
{
  THIterOperand outs[2] = {THTensor_(iterOperand)(values_), THLongTensor_iterOperand(indices_)};
  return THTensor_(reduceDimN)(2, outs, t, dimension,
                               t->stride[dimension] == 1 ? sliceLoop : rowsLoop);
}

void THTensor_(max)(THTensor *values_, THLongTensor *indices_, THTensor *t, int dimension, int keepdim)
{
  THLongStorage *dim;

  THArgCheck(dimension >= 0 && dimension < THTensor_(nDimension)(t), 2, "dimension %d out of range",
      dimension + TH_INDEX_BASE);

  dim = THTensor_(newSizeOf)(t);
  THLongStorage_set(dim, dimension, 1);
  THTensor_(resize)(values_, dim, NULL);
  THLongTensor_resize(indices_, dim, NULL);
  THLongStorage_free(dim);

  if (THTensor_(reduceDimMaxMin)(values_, indices_, t, dimension,
                                 THTensor_(maxSliceLoop), THTensor_(maxRowsLoop))) {
    /* done (THIter; same results as the legacy layouts below) */
  } else if (t->stride[dimension] == 1) {
    real theMax;
    real value;
    int64_t theIndex;
    int64_t i;
    TH_TENSOR_DIM_APPLY3(real, t, real, values_, int64_t, indices_, dimension,
                         theMax = t_data[0];
                         theIndex = 0;

                         for(i = 0; i < t_size; i++)
                         {
                           value = t_data[i*t_stride];
                           /* This is not the same as value>theMax in the case of NaNs */
                           if(!(value <= theMax))
                           {
                             theIndex = i;
                             theMax = value;
                             th_isnan_break(value)
                           }
                         }
                         *indices__data = theIndex;
                         *values__data = theMax;);
  } else {
    if (THTensor_(nDimension)(t) > 1) {
      THTensor *t0 = THTensor_(newSelect)(t, dimension, 0);
      THTensor_(copy)(values_, t0);
      THTensor_(free)(t0);
    } else {
      THTensor_(fill)(values_, THTensor_(get1d)(t, 0));
    }
    THLongTensor_zero(indices_);

    if(t->size[dimension] == 1) {
      if (!keepdim) {
        THTensor_(squeeze1d)(values_, values_, dimension);
        THLongTensor_squeeze1d(indices_, indices_, dimension);
      }
      return;
    }

    THTensor *tempValues_ = THTensor_(newWithTensor)(values_);
    // tempValues_.expand_as(t)
    tempValues_->size[dimension] = t->size[dimension];
    tempValues_->stride[dimension] = 0;

    THLongTensor *tempIndices_ = THLongTensor_newWithTensor(indices_);
    // tempIndices_.expand_as(t)
    tempIndices_->size[dimension] = t->size[dimension];
    tempIndices_->stride[dimension] = 0;

    TH_TENSOR_APPLY3_D(real, t, real, tempValues_, int64_t, tempIndices_, dimension,
                          if(!(*t_data <= *tempValues__data) && !th_isnan(*tempValues__data)) {
                            *tempValues__data = *t_data;
                            *tempIndices__data = *tempIndices__dimOffset;
                          });

    THTensor_(free)(tempValues_);
    THLongTensor_free(tempIndices_);
  }

  if (!keepdim) {
    THTensor_(squeeze1d)(values_, values_, dimension);
    THLongTensor_squeeze1d(indices_, indices_, dimension);
  }
}

void THTensor_(min)(THTensor *values_, THLongTensor *indices_, THTensor *t, int dimension, int keepdim)
{
  THLongStorage *dim;

  THArgCheck(dimension >= 0 && dimension < THTensor_(nDimension)(t), 2, "dimension %d out of range",
      dimension + TH_INDEX_BASE);

  dim = THTensor_(newSizeOf)(t);
  THLongStorage_set(dim, dimension, 1);
  THTensor_(resize)(values_, dim, NULL);
  THLongTensor_resize(indices_, dim, NULL);
  THLongStorage_free(dim);

  if (THTensor_(reduceDimMaxMin)(values_, indices_, t, dimension,
                                 THTensor_(minSliceLoop), THTensor_(minRowsLoop))) {
    /* done (THIter; same results as the legacy layouts below) */
  } else if (t->stride[dimension] == 1) {
    real theMax;
    real value;
    int64_t theIndex;
    int64_t i;
    TH_TENSOR_DIM_APPLY3(real, t, real, values_, int64_t, indices_, dimension,
                         theMax = t_data[0];
                         theIndex = 0;

                         for(i = 0; i < t_size; i++)
                         {
                           value = t_data[i*t_stride];
                           /* This is not the same as value>theMax in the case of NaNs */
                           if(!(value >= theMax))
                           {
                             theIndex = i;
                             theMax = value;
                             th_isnan_break(value)
                           }
                         }
                         *indices__data = theIndex;
                         *values__data = theMax;);
  } else {
    if (THTensor_(nDimension)(t) > 1) {
      THTensor *t0 = THTensor_(newSelect)(t, dimension, 0);
      THTensor_(copy)(values_, t0);
      THTensor_(free)(t0);
    } else {
      THTensor_(fill)(values_, THTensor_(get1d)(t, 0));
    }
    THLongTensor_zero(indices_);

    if(t->size[dimension] == 1) {
      if (!keepdim) {
        THTensor_(squeeze1d)(values_, values_, dimension);
        THLongTensor_squeeze1d(indices_, indices_, dimension);
      }
      return;
    }

    THTensor *tempValues_ = THTensor_(newWithTensor)(values_);
    // tempValues_.expand_as(t)
    tempValues_->size[dimension] = t->size[dimension];
    tempValues_->stride[dimension] = 0;

    THLongTensor *tempIndices_ = THLongTensor_newWithTensor(indices_);
    // tempIndices_.expand_as(t)
    tempIndices_->size[dimension] = t->size[dimension];
    tempIndices_->stride[dimension] = 0;

    TH_TENSOR_APPLY3_D(real, t, real, tempValues_, int64_t, tempIndices_, dimension,
                          if(!(*t_data >= *tempValues__data) && !th_isnan(*tempValues__data)) {
                            *tempValues__data = *t_data;
                            *tempIndices__data = *tempIndices__dimOffset;
                          });
  }

  if (!keepdim) {
    THTensor_(squeeze1d)(values_, values_, dimension);
    THLongTensor_squeeze1d(indices_, indices_, dimension);
  }
}


void THTensor_(sum)(THTensor *r_, THTensor *t, int dimension, int keepdim)
{
  THLongStorage *dim;

  THArgCheck(dimension >= 0 && dimension < THTensor_(nDimension)(t), 2, "dimension %d out of range",
      dimension + TH_INDEX_BASE);

  dim = THTensor_(newSizeOf)(t);
  THLongStorage_set(dim, dimension, 1);
  THTensor_(resize)(r_, dim, NULL);
  THLongStorage_free(dim);

  if (THTensor_(reduceDim)(r_, t, dimension, t->stride[dimension] == 1
                                               ? THTensor_(sumSliceLoop) : THTensor_(sumRowsLoop))) {
    /* done */
  } else if (t->stride[dimension] == 1) {
    TH_TENSOR_DIM_APPLY2(real, t, real, r_, dimension,
                         accreal sum = 0;
                         int64_t i;
                         for(i = 0; i < t_size; i++)
                           sum += t_data[i*t_stride];
                         *r__data = (real)sum;);
  } else {
    THTensor_(zero)(r_);
    THTensor *temp_ = THTensor_(newWithTensor)(r_);
    // r_.expand_as(t)
    temp_->size[dimension] = t->size[dimension];
    temp_->stride[dimension] = 0;

    TH_TENSOR_APPLY2(real, temp_, real, t, *temp__data = *temp__data + *t_data;);
    THTensor_(free)(temp_);
  }

  if (!keepdim) {
    THTensor_(squeeze1d)(r_, r_, dimension);
  }
}

void THTensor_(prod)(THTensor *r_, THTensor *t, int dimension, int keepdim)
{
  THLongStorage *dim;

  THArgCheck(dimension >= 0 && dimension < THTensor_(nDimension)(t), 2, "dimension %d out of range",
      dimension + TH_INDEX_BASE);

  dim = THTensor_(newSizeOf)(t);
  THLongStorage_set(dim, dimension, 1);
  THTensor_(resize)(r_, dim, NULL);
  THLongStorage_free(dim);

  if (THTensor_(reduceDim)(r_, t, dimension, t->stride[dimension] == 1
                                               ? THTensor_(prodSliceLoop) : THTensor_(prodRowsLoop))) {
    /* done */
  } else if (t->stride[dimension] == 1) {
    TH_TENSOR_DIM_APPLY2(real, t, real, r_, dimension,
                         accreal prod = 1;
                         int64_t i;
                         for(i = 0; i < t_size; i++)
                           prod *= t_data[i*t_stride];
                         *r__data = (real)prod;);
  } else {
    THTensor_(fill)(r_, 1);
    THTensor *temp_ = THTensor_(newWithTensor)(r_);
    // r_.expand_as(t)
    temp_->size[dimension] = t->size[dimension];
    temp_->stride[dimension] = 0;

    TH_TENSOR_APPLY2(real, temp_, real, t, *temp__data = *temp__data * *t_data;);
    THTensor_(free)(temp_);
  }

  if (!keepdim) {
    THTensor_(squeeze1d)(r_, r_, dimension);
  }
}

void THTensor_(cumsum)(THTensor *r_, THTensor *t, int dimension)
{
  THArgCheck(dimension >= 0 && dimension < THTensor_(nDimension)(t), 2, "dimension %d out of range",
      dimension + TH_INDEX_BASE);

  THTensor_(resizeAs)(r_, t);

  TH_TENSOR_DIM_APPLY2(real, t, real, r_, dimension,
                       accreal cumsum = 0;
                       int64_t i;
                       for(i = 0; i < t_size; i++)
                       {
                         cumsum += t_data[i*t_stride];
                         r__data[i*r__stride] = (real)cumsum;
                       });
}

void THTensor_(cumprod)(THTensor *r_, THTensor *t, int dimension)
{
  THArgCheck(dimension >= 0 && dimension < THTensor_(nDimension)(t), 2, "dimension %d out of range",
      dimension + TH_INDEX_BASE);

  THTensor_(resizeAs)(r_, t);

  TH_TENSOR_DIM_APPLY2(real, t, real, r_, dimension,
                       accreal cumprod = 1;
                       int64_t i;
                       for(i = 0; i < t_size; i++)
                       {
                         cumprod *= t_data[i*t_stride];
                         r__data[i*r__stride] = (real)cumprod;
                       });
}


void THTensor_(sign)(THTensor *r_, THTensor *t)
{
  THTensor_(resizeAs)(r_, t);

#if defined (TH_REAL_IS_BYTE)
  TH_TENSOR_APPLY2(real, r_, real, t,
    if (*t_data > 0) *r__data = 1;
    else *r__data = 0;);
#else
  TH_TENSOR_APPLY2(real, r_, real, t,
    if (*t_data > 0) *r__data = 1;
    else if (*t_data < 0) *r__data = -1;
    else *r__data = 0;);
#endif
}


accreal THTensor_(trace)(THTensor *t)
{
  real *t_data = THTensor_(data)(t);
  accreal sum = 0;
  int64_t i = 0;
  int64_t t_stride_0, t_stride_1, t_diag_size;

  THArgCheck(THTensor_(nDimension)(t) == 2, 1, "expected a matrix");

  t_stride_0 = THTensor_(stride)(t, 0);
  t_stride_1 = THTensor_(stride)(t, 1);
  t_diag_size = THMin(THTensor_(size)(t, 0), THTensor_(size)(t, 1));
  while(i < t_diag_size)
  {
    sum += t_data[i*(t_stride_0+t_stride_1)];
    i++;
  }

  return sum;
}

void THTensor_(cross)(THTensor *r_, THTensor *a, THTensor *b, int dimension)
{
  int i;

  if(THTensor_(nDimension)(a) != THTensor_(nDimension)(b))
    THError("inconsistent tensor dimension %dD, %dD",
        THTensor_(nDimension)(a), THTensor_(nDimension)(b));

  for(i = 0; i < THTensor_(nDimension)(a); i++)
  {
    if(THTensor_(size)(a, i) != THTensor_(size)(b, i)) {
        THDescBuff ba = THTensor_(sizeDesc)(a);
        THDescBuff bb = THTensor_(sizeDesc)(b);
        THError("inconsistent tensor sizes %s, %s", ba.str, bb.str);
    }
  }

  if(dimension < 0)
  {
    for(i = 0; i < THTensor_(nDimension)(a); i++)
    {
      if(THTensor_(size)(a, i) == 3)
      {
        dimension = i;
        break;
      }
    }
    if(dimension < 0) {
      THDescBuff ba = THTensor_(sizeDesc)(a);
      THError("no dimension of size 3 in a: %s", ba.str);
    }
  }

  THArgCheck(dimension >= 0 && dimension < THTensor_(nDimension)(a), 3, "dimension %d out of range",
      dimension + TH_INDEX_BASE);
  THArgCheck(THTensor_(size)(a, dimension) == 3, 3, "dimension %d does not have size 3",
      dimension + TH_INDEX_BASE);

  THTensor_(resizeAs)(r_, a);

  TH_TENSOR_DIM_APPLY3(real, a, real, b, real, r_, dimension,
                       r__data[0*r__stride] = a_data[1*a_stride]*b_data[2*b_stride] - a_data[2*a_stride]*b_data[1*b_stride];
                       r__data[1*r__stride] = a_data[2*a_stride]*b_data[0*b_stride] - a_data[0*a_stride]*b_data[2*b_stride];
                       r__data[2*r__stride] = a_data[0*a_stride]*b_data[1*b_stride] - a_data[1*a_stride]*b_data[0*b_stride];);
}

void THTensor_(cmax)(THTensor *r, THTensor *t, THTensor *src) {
  THTensor_(resizeAs)(r, t);
  TH_TENSOR_APPLY3(real, r, real, t, real, src,
                   *r_data = *t_data > *src_data ? *t_data : *src_data;);
}

void THTensor_(cmin)(THTensor *r, THTensor *t, THTensor *src) {
  THTensor_(resizeAs)(r, t);
  TH_TENSOR_APPLY3(real, r, real, t, real, src,
                   *r_data = *t_data < *src_data ? *t_data : *src_data;);
}

void THTensor_(cmaxValue)(THTensor *r, THTensor *t, real value) {
  THTensor_(resizeAs)(r, t);
  TH_TENSOR_APPLY2(real, r, real, t,
                   *r_data = *t_data > value ? *t_data : value;);
}

void THTensor_(cminValue)(THTensor *r, THTensor *t, real value) {
  THTensor_(resizeAs)(r, t);
  TH_TENSOR_APPLY2(real, r, real, t,
                   *r_data = *t_data < value ? *t_data : value;);
}

void THTensor_(zeros)(THTensor *r_, THLongStorage *size)
{
  THTensor_(resize)(r_, size, NULL);
  THTensor_(zero)(r_);
}

void THTensor_(zerosLike)(THTensor *r_, THTensor *input)
{
  THTensor_(resizeAs)(r_, input);
  THTensor_(zero)(r_);
}

void THTensor_(onesLike)(THTensor *r_, THTensor *input)
{
  THTensor_(resizeAs)(r_, input);
  THTensor_(fill)(r_, 1);
}

void THTensor_(ones)(THTensor *r_, THLongStorage *size)
{
  THTensor_(resize)(r_, size, NULL);
  THTensor_(fill)(r_, 1);
}

void THTensor_(diag)(THTensor *r_, THTensor *t, int k)
{
  THArgCheck(THTensor_(nDimension)(t) == 1 || THTensor_(nDimension)(t) == 2, 1, "matrix or a vector expected");

  if(THTensor_(nDimension)(t) == 1)
  {
    real *t_data = THTensor_(data)(t);
    int64_t t_stride_0 = THTensor_(stride)(t, 0);
    int64_t t_size = THTensor_(size)(t, 0);
    int64_t sz = t_size + (k >= 0 ? k : -k);
    real *r__data;
    int64_t r__stride_0;
    int64_t r__stride_1;
    int64_t i;

    THTensor_(resize2d)(r_, sz, sz);
    THTensor_(zero)(r_);
    r__data = THTensor_(data)(r_);
    r__stride_0 = THTensor_(stride)(r_, 0);
    r__stride_1 = THTensor_(stride)(r_, 1);
    r__data += (k >= 0 ? k*r__stride_1 : -k*r__stride_0);

    for(i = 0; i < t_size; i++)
      r__data[i*(r__stride_0+r__stride_1)] = t_data[i*t_stride_0];
  }
  else
  {
    real *t_data = THTensor_(data)(t);
    int64_t t_stride_0 = THTensor_(stride)(t, 0);
    int64_t t_stride_1 = THTensor_(stride)(t, 1);
    int64_t sz;
    real *r__data;
    int64_t r__stride_0;
    int64_t i;

    if(k >= 0)
      sz = THMin(THTensor_(size)(t, 0), THTensor_(size)(t, 1)-k);
    else
      sz = THMin(THTensor_(size)(t, 0)+k, THTensor_(size)(t, 1));
    THTensor_(resize1d)(r_, sz);
    r__data = THTensor_(data)(r_);
    r__stride_0 = THTensor_(stride)(r_, 0);

    t_data += (k >= 0 ? k*t_stride_1 : -k*t_stride_0);
    for(i = 0; i < sz; i++)
      r__data[i*r__stride_0] = t_data[i*(t_stride_0+t_stride_1)];
  }
}

void THTensor_(eye)(THTensor *r_, int64_t n, int64_t m)
{
  real *r__data;
  int64_t i, sz;

  THArgCheck(n > 0, 1, "invalid argument");

  if(m <= 0)
    m = n;

  THTensor_(resize2d)(r_, n, m);
  THTensor_(zero)(r_);

  i = 0;
  r__data = THTensor_(data)(r_);
  sz = THMin(THTensor_(size)(r_, 0), THTensor_(size)(r_, 1));
  for(i = 0; i < sz; i++)
    r__data[i*(r_->stride[0]+r_->stride[1])] = 1;
}


void THTensor_(range)(THTensor *r_, accreal xmin, accreal xmax, accreal step)
{
  ptrdiff_t size;
  real i = 0;

  THArgCheck(step > 0 || step < 0, 3, "step must be a non-null number");
  THArgCheck(((step > 0) && (xmax >= xmin)) || ((step < 0) && (xmax <= xmin))
              , 2, "upper bound and larger bound incoherent with step sign");

  size = (ptrdiff_t) (((xmax - xmin) / step) + 1);

  if (THTensor_(nElement)(r_) != size) {
    THTensor_(resize1d)(r_, size);
  }

  TH_TENSOR_APPLY(real, r_, *r__data = xmin + (i++)*step;);
}

void THTensor_(arange)(THTensor *r_, accreal xmin, accreal xmax, accreal step) {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
  int m = fmod(xmax - xmin,step) == 0;
#else
  int m = (xmax - xmin) % step == 0;
#endif
  if (m)
    xmax -= step;
  THTensor_(range)(r_,xmin,xmax,step);
}

void THTensor_(randperm)(THTensor *r_, THGenerator *_generator, int64_t n)
{
  real *r__data;
  int64_t r__stride_0;
  int64_t i;

  THArgCheck(n > 0, 1, "must be strictly positive");

  THTensor_(resize1d)(r_, n);
  r__data = THTensor_(data)(r_);
  r__stride_0 = THTensor_(stride)(r_,0);

  for(i = 0; i < n; i++)
    r__data[i*r__stride_0] = (real)(i);

  for(i = 0; i < n-1; i++)
  {
    int64_t z = THRandom_random(_generator) % (n-i);
    real sav = r__data[i*r__stride_0];
    r__data[i*r__stride_0] = r__data[(z+i)*r__stride_0];
    r__data[(z+i)*r__stride_0] = sav;
  }
}

void THTensor_(reshape)(THTensor *r_, THTensor *t, THLongStorage *size)
{
  THTensor_(resize)(r_, size, NULL);
  THTensor_(copy)(r_, t);
}

/* I cut and pasted (slightly adapted) the quicksort code from
   Sedgewick's 1978 "Implementing Quicksort Programs" article
   http://www.csie.ntu.edu.tw/~b93076/p847-sedgewick.pdf

   It is the state of the art existing implementation. The macros
   are here to make as close a match as possible to the pseudocode of
   Program 2 p.851

   Note that other partition schemes exist, and are typically presented
   in textbook, but those are less efficient. See e.g.
   http://cs.stackexchange.com/questions/11458/quicksort-partitioning-hoare-vs-lomuto

   Julien, November 12th 2013
*/
#define MAX_LEVELS  300
#define M_SMALL 10 /* Limit for small subfiles */

#define ARR(III) arr[(III)*stride]
#define IDX(III) idx[(III)*stride]

#define LONG_SWAP(AAA, BBB) swap = AAA; AAA = BBB; BBB = swap
#define REAL_SWAP(AAA, BBB) rswap = AAA; AAA = BBB; BBB = rswap

#define ARR_SWAP(III, JJJ) \
  REAL_SWAP(ARR(III), ARR(JJJ));

#define BOTH_SWAP(III, JJJ) \
  REAL_SWAP(ARR(III), ARR(JJJ)); \
  LONG_SWAP(IDX(III), IDX(JJJ))

static void THTensor_(quicksortascend)(real *arr, int64_t *idx, int64_t elements, int64_t stride)
{
  int64_t beg[MAX_LEVELS], end[MAX_LEVELS], i, j, L, R, P, swap, pid, stack = 0, sz_right, sz_left;
  real rswap, piv;
  unsigned char done = 0;

  /* beg[0]=0; end[0]=elements; */
  stack = 0;
  L = 0; R = elements-1;
  done = elements-1 <= M_SMALL;

  while(!done) {
      /* Use median of three for pivot choice */
    P=(L+R)>>1;
    BOTH_SWAP(P, L+1);
    if (ARR(L+1) > ARR(R)) { BOTH_SWAP(L+1, R); }
    if (ARR(L) > ARR(R)) { BOTH_SWAP(L, R); }
    if (ARR(L+1) > ARR(L)) { BOTH_SWAP(L+1, L); }

    i = L+1; j = R; piv = ARR(L); pid = IDX(L);

    do {
      do { i = i+1; } while(ARR(i) < piv);
      do { j = j-1; } while(ARR(j) > piv);
      if (j < i)
          break;
      BOTH_SWAP(i, j);
    } while(1);
    BOTH_SWAP(L, j);
    /* Left subfile is (L, j-1) */
    /* Right subfile is (i, R) */
    sz_left = j-L;
    sz_right = R-i+1;
    if (sz_left <= M_SMALL && sz_right <= M_SMALL) {
      /* both subfiles are small */
      /* if stack empty */
      if (stack == 0) {
        done = 1;
      } else {
        stack--;
        L = beg[stack];
        R = end[stack];
      }
    } else if (sz_left <= M_SMALL || sz_right <= M_SMALL) {
      /* exactly one of the subfiles is small */
      /* (L,R) = large subfile */
      if (sz_left > sz_right) {
        /* Implicit: L = L; */
        R = j-1;
      } else {
        L = i;
        /* Implicit: R = R; */
      }
    } else {
      /* none of the subfiles is small */
      /* push large subfile */
      /* (L,R) = small subfile */
      if (sz_left > sz_right) {
        beg[stack] = L;
        end[stack] = j-1;
        stack++;
        L = i;
        /* Implicit: R = R */
      } else {
        beg[stack] = i;
        end[stack] = R;
        stack++;
        /* Implicit: L = L; */
        R = j-1;
      }
    }
  } /* while not done */
  /* Now insertion sort on the concatenation of subfiles */
  for(i=elements-2; i>=0; i--) {
    if (ARR(i) > ARR(i+1)) {
      piv = ARR(i);
      pid = IDX(i);
      j = i+1;
      do {
        ARR(j-1) = ARR(j);
        IDX(j-1) = IDX(j);
        j = j+1;
      } while(j < elements && ARR(j) < piv);
      ARR(j-1) = piv;
      IDX(j-1) = pid;
     }
  }
}

static void THTensor_(quicksortdescend)(real *arr, int64_t *idx, int64_t elements, int64_t stride)
{
  int64_t beg[MAX_LEVELS], end[MAX_LEVELS], i, j, L, R, P, swap, pid, stack = 0, sz_right, sz_left;
  real rswap, piv;
  unsigned char done = 0;

  /* beg[0]=0; end[0]=elements; */
  stack = 0;
  L = 0; R = elements-1;
  done = elements-1 <= M_SMALL;

  while(!done) {
      /* Use median of three for pivot choice */
    P=(L+R)>>1;
    BOTH_SWAP(P, L+1);
    if (ARR(L+1) < ARR(R)) { BOTH_SWAP(L+1, R); }
    if (ARR(L) < ARR(R)) { BOTH_SWAP(L, R); }
    if (ARR(L+1) < ARR(L)) { BOTH_SWAP(L+1, L); }

    i = L+1; j = R; piv = ARR(L); pid = IDX(L);

    do {
      do { i = i+1; } while(ARR(i) > piv);
      do { j = j-1; } while(ARR(j) < piv);
      if (j < i)
          break;
      BOTH_SWAP(i, j);
    } while(1);
    BOTH_SWAP(L, j);
    /* Left subfile is (L, j-1) */
    /* Right subfile is (i, R) */
    sz_left = j-L;
    sz_right = R-i+1;
    if (sz_left <= M_SMALL && sz_right <= M_SMALL) {
      /* both subfiles are small */
      /* if stack empty */
      if (stack == 0) {
        done = 1;
      } else {
        stack--;
        L = beg[stack];
        R = end[stack];
      }
    } else if (sz_left <= M_SMALL || sz_right <= M_SMALL) {
      /* exactly one of the subfiles is small */
      /* (L,R) = large subfile */
      if (sz_left > sz_right) {
        /* Implicit: L = L; */
        R = j-1;
      } else {
        L = i;
        /* Implicit: R = R; */
      }
    } else {
      /* none of the subfiles is small */
      /* push large subfile */
      /* (L,R) = small subfile */
      if (sz_left > sz_right) {
        beg[stack] = L;
        end[stack] = j-1;
        stack++;
        L = i;
        /* Implicit: R = R */
      } else {
        beg[stack] = i;
        end[stack] = R;
        stack++;
        /* Implicit: L = L; */
        R = j-1;
      }
    }
  } /* while not done */
  /* Now insertion sort on the concatenation of subfiles */
  for(i=elements-2; i>=0; i--) {
    if (ARR(i) < ARR(i+1)) {
      piv = ARR(i);
      pid = IDX(i);
      j = i+1;
      do {
        ARR(j-1) = ARR(j);
        IDX(j-1) = IDX(j);
        j = j+1;
      } while(j < elements && ARR(j) > piv);
      ARR(j-1) = piv;
      IDX(j-1) = pid;
     }
  }
}

#undef MAX_LEVELS
#undef M_SMALL

void THTensor_(sort)(THTensor *rt_, THLongTensor *ri_, THTensor *t, int dimension, int descendingOrder)
{
  THArgCheck(dimension >= 0 && dimension < THTensor_(nDimension)(t), 2, "invalid dimension %d",
      dimension + TH_INDEX_BASE);

  THTensor_(resizeAs)(rt_, t);
  THTensor_(copy)(rt_, t);

  {
    THLongStorage *size = THTensor_(newSizeOf)(t);
    THLongTensor_resize(ri_, size, NULL);
    THLongStorage_free(size);
  }

  if(descendingOrder)
  {
    TH_TENSOR_DIM_APPLY2(real, rt_, int64_t, ri_, dimension,
                         int64_t i;
                         for(i = 0; i < ri__size; i++)
                           ri__data[i*ri__stride] = i;
                         THTensor_(quicksortdescend)(rt__data, ri__data, rt__size, rt__stride);)
      }
  else
  {
    TH_TENSOR_DIM_APPLY2(real, rt_, int64_t, ri_, dimension,
                         int64_t i;
                         for(i = 0; i < ri__size; i++)
                           ri__data[i*ri__stride] = i;
                         THTensor_(quicksortascend)(rt__data, ri__data, rt__size, rt__stride);)
      }
}

/* Implementation of the Quickselect algorithm, based on Nicolas Devillard's
public domain implementation at http://ndevilla.free.fr/median/median/
Adapted similarly to the above Quicksort algorithm.
This version does not produce indices along with values. */
static void THTensor_(quickselectnoidx)(real *arr, int64_t k, int64_t elements, int64_t stride)
{
  int64_t P, L, R, i, j, swap;
  real rswap, piv;
  L = 0;
  R = elements-1;

  do {
    if (R <= L) /* One element only */
      return;

    if (R == L+1) {  /* Two elements only */
      if (ARR(L) > ARR(R)) {
        ARR_SWAP(L, R);
      }
      return;
    }

    /* Use median of three for pivot choice */
    P=(L+R)>>1;
    ARR_SWAP(P, L+1);
    if (ARR(L+1) > ARR(R)) { ARR_SWAP(L+1, R); }
    if (ARR(L) > ARR(R)) { ARR_SWAP(L, R); }
    if (ARR(L+1) > ARR(L)) { ARR_SWAP(L+1, L); }

    i = L+1;
    j = R;
    piv = ARR(L);
    do {
      do i++; while(ARR(i) < piv);
      do j--; while(ARR(j) > piv);
      if (j < i)
        break;
      ARR_SWAP(i, j);
    } while(1);
    ARR_SWAP(L, j);

    /* Re-set active partition */
    if (j <= k) L=i;
    if (j >= k) R=j-1;
  } while(1);
}

/* Implementation of the Quickselect algorithm, based on Nicolas Devillard's
public domain implementation at http://ndevilla.free.fr/median/median/
Adapted similarly to the above Quicksort algorithm. */
static void THTensor_(quickselect)(real *arr, int64_t *idx, int64_t k, int64_t elements, int64_t stride)
{
  int64_t P, L, R, i, j, swap, pid;
  real rswap, piv;
  L = 0;
  R = elements-1;

  do {
    if (R <= L) /* One element only */
      return;

    if (R == L+1) {  /* Two elements only */
      if (ARR(L) > ARR(R)) {
        BOTH_SWAP(L, R);
      }
      return;
    }

    /* Use median of three for pivot choice */
    P=(L+R)>>1;
    BOTH_SWAP(P, L+1);
    if (ARR(L+1) > ARR(R)) { BOTH_SWAP(L+1, R); }
    if (ARR(L) > ARR(R)) { BOTH_SWAP(L, R); }
    if (ARR(L+1) > ARR(L)) { BOTH_SWAP(L+1, L); }

    i = L+1;
    j = R;
    piv = ARR(L);
    pid = IDX(L);
    do {
      do i++; while(ARR(i) < piv);
      do j--; while(ARR(j) > piv);
      if (j < i)
        break;
      BOTH_SWAP(i, j);
    } while(1);
    BOTH_SWAP(L, j);

    /* Re-set active partition */
    if (j <= k) L=i;
    if (j >= k) R=j-1;
  } while(1);
}

#undef ARR
#undef IDX
#undef LONG_SWAP
#undef REAL_SWAP
#undef BOTH_SWAP

void THTensor_(mode)(THTensor *values_, THLongTensor *indices_, THTensor *t, int dimension, int keepdim)
{
  THLongStorage *dim;
  THTensor *temp_;
  THLongTensor *tempi_;
  real *temp__data;
  int64_t *tempi__data;
  int64_t t_size_dim;

  THArgCheck(dimension >= 0 && dimension < THTensor_(nDimension)(t), 3, "dimension out of range");

  dim = THTensor_(newSizeOf)(t);
  THLongStorage_set(dim, dimension, 1);
  THTensor_(resize)(values_, dim, NULL);
  THLongTensor_resize(indices_, dim, NULL);
  THLongStorage_free(dim);

  t_size_dim = THTensor_(size)(t, dimension);

  temp_ = THTensor_(new)();
  THTensor_(resize1d)(temp_, t_size_dim);
  temp__data = THTensor_(data)(temp_);

  tempi_ = THLongTensor_new();
  THLongTensor_resize1d(tempi_, t_size_dim);
  tempi__data = THLongTensor_data(tempi_);

  TH_TENSOR_DIM_APPLY3(real, t, real, values_, int64_t, indices_, dimension,
                       int64_t i;
                       real mode = 0;
                       int64_t modei = 0;
                       int64_t temp_freq = 0;
                       int64_t max_freq = 0;
                       for(i = 0; i < t_size_dim; i++)
                          temp__data[i] = t_data[i*t_stride];
                       for(i = 0; i < t_size_dim; i++)
                          tempi__data[i] = i;
                       THTensor_(quicksortascend)(temp__data, tempi__data, t_size_dim, 1);

                       for(i = 0; i < t_size_dim; i++)
                       {
                          temp_freq++;
                          if ((i == t_size_dim - 1) || (temp__data[i] != temp__data[i+1]))
                          {
                              if (temp_freq > max_freq)
                              {
                                 mode = temp__data[i];
                                 modei = tempi__data[i];
                                 max_freq = temp_freq;
                              }
                              temp_freq = 0;
                          }
                       }
                       *values__data = mode;
                       *indices__data = modei;);

  THTensor_(free)(temp_);
  THLongTensor_free(tempi_);
  if (!keepdim) {
    THTensor_(squeeze1d)(values_, values_, dimension);
    THLongTensor_squeeze1d(indices_, indices_, dimension);
  }
}

void THTensor_(kthvalue)(THTensor *values_, THLongTensor *indices_, THTensor *t, int64_t k, int dimension, int keepdim)
{
  THLongStorage *dim;
  THTensor *temp_;
  THLongTensor *tempi_;
  real *temp__data;
  int64_t *tempi__data;
  int64_t t_size_dim;

  THArgCheck(dimension >= 0 && dimension < THTensor_(nDimension)(t), 3, "dimension out of range");
  THArgCheck(k > 0 && k <= t->size[dimension], 2, "selected index out of range");

  dim = THTensor_(newSizeOf)(t);
  THLongStorage_set(dim, dimension, 1);
  THTensor_(resize)(values_, dim, NULL);
  THLongTensor_resize(indices_, dim, NULL);
  THLongStorage_free(dim);

  t_size_dim = THTensor_(size)(t, dimension);

  temp_ = THTensor_(new)();
  THTensor_(resize1d)(temp_, t_size_dim);
  temp__data = THTensor_(data)(temp_);

  tempi_ = THLongTensor_new();
  THLongTensor_resize1d(tempi_, t_size_dim);
  tempi__data = THLongTensor_data(tempi_);

  TH_TENSOR_DIM_APPLY3(real, t, real, values_, int64_t, indices_, dimension,
                       int64_t i;
                       for(i = 0; i < t_size_dim; i++)
                          temp__data[i] = t_data[i*t_stride];
                       for(i = 0; i < t_size_dim; i++)
                          tempi__data[i] = i;
                       THTensor_(quickselect)(temp__data, tempi__data, k - 1, t_size_dim, 1);
                       *values__data = temp__data[k-1];
                       *indices__data = tempi__data[k-1];);

  THTensor_(free)(temp_);
  THLongTensor_free(tempi_);
  if (!keepdim) {
    THTensor_(squeeze1d)(values_, values_, dimension);
    THLongTensor_squeeze1d(indices_, indices_, dimension);
  }
}

void THTensor_(median)(THTensor *values_, THLongTensor *indices_, THTensor *t, int dimension, int keepdim)
{
  int64_t t_size_dim, k;

  THArgCheck(dimension >= 0 && dimension < THTensor_(nDimension)(t), 3, "dimension out of range");

  t_size_dim = THTensor_(size)(t, dimension);
  k = (t_size_dim-1) >> 1; /* take middle or one-before-middle element */

  THTensor_(kthvalue)(values_, indices_, t, k+1, dimension, keepdim);
}

void THTensor_(topk)(THTensor *rt_, THLongTensor *ri_, THTensor *t, int64_t k, int dim, int dir, int sorted)
{
  int numDims = THTensor_(nDimension)(t);
  THArgCheck(dim >= 0 && dim < numDims, 3, "dim not in range");

  int64_t sliceSize = THTensor_(size)(t, dim);
  THArgCheck(k > 0 && k <= sliceSize, 2, "k not in range for dimension");

  THTensor *tmpResults = THTensor_(new)();
  THTensor_(resize1d)(tmpResults, sliceSize);
  real *tmp__data = THTensor_(data)(tmpResults);

  THLongTensor *tmpIndices = THLongTensor_new();
  THLongTensor_resize1d(tmpIndices, sliceSize);
  int64_t *tmpi__data = THLongTensor_data(tmpIndices);

  THLongStorage *topKSize = THTensor_(newSizeOf)(t);
  THLongStorage_set(topKSize, dim, k);
  THTensor_(resize)(rt_, topKSize, NULL);
  THLongTensor_resize(ri_, topKSize, NULL);
  THLongStorage_free(topKSize);

  if (dir) {
    /* k largest elements, descending order (optional: see sorted) */
    int64_t K = sliceSize - k;
    TH_TENSOR_DIM_APPLY3(real, t, real, rt_, int64_t, ri_, dim,
                         int64_t i;
                         for(i = 0; i < sliceSize; i++)
                         {
                           tmp__data[i] = t_data[i*t_stride];
                           tmpi__data[i] = i;
                         }
                         if (K > 0)
                           THTensor_(quickselect)(tmp__data, tmpi__data, K - 1, sliceSize, 1);
                         if (sorted)
                           THTensor_(quicksortdescend)(tmp__data + K, tmpi__data + K, k, 1);
                         for(i = 0; i < k; i++)
                         {
                           rt__data[i*rt__stride] = tmp__data[i + K];
                           ri__data[i*ri__stride] = tmpi__data[i + K];
                         })
  }
  else {
    /* k smallest elements, ascending order (optional: see sorted) */
    TH_TENSOR_DIM_APPLY3(real, t, real, rt_, int64_t, ri_, dim,
                         int64_t i;
                         for(i = 0; i < sliceSize; i++)
                         {
                           tmp__data[i] = t_data[i*t_stride];
                           tmpi__data[i] = i;
                         }
                         THTensor_(quickselect)(tmp__data, tmpi__data, k - 1, sliceSize, 1);
                         if (sorted)
                           THTensor_(quicksortascend)(tmp__data, tmpi__data, k - 1, 1);
                         for(i = 0; i < k; i++)
                         {
                           rt__data[i*rt__stride] = tmp__data[i];
                           ri__data[i*ri__stride] = tmpi__data[i];
                         })
  }

  THTensor_(free)(tmpResults);
  THLongTensor_free(tmpIndices);
}

void THTensor_(tril)(THTensor *r_, THTensor *t, int64_t k)
{
  int64_t t_size_0, t_size_1;
  int64_t t_stride_0, t_stride_1;
  int64_t r__stride_0, r__stride_1;
  real *t_data, *r__data;
  int64_t r, c;

  THArgCheck(THTensor_(nDimension)(t) == 2, 1, "expected a matrix");

  THTensor_(resizeAs)(r_, t);

  t_size_0 = THTensor_(size)(t, 0);
  t_size_1 = THTensor_(size)(t, 1);
  t_stride_0 = THTensor_(stride)(t, 0);
  t_stride_1 = THTensor_(stride)(t, 1);
  r__stride_0 = THTensor_(stride)(r_, 0);
  r__stride_1 = THTensor_(stride)(r_, 1);
  r__data = THTensor_(data)(r_);
  t_data = THTensor_(data)(t);

  for(r = 0; r < t_size_0; r++)
  {
    int64_t sz = THMin(r+k+1, t_size_1);
    for(c = THMax(0, r+k+1); c < t_size_1; c++)
      r__data[r*r__stride_0+c*r__stride_1] = 0;
    for(c = 0; c < sz; c++)
      r__data[r*r__stride_0+c*r__stride_1] = t_data[r*t_stride_0+c*t_stride_1];
  }
}

void THTensor_(triu)(THTensor *r_, THTensor *t, int64_t k)
{
  int64_t t_size_0, t_size_1;
  int64_t t_stride_0, t_stride_1;
  int64_t r__stride_0, r__stride_1;
  real *t_data, *r__data;
  int64_t r, c;

  THArgCheck(THTensor_(nDimension)(t) == 2, 1, "expected a matrix");

  THTensor_(resizeAs)(r_, t);

  t_size_0 = THTensor_(size)(t, 0);
  t_size_1 = THTensor_(size)(t, 1);
  t_stride_0 = THTensor_(stride)(t, 0);
  t_stride_1 = THTensor_(stride)(t, 1);
  r__stride_0 = THTensor_(stride)(r_, 0);
  r__stride_1 = THTensor_(stride)(r_, 1);
  r__data = THTensor_(data)(r_);
  t_data = THTensor_(data)(t);

  for(r = 0; r < t_size_0; r++)
  {
    int64_t sz = THMin(r+k, t_size_1);
    for(c = THMax(0, r+k); c < t_size_1; c++)
      r__data[r*r__stride_0+c*r__stride_1] = t_data[r*t_stride_0+c*t_stride_1];
    for(c = 0; c < sz; c++)
      r__data[r*r__stride_0+c*r__stride_1] = 0;
  }
}

void THTensor_(cat)(THTensor *r_, THTensor *ta, THTensor *tb, int dimension)
{
  THTensor* inputs[2];
  inputs[0] = ta;
  inputs[1] = tb;
  THTensor_(catArray)(r_, inputs, 2, dimension);
}

void THTensor_(catArray)(THTensor *result, THTensor **inputs, int numInputs, int dimension)
{
  THLongStorage *size;
  int i, j;
  int64_t offset;
  int maxDim = dimension + 1;
  int allEmpty = 1;
  int allContiguous = 1;

  // cat_dimension is the actual dimension we cat along
  int cat_dimension = dimension;

  for (i = 0; i < numInputs; i++)
  {
    maxDim = THMax(maxDim, inputs[i]->nDimension);
  }

  // When the user input dimension is -1 (i.e. -2 in C)
  // Then we pick the maximum last dimension across all tensors.
  if ( dimension + TH_INDEX_BASE == -1 )
  {
    cat_dimension = maxDim?(maxDim-1):0;
  }

  THArgCheck(numInputs > 0, 3, "invalid number of inputs %d", numInputs);
  THArgCheck(cat_dimension >= 0, 4, "invalid dimension %d", dimension + TH_INDEX_BASE);

  size = THLongStorage_newWithSize(maxDim);

  for(i = 0; i < maxDim; i++)
  {
    // dimSize is either the size of the dim if it exists, either 1 if #dim > 0, otherwise 0
    int64_t dimSize = i < inputs[0]->nDimension ? inputs[0]->size[i] : THMin(inputs[0]->nDimension, 1);
    if (i == cat_dimension)
    {
      for (j = 1; j < numInputs; j++)
      {
        // accumulate the size over the dimension we want to cat on.
        // Empty tensors are allowed
        dimSize += i < inputs[j]->nDimension ? inputs[j]->size[i] : THMin(inputs[j]->nDimension, 1);
      }
    }
    else
    {
      for (j = 1; j < numInputs; j++)
      {
        int64_t sz = (i < inputs[j]->nDimension ? inputs[j]->size[i] : THMin(inputs[j]->nDimension, 1));
        // If it's a dimension we're not catting on
        // Then fail if sizes are different AND > 0
        if (dimSize != sz && dimSize && sz)
        {
          THLongStorage_free(size);
          THError("inconsistent tensor sizes");
        }
        else if(!dimSize)
        {
          dimSize = sz;
        }
      }
    }
    allEmpty = allEmpty && !dimSize;
    size->data[i] = dimSize;
  }

  // Initiate catting and resizing
  // If at least one of the input is not empty
  if (!allEmpty)
  {
    THTensor_(resize)(result, size, NULL);

    // Check contiguity of all inputs and result
    for (i = 0; i < numInputs; i++) {
      if(inputs[i]->nDimension) {
        allContiguous = allContiguous && THTensor_(isContiguous)(inputs[i]);
      }
    }
    allContiguous = allContiguous && THTensor_(isContiguous)(result);

    // First path is for contiguous inputs along dim 1
    // Second path for non-contiguous
    if (cat_dimension == 0 && allContiguous)
    {
      real* result_data = result->storage->data + result->storageOffset;
      offset = 0;
      for (j = 0; j < numInputs; j++)
      {
        if (inputs[j]->nDimension)
        {
          THTensor* input0 = inputs[j];
          real* input0_data = input0->storage->data + input0->storageOffset;
          int64_t input0_size = THTensor_(nElement)(input0);
          memcpy(result_data + offset, input0_data, input0_size*sizeof(real));
          offset += input0_size;
        }
      }
    }
    else
    {
      offset = 0;
      for (j = 0; j < numInputs; j++)
      {
        if (inputs[j]->nDimension)
        {
          int64_t dimSize = cat_dimension < inputs[j]->nDimension ? inputs[j]->size[cat_dimension] : 1;
          THTensor *nt = THTensor_(newWithTensor)(result);
          THTensor_(narrow)(nt, NULL, cat_dimension, offset, dimSize);
          THTensor_(copy)(nt, inputs[j]);
          THTensor_(free)(nt);
          offset += dimSize;
        }
      }
    }
  }
  THLongStorage_free(size);
}

int THTensor_(equal)(THTensor *ta, THTensor* tb)
{
  int equal = 1;
  if(!THTensor_(isSameSizeAs)(ta, tb))
    return 0;

  if (THTensor_(isContiguous)(ta) && THTensor_(isContiguous)(tb)) {
    real *tap = THTensor_(data)(ta);
    real *tbp = THTensor_(data)(tb);
    ptrdiff_t sz = THTensor_(nElement)(ta);
    ptrdiff_t i;
    for (i=0; i<sz; ++i){
      if(tap[i] != tbp[i]) return 0;
    }
  } else {
    // Short-circuit the apply function on inequality
    TH_TENSOR_APPLY2(real, ta, real, tb,
                     if (equal && *ta_data != *tb_data) {
                        equal = 0;
                        TH_TENSOR_APPLY_hasFinished = 1; break;
                     })
  }
  return equal;
}

#define TENSOR_IMPLEMENT_LOGICAL(NAME,OP)				\
  void THTensor_(NAME##Value)(THByteTensor *r_, THTensor* t, real value)	\
  {									\
    THByteTensor_resizeNd(r_, t->nDimension, t->size, NULL);		\
    TH_TENSOR_APPLY2(unsigned char, r_, real, t,			\
		     *r__data = (*t_data OP value) ? 1 : 0;); \
  }									\
  void THTensor_(NAME##ValueT)(THTensor* r_, THTensor* t, real value)	\
  {									\
    THTensor_(resizeNd)(r_, t->nDimension, t->size, NULL);		\
    TH_TENSOR_APPLY2(real, r_, real, t,					\
		     *r__data = (*t_data OP value) ? 1 : 0;); \
  }									\
  void THTensor_(NAME##Tensor)(THByteTensor *r_, THTensor *ta, THTensor *tb) \
  {									\
    THByteTensor_resizeNd(r_, ta->nDimension, ta->size, NULL);		\
    TH_TENSOR_APPLY3(unsigned char, r_, real, ta, real, tb,		\
		     *r__data = (*ta_data OP *tb_data) ? 1 : 0;); \
  }									\
  void THTensor_(NAME##TensorT)(THTensor *r_, THTensor *ta, THTensor *tb) \
  {									\
    THTensor_(resizeNd)(r_, ta->nDimension, ta->size, NULL);		\
    TH_TENSOR_APPLY3(real, r_, real, ta, real, tb,			\
		     *r__data = (*ta_data OP *tb_data) ? 1 : 0;); \
  }									\


TENSOR_IMPLEMENT_LOGICAL(lt,<)
TENSOR_IMPLEMENT_LOGICAL(gt,>)
TENSOR_IMPLEMENT_LOGICAL(le,<=)
TENSOR_IMPLEMENT_LOGICAL(ge,>=)
TENSOR_IMPLEMENT_LOGICAL(eq,==)
TENSOR_IMPLEMENT_LOGICAL(ne,!=)

/* Unary functions run through THIter (parallel, coalesced; same per-element
   CFUNC call, so results are unchanged). The legacy loop is kept only for an
   output that partially overlaps its input. */
#define LAB_IMPLEMENT_BASIC_FUNCTION(NAME, CFUNC)             \
  TH_ITER_KERNEL2(NAME##_iterLoop, CFUNC(x))                  \
  void THTensor_(NAME)(THTensor *r_, THTensor *t)                \
  {                                                           \
    THTensor_(resizeAs)(r_, t);                               \
    if (!THTensor_(iterate2)(r_, t, THTensor_(NAME##_iterLoop), NULL)) \
      TH_TENSOR_APPLY2(real, t, real, r_, *r__data = CFUNC(*t_data);); \
  }                                                           \

#if defined(TH_REAL_IS_LONG)
LAB_IMPLEMENT_BASIC_FUNCTION(abs,llabs)
LAB_IMPLEMENT_BASIC_FUNCTION(neg,-)
#endif /* Long-only part */

#if defined(TH_REAL_IS_SHORT) || defined(TH_REAL_IS_INT)
LAB_IMPLEMENT_BASIC_FUNCTION(abs,abs)
LAB_IMPLEMENT_BASIC_FUNCTION(neg,-)
#endif /* int only part */

#if defined(TH_REAL_IS_BYTE)

#define TENSOR_IMPLEMENT_LOGICAL_SUM(NAME, OP, INIT_VALUE) \
  int THTensor_(NAME)(THTensor *tensor) \
  { \
    THArgCheck(tensor->nDimension > 0, 1, "empty Tensor"); \
    int sum = INIT_VALUE;                               \
    TH_TENSOR_APPLY(real, tensor, sum = sum OP *tensor_data;); \
    return sum; \
  }

TENSOR_IMPLEMENT_LOGICAL_SUM(logicalall, &&, 1)
TENSOR_IMPLEMENT_LOGICAL_SUM(logicalany, ||, 0)

#endif /* Byte only part */

/* floating point only now */
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)

#if defined (TH_REAL_IS_FLOAT)
#define TH_MATH_NAME(fn) fn##f
#else
#define TH_MATH_NAME(fn) fn
#endif

LAB_IMPLEMENT_BASIC_FUNCTION(log,TH_MATH_NAME(log))
LAB_IMPLEMENT_BASIC_FUNCTION(lgamma,TH_MATH_NAME(lgamma))
LAB_IMPLEMENT_BASIC_FUNCTION(log1p,TH_MATH_NAME(log1p))
LAB_IMPLEMENT_BASIC_FUNCTION(sigmoid,TH_MATH_NAME(TH_sigmoid))
LAB_IMPLEMENT_BASIC_FUNCTION(exp,TH_MATH_NAME(exp))
LAB_IMPLEMENT_BASIC_FUNCTION(cos,TH_MATH_NAME(cos))
LAB_IMPLEMENT_BASIC_FUNCTION(acos,TH_MATH_NAME(acos))
LAB_IMPLEMENT_BASIC_FUNCTION(cosh,TH_MATH_NAME(cosh))
LAB_IMPLEMENT_BASIC_FUNCTION(sin,TH_MATH_NAME(sin))
LAB_IMPLEMENT_BASIC_FUNCTION(asin,TH_MATH_NAME(asin))
LAB_IMPLEMENT_BASIC_FUNCTION(sinh,TH_MATH_NAME(sinh))
LAB_IMPLEMENT_BASIC_FUNCTION(tan,TH_MATH_NAME(tan))
LAB_IMPLEMENT_BASIC_FUNCTION(atan,TH_MATH_NAME(atan))
LAB_IMPLEMENT_BASIC_FUNCTION(tanh,TH_MATH_NAME(tanh))
LAB_IMPLEMENT_BASIC_FUNCTION(sqrt,TH_MATH_NAME(sqrt))
LAB_IMPLEMENT_BASIC_FUNCTION(rsqrt,TH_MATH_NAME(TH_rsqrt))
LAB_IMPLEMENT_BASIC_FUNCTION(ceil,TH_MATH_NAME(ceil))
LAB_IMPLEMENT_BASIC_FUNCTION(floor,TH_MATH_NAME(floor))
LAB_IMPLEMENT_BASIC_FUNCTION(round,TH_MATH_NAME(round))
LAB_IMPLEMENT_BASIC_FUNCTION(abs,TH_MATH_NAME(fabs))
LAB_IMPLEMENT_BASIC_FUNCTION(trunc,TH_MATH_NAME(trunc))
LAB_IMPLEMENT_BASIC_FUNCTION(frac,TH_MATH_NAME(TH_frac))
LAB_IMPLEMENT_BASIC_FUNCTION(neg,-)
LAB_IMPLEMENT_BASIC_FUNCTION(cinv, TH_MATH_NAME(1.0) / )


void THTensor_(pow)(THTensor *r_, THTensor *t, real value)
{
  THTensor_(resizeAs)(r_, t);
  if(value == 1){
    THTensor_(copy)(r_, t);
  }
  else if(value == 2){
    THTensor_(cmul)(r_, t, t);
  }
  else if(value == 3){
    TH_TENSOR_APPLY2(real, t, real, r_, *r__data = *t_data * *t_data * *t_data;);
  }
  else if(value == 0.5){
    THTensor_(sqrt)(r_, t);
  }
  else if(value == -0.5){
    THTensor_(rsqrt)(r_, t);
  }
  else if(value == -1){
    THTensor_(cinv)(r_, t);
  }
  else if(value == -2){
    TH_TENSOR_APPLY2(real, t, real, r_, *r__data = TH_MATH_NAME(1.0) / (*t_data * *t_data););
  }
  else{
    TH_TENSOR_APPLY2(real, t, real, r_, *r__data = TH_MATH_NAME(pow)(*t_data, value););
  }
}

void THTensor_(atan2)(THTensor *r_, THTensor *tx, THTensor *ty)
{
  THTensor_(resizeAs)(r_, tx);
  TH_TENSOR_APPLY3(real, r_, real, tx, real, ty, *r__data = TH_MATH_NAME(atan2)(*tx_data,*ty_data););
}

void THTensor_(lerp)(THTensor *r_, THTensor *a, THTensor *b, real weight)
{
  THArgCheck(THTensor_(nElement)(a) == THTensor_(nElement)(b), 2, "sizes do not match");
  THTensor_(resizeAs)(r_, a);
  TH_TENSOR_APPLY3(real, r_, real, a, real, b, *r__data = TH_MATH_NAME(TH_lerp)(*a_data, *b_data, weight););
}

void THTensor_(mean)(THTensor *r_, THTensor *t, int dimension, int keepdim)
{
  THArgCheck(dimension >= 0 && dimension < THTensor_(nDimension)(t), 2, "invalid dimension %d",
      dimension + TH_INDEX_BASE);

  THTensor_(sum)(r_, t, dimension, keepdim);
  THTensor_(div)(r_, r_, t->size[dimension]);
}

void THTensor_(std)(THTensor *r_, THTensor *t, int dimension, int biased, int keepdim)
{
  THLongStorage *dim;

  THArgCheck(dimension >= 0 && dimension < THTensor_(nDimension)(t), 3, "invalid dimension %d",
      dimension + TH_INDEX_BASE);

  dim = THTensor_(newSizeOf)(t);
  THLongStorage_set(dim, dimension, 1);
  THTensor_(resize)(r_, dim, NULL);
  THLongStorage_free(dim);

  TH_TENSOR_DIM_APPLY2(real, t, real, r_, dimension,
                       accreal sum = 0;
                       accreal sum2 = 0;
                       int64_t i;
                       for(i = 0; i < t_size; i++)
                       {
                         real z = t_data[i*t_stride];
                         sum += z;
                         sum2 += z*z;
                       }

                       if(biased)
                       {
                         sum /= t_size;
                         sum2 /= t_size;
                         sum2 -= sum*sum;
                         sum2 = (sum2 < 0 ? 0 : sum2);
                         *r__data = (real)TH_MATH_NAME(sqrt)(sum2);
                       }
                       else
                       {
                         sum /= t_size;
                         sum2 /= t_size-1;
                         sum2 -= ((real)t_size)/((real)(t_size-1))*sum*sum;
                         sum2 = (sum2 < 0 ? 0 : sum2);
                         *r__data = (real)TH_MATH_NAME(sqrt)(sum2);
                       });

  if (!keepdim) {
    THTensor_(squeeze1d)(r_, r_, dimension);
  }
}

void THTensor_(var)(THTensor *r_, THTensor *t, int dimension, int biased, int keepdim)
{
  THLongStorage *dim;

  THArgCheck(dimension >= 0 && dimension < THTensor_(nDimension)(t), 3, "invalid dimension %d",
      dimension + TH_INDEX_BASE);

  dim = THTensor_(newSizeOf)(t);
  THLongStorage_set(dim, dimension, 1);
  THTensor_(resize)(r_, dim, NULL);
  THLongStorage_free(dim);

  TH_TENSOR_DIM_APPLY2(real, t, real, r_, dimension,
                       accreal sum = 0;
                       accreal sum2 = 0;
                       int64_t i;
                       for(i = 0; i < t_size; i++)
                       {
                         real z = t_data[i*t_stride];
                         sum += z;
                         sum2 += z*z;
                       }

                       if(biased)
                       {
                         sum /= t_size;
                         sum2 /= t_size;
                         sum2 -= sum*sum;
                         sum2 = (sum2 < 0 ? 0 : sum2);
                         *r__data = sum2;
                       }
                       else
                       {
                         sum /= t_size;
                         sum2 /= t_size-1;
                         sum2 -= ((real)t_size)/((real)(t_size-1))*sum*sum;
                         sum2 = (sum2 < 0 ? 0 : sum2);
                         *r__data = (real)sum2;
                       });

  if (!keepdim) {
    THTensor_(squeeze1d)(r_, r_, dimension);
  }
}

void THTensor_(norm)(THTensor *r_, THTensor *t, real value, int dimension, int keepdim)
{
  THLongStorage *dim;

  THArgCheck(dimension >= 0 && dimension < THTensor_(nDimension)(t), 3, "invalid dimension %d",
      dimension + TH_INDEX_BASE);

  dim = THTensor_(newSizeOf)(t);
  THLongStorage_set(dim, dimension, 1);
  THTensor_(resize)(r_, dim, NULL);
  THLongStorage_free(dim);

  if(value == 0) {
    TH_TENSOR_DIM_APPLY2(real, t, real, r_, dimension,
                         accreal sum = 0;
                         int64_t i;
                         for(i = 0; i < t_size; i++)
                           sum += t_data[i*t_stride] != 0.0;
                         *r__data = sum;)
  } else {
    TH_TENSOR_DIM_APPLY2(real, t, real, r_, dimension,
                         accreal sum = 0;
                         int64_t i;
                         for(i = 0; i < t_size; i++) {
                           sum += TH_MATH_NAME(pow)(
                             TH_MATH_NAME(fabs)(t_data[i*t_stride]), value);
                         }
                         *r__data = TH_MATH_NAME(pow)(sum, 1.0/value);)
  }

  if (!keepdim) {
    THTensor_(squeeze1d)(r_, r_, dimension);
  }
}

accreal THTensor_(normall)(THTensor *tensor, real value)
{
  accreal sum = 0;
  if(value == 0) {
    TH_TENSOR_APPLY(real, tensor, sum += *tensor_data != 0.0;);
    return sum;
  } else if(value == 1) {
    TH_TENSOR_APPLY(real, tensor, sum += TH_MATH_NAME(fabs)(*tensor_data););
    return sum;
  } else if(value == 2) {
    TH_TENSOR_APPLY(real, tensor, accreal z = *tensor_data; sum += z*z;);
    return sqrt(sum);
  } else {
    TH_TENSOR_APPLY(real, tensor, sum += TH_MATH_NAME(pow)(TH_MATH_NAME(fabs)(*tensor_data), value););
    return TH_MATH_NAME(pow)(sum, 1.0/value);
  }
}

void THTensor_(renorm)(THTensor *res, THTensor *src, real value, int dimension, real maxnorm)
{
  int i;
  THTensor *rowR, *rowS;

  THArgCheck(dimension >= 0 && dimension < THTensor_(nDimension)(src), 3, "invalid dimension %d",
      dimension + TH_INDEX_BASE);
  THArgCheck(value > 0, 2, "non-positive-norm not supported");
  THArgCheck(THTensor_(nDimension)(src) > 1, 1, "need at least 2 dimensions, got %d dimensions",
      THTensor_(nDimension)(src));

  rowR = THTensor_(new)();
  rowS = THTensor_(new)();

  THTensor_(resizeAs)(res, src);

  for (i=0; i<src->size[dimension]; i++)
  {
    real norm = 0;
    real new_norm;

    THTensor_(select)(rowS, src, dimension, i);
    THTensor_(select)(rowR, res, dimension, i);
    if (value == 1) {
      TH_TENSOR_APPLY(real, rowS, norm += fabs(*rowS_data););
    } else if (value == 2) {
      TH_TENSOR_APPLY(real, rowS, accreal z = *rowS_data; norm += z*z;);
    } else {
      TH_TENSOR_APPLY(real, rowS, norm += TH_MATH_NAME(pow)(TH_MATH_NAME(fabs)(*rowS_data), value););
    }

    norm = pow(norm, 1/value);

    if (norm > maxnorm)
    {
      new_norm = maxnorm / (norm + 1e-7);

      TH_TENSOR_APPLY2(
        real, rowR, real, rowS,
        *rowR_data = (*rowS_data) * new_norm;
      )
    }
    else
      THTensor_(copy)(rowR, rowS);
  }

  THTensor_(free)(rowR);
  THTensor_(free)(rowS);
}

accreal THTensor_(dist)(THTensor *tensor, THTensor *src, real value)
{
  real sum = 0;
  TH_TENSOR_APPLY2(real, tensor, real, src,
                   sum += TH_MATH_NAME(pow)(
                     TH_MATH_NAME(fabs)(*tensor_data - *src_data), value););
  return TH_MATH_NAME(pow)(sum, 1.0/value);
}

accreal THTensor_(meanall)(THTensor *tensor)
{
  THArgCheck(tensor->nDimension > 0, 1, "empty Tensor");
  return THTensor_(sumall)(tensor)/THTensor_(nElement)(tensor);
}

accreal THTensor_(varall)(THTensor *tensor, int biased)
{
  accreal mean = THTensor_(meanall)(tensor);
  accreal sum = 0;
  TH_TENSOR_APPLY(real, tensor, sum += (*tensor_data - mean)*(*tensor_data - mean););
  sum /= THTensor_(nElement)(tensor) - (biased ? 0 : 1);
  return sum;
}

accreal THTensor_(stdall)(THTensor *tensor, int biased)
{
  return sqrt(THTensor_(varall)(tensor, biased));
}

void THTensor_(linspace)(THTensor *r_, real a, real b, int64_t n)
{
  real i = 0;

  THArgCheck(n > 1 || (n == 1 && (a == b)), 3, "invalid number of points");

  if (THTensor_(nElement)(r_) != n) {
    THTensor_(resize1d)(r_, n);
  }

  if(n == 1) {
    THTensor_(set1d)(r_, 0, a);
  } else {
     TH_TENSOR_APPLY(real, r_,
             *r__data = a + i*(b-a)/((real)(n-1));
             i++;
           );
  }
}

void THTensor_(logspace)(THTensor *r_, real a, real b, int64_t n)
{
  real i = 0;

  THArgCheck(n > 1 || (n == 1 && (a == b)), 3, "invalid number of points");

  if (THTensor_(nElement)(r_) != n) {
    THTensor_(resize1d)(r_, n);
  }

  if(n == 1) {
    THTensor_(set1d)(r_, 0, TH_MATH_NAME(pow)(10.0, a));
  } else {
    TH_TENSOR_APPLY(real, r_,
        *r__data = TH_MATH_NAME(pow)(10.0, a + i*(b-a)/((real)(n-1)));
        i++;
        );
  }
}

void THTensor_(rand)(THTensor *r_, THGenerator *_generator, THLongStorage *size)
{
  THTensor_(resize)(r_, size, NULL);
  THTensor_(uniform)(r_, _generator, 0, 1);
}

void THTensor_(randn)(THTensor *r_, THGenerator *_generator, THLongStorage *size)
{
  THTensor_(resize)(r_, size, NULL);
  THTensor_(normal)(r_, _generator, 0, 1);
}

void THTensor_(histc)(THTensor *hist, THTensor *tensor, int64_t nbins, real minvalue, real maxvalue)
{
  real minval;
  real maxval;
  real *h_data;

  THTensor_(resize1d)(hist, nbins);
  THTensor_(zero)(hist);
  minval = minvalue;
  maxval = maxvalue;
  if (minval == maxval)
  {
    minval = THTensor_(minall)(tensor);
    maxval = THTensor_(maxall)(tensor);
  }
  if (minval == maxval)
  {
    minval = minval - 1;
    maxval = maxval + 1;
  }

  h_data = THTensor_(data)(hist);

  TH_TENSOR_APPLY(real, tensor,
    if (*tensor_data >= minval && *tensor_data <= maxval) {
      const int bin = (int)((*tensor_data-minval) / (maxval-minval) * nbins);
      h_data[THMin(bin, nbins-1)] += 1;
    }
  );
}

void THTensor_(bhistc)(THTensor *hist, THTensor *tensor, int64_t nbins, real minvalue, real maxvalue)
{
  THArgCheck(THTensor_(nDimension)(tensor) < 3, 2, "invalid dimension %d, the input must be a 2d tensor", THTensor_(nDimension)(tensor));

  int dimension = 1;
  THArgCheck(dimension >= 0 && dimension < THTensor_(nDimension)(tensor), 2, "invalid dimension %d",
      dimension + TH_INDEX_BASE);

  real minval;
  real maxval;
  real *h_data;

  THTensor_(resize2d)(hist, tensor->size[0], nbins);
  THTensor_(zero)(hist);

  minval = minvalue;
  maxval = maxvalue;
  if (minval == maxval)
  {
    minval = THTensor_(minall)(tensor);
    maxval = THTensor_(maxall)(tensor);
  }
  if (minval == maxval)
  {
    minval = minval - 1;
    maxval = maxval + 1;
  }

  TH_TENSOR_DIM_APPLY2(real, tensor, real, hist, dimension, int64_t i;
                        for(i = 0; i < tensor_size; i++)
                        {
                          if(tensor_data[i*tensor_stride] >= minval && tensor_data[i*tensor_stride] <= maxval) {
                            const int bin = (int)((tensor_data[i*tensor_stride]-minval) / (maxval-minval) * nbins);
                            hist_data[THMin(bin, nbins-1)] += 1;
                          }
                        }
  );
}

#undef TH_MATH_NAME
#endif /* floating point only part */
#undef IS_NONZERO
#endif
