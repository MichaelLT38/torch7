/* Tests for THIter (lib/TH/THIter.c), included into test_th.c as suite "iter". */

static THIterOperand iter_op_float(THFloatTensor *t)
{
  THIterOperand op;
  op.data = (char *)THFloatTensor_data(t);
  op.elsize = sizeof(float);
  op.ndim = THFloatTensor_nDimension(t);
  op.size = t->size;
  op.stride = t->stride;
  return op;
}

/* out = a + b, typed for float */
static void iter_add_loop(char **data, const int64_t *strides, int64_t n, void *ctx)
{
  (void)ctx;
  for (int64_t i = 0; i < n; i++)
    *(float *)(data[0] + i * strides[0]) =
      *(float *)(data[1] + i * strides[1]) + *(float *)(data[2] + i * strides[2]);
}

/* records how many inner calls / elements were made (serial use only) */
typedef struct { int64_t calls, elems, maxn; } iter_count_ctx;
static void iter_count_loop(char **data, const int64_t *strides, int64_t n, void *ctx)
{
  (void)data; (void)strides;
  iter_count_ctx *c = ctx;
  c->calls++; c->elems += n;
  if (n > c->maxn) c->maxn = n;
}

static void iter_broadcast_shapes(void)
{
  int64_t sa[3] = {4, 1, 3}, sb[2] = {5, 1}, st[3] = {0, 0, 0};
  THIterOperand ops[2] = {{NULL, 4, 3, sa, st}, {NULL, 4, 2, sb, st}};
  int nd = 0;
  int64_t shape[TH_ITER_MAX_DIMS];
  char err[512];
  CHECK(THIter_broadcastShape(2, ops, &nd, shape, err, sizeof(err)), "broadcast (4,1,3)+(5,1): %s", err);
  CHECK(nd == 3 && shape[0] == 4 && shape[1] == 5 && shape[2] == 3,
        "(4,1,3)+(5,1) -> (4,5,3), got nd=%d (%" PRId64 ",%" PRId64 ",%" PRId64 ")",
        nd, shape[0], shape[1], shape[2]);

  int64_t sc[2] = {2, 8}, sd[2] = {4, 4};
  THIterOperand bad[2] = {{NULL, 4, 2, sc, st}, {NULL, 4, 2, sd, st}};
  CHECK(!THIter_broadcastShape(2, bad, &nd, shape, err, sizeof(err)), "(2,8)+(4,4) must not broadcast");
  CHECK(strstr(err, "not broadcastable") != NULL, "error message: %s", err);
}

static void iter_coalescing(void)
{
  THIter it;
  /* contiguous 3-D: one dimension of 24 */
  THFloatTensor *a = THFloatTensor_newWithSize3d(2, 3, 4);
  THIterOperand ops[2] = {iter_op_float(a), iter_op_float(a)};
  THIter_build(&it, 2, ops);
  CHECK(it.ndim == 1 && it.shape[0] == 24 && it.stride[0][0] == 4, "contiguous 2x3x4 -> 1 dim of 24, got ndim %d", it.ndim);

  /* transposed output and input with the same layout: still one dimension,
     walked in memory order (positive inner stride of 4 bytes) */
  THFloatTensor *t = THFloatTensor_newTranspose(a, 0, 2);   /* 4x3x2 */
  THIterOperand tops[2] = {iter_op_float(t), iter_op_float(t)};
  THIter_build(&it, 2, tops);
  CHECK(it.ndim == 1 && it.shape[0] == 24 && it.stride[0][0] == 4,
        "permuted but dense -> 1 dim in memory order, got ndim %d stride %" PRId64, it.ndim, it.stride[0][0]);

  /* contiguous output, transposed input: cannot coalesce */
  THFloatTensor *o = THFloatTensor_newWithSize3d(4, 3, 2);
  THIterOperand mops[2] = {iter_op_float(o), iter_op_float(t)};
  THIter_build(&it, 2, mops);
  CHECK(it.ndim == 3 && it.stride[0][0] == 4, "contiguous out + permuted in -> 3 dims, got %d", it.ndim);

  /* broadcast input along dim 0: stride 0 there, and size-1 dims dropped */
  THFloatTensor *m = THFloatTensor_newWithSize2d(5, 7), *row = THFloatTensor_newWithSize2d(1, 7);
  THIterOperand bops[2] = {iter_op_float(m), iter_op_float(row)};
  THIter_build(&it, 2, bops);
  CHECK(it.ndim == 2 && it.shape[0] == 7 && it.shape[1] == 5 && it.stride[1][1] == 0 && it.stride[0][1] == 4,
        "row broadcast: inner 7 (stride 4), outer 5 (input stride 0), got ndim %d", it.ndim);

  /* single element */
  THFloatTensor *one = THFloatTensor_newWithSize3d(1, 1, 1);
  THIterOperand oops[1] = {iter_op_float(one)};
  THIter_build(&it, 1, oops);
  CHECK(it.ndim == 1 && it.shape[0] == 1 && it.numel == 1, "1x1x1 -> one element");

  /* shape errors */
  THFloatTensor *w = THFloatTensor_newWithSize2d(5, 6);
  THIterOperand eops[2] = {iter_op_float(m), iter_op_float(w)};
  EXPECT_TH_ERROR(THIter_build(&it, 2, eops), "cannot be broadcast");

  THFloatTensor_free(a); THFloatTensor_free(t); THFloatTensor_free(o); THFloatTensor_free(m);
  THFloatTensor_free(row); THFloatTensor_free(one); THFloatTensor_free(w);
}

/* a + b against a reference loop, for every pair of input layouts, plus
   row/column broadcasting; serial and parallel runs must be identical. */
static void iter_add_correctness(void)
{
  uint64_t rng = 21;
  const int64_t shapes[][2] = {{1, 1}, {3, 5}, {64, 33}, {700, 300}};
  for (int s = 0; s < 4; s++) {
    const int64_t m = shapes[s][0], n = shapes[s][1];
    for (int ka = 0; ka < TT_NUM_IN_LAYOUTS; ka++)
    for (int kb = 0; kb < TT_NUM_IN_LAYOUTS; kb++)
    for (int ko = 0; ko < 2; ko++) {      /* contiguous or transposed output */
      THFloatTensor *a = test_Float_make(ka, m, n, &rng, -1, 1);
      THFloatTensor *b = test_Float_make(kb, m, n, &rng, -1, 1);
      THFloatTensor *o = test_Float_make(ko ? L_TRANSPOSED : L_CONTIG, m, n, &rng, 0, 0);
      THFloatTensor *o2 = THFloatTensor_newClone(o);
      THIterOperand ops[3] = {iter_op_float(o), iter_op_float(a), iter_op_float(b)};
      THIter it;
      THIter_build(&it, 3, ops);
      THIter_run(&it, iter_add_loop, NULL, 1);          /* parallel whenever possible */
      THIterOperand ops2[3] = {iter_op_float(o2), iter_op_float(a), iter_op_float(b)};
      THIter_build(&it, 3, ops2);
      THIter_run(&it, iter_add_loop, NULL, INT64_MAX);  /* serial */
      int ok = 1;
      for (int64_t i = 0; i < m && ok; i++)
      for (int64_t j = 0; j < n && ok; j++) {
        float want = *test_Float_at(a, i, j) + *test_Float_at(b, i, j);
        ok = *test_Float_at(o, i, j) == want && *test_Float_at(o2, i, j) == want;
        CHECK(ok, "iter add %" PRId64 "x%" PRId64 " a=%s b=%s out=%s at (%" PRId64 ",%" PRId64 ")",
              m, n, tt_layout_name(ka), tt_layout_name(kb), ko ? "transposed" : "contig", i, j);
      }
      THFloatTensor_free(a); THFloatTensor_free(b); THFloatTensor_free(o); THFloatTensor_free(o2);
    }
  }

  /* genuine broadcasting: (m,1) + (1,n) -> (m,n) */
  const int64_t m = 300, n = 500;
  THFloatTensor *col = test_Float_make(L_CONTIG, m, 1, &rng, -1, 1);
  THFloatTensor *row = test_Float_make(L_CONTIG, 1, n, &rng, -1, 1);
  THFloatTensor *out = THFloatTensor_newWithSize2d(m, n);
  THIterOperand ops[3] = {iter_op_float(out), iter_op_float(col), iter_op_float(row)};
  THIter it;
  THIter_build(&it, 3, ops);
  THIter_run(&it, iter_add_loop, NULL, 1);
  int ok = 1;
  for (int64_t i = 0; i < m && ok; i++)
  for (int64_t j = 0; j < n && ok; j++) {
    ok = THFloatTensor_get2d(out, i, j) == THFloatTensor_get2d(col, i, 0) + THFloatTensor_get2d(row, 0, j);
    CHECK(ok, "broadcast (m,1)+(1,n) at (%" PRId64 ",%" PRId64 ")", i, j);
  }
  THFloatTensor_free(col); THFloatTensor_free(row); THFloatTensor_free(out);
}

/* every element visited exactly once; inner runs as long as possible */
static void iter_visits(void)
{
  THFloatTensor *a = THFloatTensor_newWithSize3d(7, 11, 13);
  THFloatTensor *t = THFloatTensor_newTranspose(a, 0, 1);       /* 11x7x13 view */
  THFloatTensor *o = THFloatTensor_newWithSize3d(11, 7, 13);
  THIterOperand ops[2] = {iter_op_float(o), iter_op_float(t)};
  THIter it;
  THIter_build(&it, 2, ops);
  iter_count_ctx c = {0, 0, 0};
  THIter_run(&it, iter_count_loop, &c, INT64_MAX);
  CHECK(c.elems == 7 * 11 * 13, "visited %" PRId64 " elements", c.elems);
  CHECK(c.maxn == 13 && c.calls == 7 * 11, "inner runs of 13 (calls %" PRId64 ", max %" PRId64 ")",
        c.calls, c.maxn);

  /* empty */
  THFloatTensor *e = THFloatTensor_new();
  THIterOperand eops[2] = {iter_op_float(e), iter_op_float(e)};
  THIter_build(&it, 2, eops);
  iter_count_ctx z = {0, 0, 0};
  THIter_run(&it, iter_count_loop, &z, 1);
  CHECK(it.numel == 0 && z.calls == 0, "empty: no calls");
  THFloatTensor_free(a); THFloatTensor_free(t); THFloatTensor_free(o); THFloatTensor_free(e);
}

static void iter_overlap_detection(void)
{
  THFloatTensor *a = THFloatTensor_newWithSize2d(4, 4);
  THFloatTensor *t = THFloatTensor_newTranspose(a, 0, 1);
  THFloatTensor *top = THFloatTensor_newNarrow(a, 0, 0, 2), *bot = THFloatTensor_newNarrow(a, 0, 2, 2);
  THIterOperand oa = iter_op_float(a), ot = iter_op_float(t), otop = iter_op_float(top), obot = iter_op_float(bot);
  CHECK(THIter_overlaps(&oa, &ot), "a and a^T overlap");
  CHECK(!THIter_sameLayout(&oa, &ot), "a and a^T differ in layout");
  CHECK(THIter_sameLayout(&oa, &oa), "a has the same layout as itself");
  CHECK(!THIter_overlaps(&otop, &obot), "disjoint halves do not overlap");
  THFloatTensor_free(a); THFloatTensor_free(t); THFloatTensor_free(top); THFloatTensor_free(bot);
}

static void iter_suite(void)
{
  RUN(iter_broadcast_shapes);
  RUN(iter_coalescing);
  RUN(iter_add_correctness);
  RUN(iter_visits);
  RUN(iter_overlap_detection);
}
