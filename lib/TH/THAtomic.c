#include "THAtomic.h"

/*
  Note: I thank Leon Bottou for his useful comments.
  Ronan.
*/

/* Torch builds only with GCC >= 13, which always provides C11 atomics. The
   <stdatomic.h> generic functions accept plain (non-_Atomic) objects on GCC. */
#include <stdatomic.h>

void THAtomicSet(int volatile *a, int newvalue)
{
  atomic_store(a, newvalue);
}

int THAtomicGet(int volatile *a)
{
  return atomic_load(a);
}

int THAtomicAdd(int volatile *a, int value)
{
  return atomic_fetch_add(a, value);
}

void THAtomicIncrementRef(int volatile *a)
{
  THAtomicAdd(a, 1);
}

int THAtomicDecrementRef(int volatile *a)
{
  return (THAtomicAdd(a, -1) == 1);
}

int THAtomicCompareAndSwap(int volatile *a, int oldvalue, int newvalue)
{
  return atomic_compare_exchange_strong(a, &oldvalue, newvalue);
}

void THAtomicSetLong(long volatile *a, long newvalue)
{
  atomic_store(a, newvalue);
}

long THAtomicGetLong(long volatile *a)
{
  return atomic_load(a);
}

long THAtomicAddLong(long volatile *a, long value)
{
  return atomic_fetch_add(a, value);
}

long THAtomicCompareAndSwapLong(long volatile *a, long oldvalue, long newvalue)
{
  return atomic_compare_exchange_strong(a, &oldvalue, newvalue);
}

void THAtomicSetPtrdiff(ptrdiff_t volatile *a, ptrdiff_t newvalue)
{
  atomic_store(a, newvalue);
}

ptrdiff_t THAtomicGetPtrdiff(ptrdiff_t volatile *a)
{
  return atomic_load(a);
}

ptrdiff_t THAtomicAddPtrdiff(ptrdiff_t volatile *a, ptrdiff_t value)
{
  return atomic_fetch_add(a, value);
}

ptrdiff_t THAtomicCompareAndSwapPtrdiff(ptrdiff_t volatile *a, ptrdiff_t oldvalue, ptrdiff_t newvalue)
{
  return atomic_compare_exchange_strong(a, &oldvalue, newvalue);
}
