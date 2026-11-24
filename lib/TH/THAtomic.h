#ifndef TH_ATOMIC_INC
#define TH_ATOMIC_INC

#include "THGeneral.h"

/******************************************************************************
 * Atomic operations for TH
 *  Implemented with C11 <stdatomic.h> (GCC is the only supported compiler).
 ******************************************************************************/


/******************************************************************************
 * all-purpose functions
 ******************************************************************************/

/*
 * *a = newvalue
*/
TH_API void THAtomicSet(int volatile *a, int newvalue);

/*
 * return *a
*/
TH_API int THAtomicGet(int volatile *a);

/*
 * *a += value,
 * return previous *a
*/
TH_API int THAtomicAdd(int volatile *a, int value);

/*
 * check if (*a == oldvalue)
 * if true: set *a to newvalue, return 1
 * if false: return 0
*/
TH_API int THAtomicCompareAndSwap(int volatile *a, int oldvalue, int newvalue);


/******************************************************************************
 * refcounting functions
 ******************************************************************************/

/*
 * *a++
*/
TH_API void THAtomicIncrementRef(int volatile *a);

/*
 * *a--,
 * return 1 if *a == 0 after the operation, 0 otherwise
*/
TH_API int THAtomicDecrementRef(int volatile *a);



/******************************************************************************
 * functions for long type
 ******************************************************************************/

/*
 * *a = newvalue
*/
TH_API void THAtomicSetLong(long volatile *a, long newvalue);

/*
 * return *a
*/
TH_API long THAtomicGetLong(long volatile *a);

/*
 * *a += value,
 * return previous *a
*/
TH_API long THAtomicAddLong(long volatile *a, long value);

/*
 * check if (*a == oldvalue)
 * if true: set *a to newvalue, return 1
 * if false: return 0
*/
TH_API long THAtomicCompareAndSwapLong(long volatile *a, long oldvalue, long newvalue);



/******************************************************************************
 * functions for ptrdiff_t type
 ******************************************************************************/

/*
 * *a = newvalue
*/
TH_API void THAtomicSetPtrdiff(ptrdiff_t volatile *a, ptrdiff_t newvalue);

/*
 * return *a
*/
TH_API ptrdiff_t THAtomicGetPtrdiff(ptrdiff_t volatile *a);

/*
 * *a += value,
 * return previous *a
*/
TH_API ptrdiff_t THAtomicAddPtrdiff(ptrdiff_t volatile *a, ptrdiff_t value);

/*
 * check if (*a == oldvalue)
 * if true: set *a to newvalue, return 1
 * if false: return 0
*/
TH_API ptrdiff_t THAtomicCompareAndSwapPtrdiff(ptrdiff_t volatile *a, ptrdiff_t oldvalue, ptrdiff_t newvalue);

/* NOTE: ATOMIC_INT_LOCK_FREE comes from <stdatomic.h>, which is not included
   here, so this is currently never defined and the refcounted IPC allocator in
   THAllocator.c is compiled out. Kept as-is to preserve existing behavior. */
#if defined(USE_C11_ATOMICS) && defined(ATOMIC_INT_LOCK_FREE) && \
  ATOMIC_INT_LOCK_FREE == 2
#define TH_ATOMIC_IPC_REFCOUNT 1
#endif

#endif
