#ifndef TH_GENERIC_FILE
#error "You must define TH_GENERIC_FILE before including THGenerateReducedTypes.h"
#endif

/* Reduced-precision floating types: stored in 16 bits, computed in float
 * (accreal = float). Each defines TH_REAL_IS_FLOATING and TH_REAL_IS_REDUCED. */
#ifndef THGenerateManyTypes
#define THReducedLocalGenerateManyTypes
#define THGenerateManyTypes
#endif

#include "THGenerateHalfType.h"
#include "THGenerateBFloat16Type.h"

#ifdef THReducedLocalGenerateManyTypes
#undef THReducedLocalGenerateManyTypes
#undef THGenerateManyTypes
#undef TH_GENERIC_FILE
#endif
