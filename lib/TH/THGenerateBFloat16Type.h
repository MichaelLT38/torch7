#ifndef TH_GENERIC_FILE
#error "You must define TH_GENERIC_FILE before including THGenerateBFloat16Type.h"
#endif

#include "THHalf.h"
#define real THBFloat16
#define accreal float
#define TH_CONVERT_REAL_TO_ACCREAL(_val) (accreal)(_val)
#define TH_CONVERT_ACCREAL_TO_REAL(_val) (real)(_val)
#define Real BFloat16
#define THInf TH_BFLOAT16_MAX
#define TH_REAL_IS_BFLOAT16
#define TH_REAL_IS_FLOATING
#define TH_REAL_IS_REDUCED
#line 1 TH_GENERIC_FILE
#include TH_GENERIC_FILE
#undef real
#undef accreal
#undef Real
#undef THInf
#undef TH_REAL_IS_BFLOAT16
#undef TH_REAL_IS_FLOATING
#undef TH_REAL_IS_REDUCED
#undef TH_CONVERT_REAL_TO_ACCREAL
#undef TH_CONVERT_ACCREAL_TO_REAL

#ifndef THGenerateManyTypes
#undef TH_GENERIC_FILE
#endif
