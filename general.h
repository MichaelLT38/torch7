#ifndef TORCH_GENERAL_INC
#define TORCH_GENERAL_INC

#include <stdlib.h>
#include <string.h>
#include <stddef.h>

#include "luaT.h"
#include "TH.h"

#if LUA_VERSION_NUM >= 503
/* one can simply enable LUA_COMPAT_5_2 to be backward compatible.
However, this does not work when we are trying to use system-installed lua,
hence these redefines
*/
#define luaL_optlong(L,n,d)     ((long)luaL_optinteger(L, (n), (d)))
#define luaL_checklong(L,n)     ((long)luaL_checkinteger(L, (n)))
#define luaL_checkint(L,n)      ((int)luaL_checkinteger(L, (n)))
#endif

#endif
