#ifndef TORCH_GENERAL_INC
#define TORCH_GENERAL_INC

#include <stdlib.h>
#include <string.h>
#include <stddef.h>

#include "luaT.h"
#include "TH.h"

/* Sizes and LongTensor values are int64_t. Lua 5.1/LuaJIT's lauxlib defines
   these macros with a cast to C `long`, which is only 32 bits on Windows, and
   Lua >= 5.3 removed them, so define them uniformly. lua_Integer is ptrdiff_t
   (64-bit) on 64-bit targets. */
#undef luaL_optlong
#undef luaL_checklong
#define luaL_optlong(L,n,d)     ((int64_t)luaL_optinteger(L, (n), (d)))
#define luaL_checklong(L,n)     ((int64_t)luaL_checkinteger(L, (n)))
#if LUA_VERSION_NUM >= 503
#define luaL_checkint(L,n)      ((int)luaL_checkinteger(L, (n)))
#endif

#endif
