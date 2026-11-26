#if !defined(real) || !defined(TH_GENERIC_FILE)
#error "luaG.h must not be included outside of a generic file."
#endif

#ifndef luaG_
#define luaG_(NAME) TH_CONCAT_3(luaG_,Real,NAME)
#endif

#undef REAL_TO_LUA_NUMBER
#undef LUA_NUMBER_TO_REAL

/* reduced-precision reals (THHalf) are native arithmetic types: one correctly rounded cast */
# define REAL_TO_LUA_NUMBER(n)   (lua_Number)(n)
# define LUA_NUMBER_TO_REAL(n)   (real)(n)



static void luaG_(pushreal)(lua_State *L, real n) {
#if defined(TH_REAL_IS_FLOATING) || LUA_VERSION_NUM < 503
  lua_pushnumber(L, REAL_TO_LUA_NUMBER(n));
#elif defined(TH_REAL_IS_BYTE) || defined(TH_REAL_IS_CHAR) || defined(TH_REAL_IS_SHORT) \
  || defined(TH_REAL_IS_INT) || defined(TH_REAL_IS_LONG)
	lua_pushinteger(L, (lua_Integer)n);
#else
	#error "unhandled real type in luaG_pushreal"
#endif
}

static real luaG_(checkreal)(lua_State *L, int idx) {
#if defined(TH_REAL_IS_FLOATING)
  return LUA_NUMBER_TO_REAL(luaL_checknumber(L, idx));
#elif defined(TH_REAL_IS_BYTE) || defined(TH_REAL_IS_CHAR) || defined(TH_REAL_IS_SHORT) || defined(TH_REAL_IS_INT) || defined(TH_REAL_IS_LONG)
        int type = lua_type(L, idx);
        if (type == LUA_TSTRING) {
          const char *str = lua_tolstring(L, idx, NULL);
          long int num = strtol(str, NULL, 0);
          return (real) num;
        } else {
#if LUA_VERSION_NUM < 503
          return (lua_Number)luaL_checkinteger(L, idx);
#else
          return (lua_Integer)luaL_checkinteger(L, idx);
#endif
        }
#else
	#error "unhandled real type in luaG_checkreal"
#endif
}

static real luaG_(optreal)(lua_State *L, int idx, real n) {
#if defined(TH_REAL_IS_FLOATING) || LUA_VERSION_NUM < 503
  return LUA_NUMBER_TO_REAL(luaL_optnumber(L, idx, REAL_TO_LUA_NUMBER(n)));
#elif defined(TH_REAL_IS_BYTE) || defined(TH_REAL_IS_CHAR) || defined(TH_REAL_IS_SHORT) || defined(TH_REAL_IS_INT) || defined(TH_REAL_IS_LONG)
	return (lua_Integer)luaL_optinteger(L, idx, (lua_Integer)n);
#else
	#error "unhandled real type in luaG_checkreal"
#endif
}
