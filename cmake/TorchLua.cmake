# Locate the Lua interpreter, headers and (on Windows) import library.
#
# luarocks passes LUA, LUA_BINDIR, LUA_INCDIR, LUA_LIBDIR, LUADIR, LIBDIR and
# LUALIB explicitly; anything it does not pass is discovered here so that a
# plain `cmake -S . -B build` also works. LuaJIT is preferred.
include_guard(GLOBAL)

if(NOT LUA)
  find_program(LUA NAMES luajit lua5.1 lua51 lua REQUIRED)
endif()
get_filename_component(_torch_lua_bindir "${LUA}" DIRECTORY)
get_filename_component(_torch_lua_prefix "${_torch_lua_bindir}" DIRECTORY)

execute_process(
  COMMAND "${LUA}" -e "io.write((_VERSION:match('%d+%.%d+')))"
  OUTPUT_VARIABLE TORCH_LUA_VERSION
  RESULT_VARIABLE _torch_lua_rc)
if(NOT _torch_lua_rc EQUAL 0 OR TORCH_LUA_VERSION STREQUAL "")
  message(FATAL_ERROR "Could not run Lua interpreter '${LUA}'")
endif()

if(NOT LUA_BINDIR)
  set(LUA_BINDIR "${_torch_lua_bindir}")
endif()
if(NOT LUA_INCDIR)
  find_path(LUA_INCDIR lua.h
    HINTS "${_torch_lua_prefix}/include"
    PATH_SUFFIXES luajit-2.1 luajit-2.0 lua${TORCH_LUA_VERSION} lua
    REQUIRED)
endif()
if(NOT LUA_LIBDIR)
  set(LUA_LIBDIR "${_torch_lua_prefix}/lib")
endif()
if(NOT LUADIR)
  set(LUADIR "${_torch_lua_prefix}/share/lua/${TORCH_LUA_VERSION}")
endif()
if(NOT LIBDIR)
  set(LIBDIR "${_torch_lua_prefix}/lib/lua/${TORCH_LUA_VERSION}")
endif()

# On Windows, Lua C modules must link against the Lua DLL. On other platforms
# the symbols come from the host executable, so we do not link liblua.
if(WIN32 AND NOT LUALIB)
  find_library(LUALIB NAMES lua51 luajit-5.1 lua${TORCH_LUA_VERSION} lua
    HINTS "${LUA_LIBDIR}" "${LUA_BINDIR}"
    REQUIRED)
endif()

message(STATUS "Lua ${TORCH_LUA_VERSION}: ${LUA} (include: ${LUA_INCDIR})")
