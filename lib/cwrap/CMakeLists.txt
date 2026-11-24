# cwrap: the Lua -> C binding generator (originally github.com/torch/cwrap,
# commit dbd0a62). Used at build time to generate TensorMath.c and random.c,
# and installed so that downstream packages (nn, ...) can use it too.

INSTALL(FILES init.lua cinterface.lua types.lua
  DESTINATION "${Torch_INSTALL_LUA_PATH_SUBDIR}/cwrap")
