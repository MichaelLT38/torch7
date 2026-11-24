cmake_minimum_required(VERSION 3.16)
project(Torch LANGUAGES C)

list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(TorchCompiler)  # GCC >= 13 only, C11, OpenMP
include(TorchLua)       # LUA, LUA_INCDIR, LUADIR, LIBDIR, LUALIB, ...

# Includes
INCLUDE(TorchPaths)
INCLUDE(TorchPathsInit)
INCLUDE(TorchPackage)
INCLUDE(TorchWrap)
INCLUDE(TorchExports)

# Torch libraries
ADD_SUBDIRECTORY(lib)

CONFIGURE_FILE(paths.lua.in "${CMAKE_CURRENT_BINARY_DIR}/paths.lua")

INCLUDE_DIRECTORIES(BEFORE "${LUA_INCDIR}")
INCLUDE_DIRECTORIES(BEFORE "${CMAKE_CURRENT_SOURCE_DIR}/lib/TH")
INCLUDE_DIRECTORIES(BEFORE "${CMAKE_CURRENT_BINARY_DIR}/lib/TH")
INCLUDE_DIRECTORIES(BEFORE "${CMAKE_CURRENT_SOURCE_DIR}/lib/luaT")
LINK_DIRECTORIES("${LUA_LIBDIR}")

SET(src DiskFile.c File.c MemoryFile.c PipeFile.c Storage.c Tensor.c Timer.c utils.c init.c TensorOperator.c TensorMath.c random.c Generator.c)
SET(luasrc init.lua File.lua Tensor.lua CmdLine.lua FFInterface.lua Tester.lua TestSuite.lua ${CMAKE_CURRENT_BINARY_DIR}/paths.lua test/test.lua)

# Necessary do generate wrapper
ADD_TORCH_WRAP(tensormathwrap TensorMath.lua)
ADD_TORCH_WRAP(randomwrap random.lua)

ADD_TORCH_PACKAGE(torch "${src}" "${luasrc}")

TARGET_LINK_LIBRARIES(torch luaT TH OpenMP::OpenMP_C)

IF(LUALIB)
  TARGET_LINK_LIBRARIES(torch ${LUALIB})
ENDIF()

INSTALL(FILES "README.md" DESTINATION "${Torch_INSTALL_LUA_PATH_SUBDIR}/torch")
INSTALL(DIRECTORY "doc" DESTINATION "${Torch_INSTALL_LUA_PATH_SUBDIR}/torch")
